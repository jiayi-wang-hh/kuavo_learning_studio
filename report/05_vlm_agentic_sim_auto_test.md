# Kuavo 仿真在线 VLM Agentic Trigger

## 实现目标

本实现把离线视频 `CONTINUE/PAUSE` trigger 接入 Kuavo closed-loop 仿真评测。
策略仍按原有流程读取 ROS 观测、生成动作并调用 `env.step()`；轻量 VLM 在后台读取最近的头部相机窗口。当 VLM 返回 `PAUSE` 时，评测端发布 `/kuavo/pause_state=True` 并暂停继续发送策略动作，等待后续人工恢复或接入第二阶段强 VLM。

第一阶段仅负责触发，不进行 failure type 分类、任务完成判断或 recovery 规划。

## 新增文件

- `kuavo_deploy/src/scripts/script_auto_test_vlm_agentic.py`
  - 新命令行入口。
  - 加载原有 Kuavo 配置和额外 VLM trigger 参数。
  - 复用原入口的 ROS 信号处理与暂停/恢复控制器。
- `kuavo_deploy/src/eval/sim_auto_test_vlm_agentic.py`
  - 新 closed-loop 评测循环。
  - `VLMTriggerAgent` 在独立 worker thread 中执行 VLM 推理。
  - 保存 trigger 输入视频、逐次判断记录和 episode 汇总。

现有 `script_auto_test.py`、`sim_auto_test.py` 和配置文件均未修改。

## 在线数据流

```text
ROS observation
   ├─> policy inference ─> env.step(action) ─> 下一帧 observation
   │
   └─> head camera ring buffer
           └─ 每 check_interval_steps 提交一次
                └─ background VLM trigger
                     ├─ CONTINUE：保持策略闭环
                     └─ PAUSE：发布 /kuavo/pause_state=True
                                  └─ 停止发送新动作并等待恢复
```

VLM 推理在后台执行，因此不会把每次约数秒的模型延迟直接加入每个策略 step。检测结果会记录源窗口 step 和主循环实际收到结果时的 step，可据此计算异步检测滞后。

## Trigger 输出

```json
{
  "trigger_decision": "CONTINUE | PAUSE",
  "confidence": "LOW | MEDIUM | HIGH",
  "evidence": "one short directly visible observation"
}
```

非法输出按 fail-safe 规则转换为 `PAUSE`。`--vlm-pause-confirmations` 控制连续多少次 PAUSE 后真正暂停。

## 启动方式

先按原流程启动：

1. `roscore`
2. `kuavo-ros-opensource` 自动测试仿真脚本
3. 使用远程策略时启动 `localhost:5555` 的 PolicyClient 服务

然后从仓库根目录运行：

```bash
python kuavo_deploy/src/scripts/script_auto_test_vlm_agentic.py \
  --config /path/to/config.yaml \
  --vlm-mode qwen25_vl_7b \
  --vlm-device-map cuda:1 \
  --vlm-camera-key observation.images.head_cam_h \
  --vlm-window-seconds 3 \
  --vlm-fps 4 \
  --vlm-check-interval-steps 10 \
  --vlm-pause-confirmations 1
```

当前机器的策略通常使用 `cuda:0`，默认将 VLM 放在 `cuda:1`。如果策略运行在其他设备，应相应调整 `--vlm-device-map`。

只验证配置和 ROS 入口、不开始 episode：

```bash
python kuavo_deploy/src/scripts/script_auto_test_vlm_agentic.py \
  --config /path/to/config.yaml \
  --dry-run
```

## 参数说明

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--vlm-mode` | `qwen25_vl_7b` | VLM 类型，沿用离线 trigger 的模型注册表 |
| `--vlm-model-path` | 模型注册表路径 | 自定义本地 checkpoint |
| `--vlm-camera-key` | `observation.images.head_cam_h` | trigger 使用的 observation 图像键 |
| `--vlm-window-seconds` | `3.0` | 环形视频窗口长度 |
| `--vlm-fps` | `4.0` | VLM 解码采样 FPS |
| `--vlm-check-interval-steps` | `10` | 每多少个 policy step 提交一次检测 |
| `--vlm-max-new-tokens` | `96` | trigger 最大输出 token 数 |
| `--vlm-device-map` | `cuda:1` | VLM 放置设备 |
| `--vlm-pause-confirmations` | `1` | 实际暂停前所需连续 PAUSE 数 |

当 `env.ros_rate=10` 且 `--vlm-check-interval-steps=10` 时，每秒提交一次检测；一个 worker 同时只执行一次推理，不积压旧窗口。

## 暂停与恢复

VLM 确认 PAUSE 后会同时更新本进程暂停状态并发布 ROS topic。终端会停在暂停检查循环，不再调用新的 `policy.select_action()` 或 `env.step()`。

入口启动时会打印 PID。人工恢复：

```bash
kill -USR1 <PID>
```

停止整个测试：

```bash
kill -USR2 <PID>
```

恢复后会清空旧视频窗口和 PAUSE streak，重新积累一个完整窗口，避免立刻使用暂停前的旧画面重复触发。

## 输出文件

输出位于原策略 eval 目录下的独立时间戳目录：

```text
outputs/eval/<policy-name>/vlm_agentic_YYYYmmdd_HHMMSS/
```

包含：

- `evaluation_autotest_vlm_agentic.log`：episode 成功情况和 VLM 配置。
- `vlm_trigger_events.jsonl`：每次完成的异步 trigger 推理结果。
- `vlm_agentic_episode_summary.csv`：每个 episode 的成功状态和暂停次数。
- `vlm_trigger_clips/*.mp4`：每次提交给 VLM 的原始窗口。
- `rollout_<episode>_<camera-key>.mp4`：原有各相机 rollout 视频。

`vlm_trigger_events.jsonl` 主要字段：

| 字段 | 含义 |
|---|---|
| `source_step` | 该视频窗口结束时的 policy step |
| `observed_step` | 主循环收到 VLM 结果时的 step |
| `step_lag` | 异步推理期间策略额外执行的 step 数 |
| `final_decision` | 最终 CONTINUE/PAUSE |
| `confidence` | 模型置信等级 |
| `evidence` | 模型给出的可见依据 |
| `inference_seconds` | VLM 端到端推理时间 |
| `clip_path` | 对应输入窗口，可用于复盘 |

## 当前边界

该文件只验证第一阶段轻量 VLM 是否能在真实 closed-loop inference 中及时触发暂停。PAUSE 后尚未自动调用第二阶段强模型；当前通过 SIGUSR1 人工恢复。后续可在暂停点读取 `clip_path`，把触发前窗口与暂停后的最新帧交给强 VLM，输出 `RESUME` 或 failure type 与 recovery action。

