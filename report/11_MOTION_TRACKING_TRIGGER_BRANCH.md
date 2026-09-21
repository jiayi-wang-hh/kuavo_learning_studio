# `motion_tracking_trigger` 分支说明

## 1. 分支目标

本分支在 `lingbot` 分支基础上，引入一套面向机器人操作任务的失败检测与评测流程。核心目标是解决纯视觉规则误报率高的问题：不再仅根据“玩具是否移动”或“玩具是否持续接近篮筐”判断失败，而是联合机器人运动状态与视觉观测，检查二者是否一致。

核心判断可概括为：

> 机器人正在执行明显运动，但对应侧的目标物体在可靠视觉观测下持续静止，则产生候选失败事件。

在此基础上，本分支还提供了可选的第二阶段 VLM 验证与恢复，以及用于比较不同检测器、分割器和跟踪器的离线基准。

## 2. 整体架构

```text
VLA policy / simulator
        │
        ├── head camera ──> YOLO-World ──> left/right toy tracking
        │                                      │
        └── EE PoseStamped / joint state ──────┤
                                               ↓
                                  cross-signal consistency check
                                               │
                              confirmed event ─┴─> publish PAUSE
                                               │
                                  optional Qwen stage-2 verifier
                                      ├── RESUME
                                      ├── RESET_AND_RETRY
                                      └── STOP
```

实现分为三条相互关联但可独立运行的路径：

1. **在线 Stage-1 触发**：仿真闭环中联合 YOLO-World 与 EE pose/关节状态，触发暂停。
2. **在线 Stage-2 恢复**：Stage-1 触发后生成视频片段，由 Qwen VLM 判断失败类型并选择恢复动作。
3. **离线跟踪与消融**：统一比较 YOLO-World、Grounding DINO、SAM 2.1、SAM 3 和 CSRT，分离初始化与时序跟踪的误差来源。

## 3. 在线失败触发

### 3.1 输入信号

- 视觉：`observation.images.head_cam_h`；
- 首选机器人状态：左右末端的 `geometry_msgs/PoseStamped`；
- 回退信号：`observation.state` 中按左右臂切分的关节与夹爪状态；
- 输出：配置项 `failure_trigger_pause_topic` 指定的 `std_msgs/Bool` 暂停消息，默认 topic 为 `/kuavo/pause_state`。

EE pose 与关节状态的语义不同：EE pose 使用世界/基座坐标系中的 XY 平面速度进行判断，并单独记录 Z 方向速度；关节状态仅作为“机器人是否运动”的代理量，不能解释为笛卡尔末端速度。

### 3.2 视觉跟踪

在线监控器复用 `yolo/test_yolo_world_roi_filter.py` 中的目标过滤、左右侧分配与时序跟踪逻辑。目标状态包括：

- `DETECTED`：当前帧由检测器可靠观测；
- `TRACKED`：当前帧由视觉跟踪器可靠更新；
- `PREDICTED`：仅由运动模型预测；
- `LOST` / `UNINITIALIZED`：目标不可用。

只有 `DETECTED` 和 `TRACKED` 会参与硬触发。预测或丢失状态会被视为视觉不确定，并清空当前候选持续时间，避免因遮挡直接暂停机器人。

### 3.3 触发条件

每一侧独立维护一个长度为 `failure_trigger_motion_window_s` 的时间窗口。窗口预热完成后，触发器比较：

- 机器人运动是否超过 `failure_trigger_ee_motion_min`；
- 玩具归一化图像位移是否不超过 `failure_trigger_toy_motion_max`；
- 这种不一致是否持续至少 `failure_trigger_duration_s`；
- 原始条件是否连续满足 `failure_trigger_confirm_frames` 帧。

满足以上条件后只产生一次最终暂停事件。以下情况不会积累触发证据：

- 尚处于 `failure_trigger_start_after_s` 指定的预抓取保护时间；
- 玩具不存在或视觉来源不可靠；
- EE pose 缺失或超过 `failure_trigger_pose_max_age_s`；
- 时间窗口尚未完成预热；
- 机器人运动与玩具运动没有形成不一致。

当前实现中的 Z 速度与目标框面积变化会写入日志，便于后续分析抓取和抬升阶段，但 Stage-1 的 EE pose 硬触发仍以 XY 速度为主。

### 3.4 隔离 YOLO 环境

如果当前 LingBot/仿真环境不能直接安装 Ultralytics，可配置 `failure_trigger_yolo_python`。主评测进程会启动持久化的 `yolo/yolo_world_worker.py` 子进程，通过逐帧请求调用独立环境中的 YOLO-World，避免 Torch 与 Ultralytics 依赖污染策略环境。

## 4. 两种在线评测模式

### 4.1 Stage-1：YOLO + robot state

入口：

```bash
python kuavo_deploy/src/scripts/script_auto_test_yolo_robot_state.py \
  --task auto_test_yolo_robot_state \
  --config configs/deploy/deploy_lingbot_yolo_trigger_sim_task1.yaml
```

该模式在普通 checkpoint 仿真推理循环之后同步执行触发检测。检测到事件后发布暂停、结束当前 rollout，并保存：

```text
<eval-output>/yolo_robot_state_<timestamp>/
├── rollout_<episode>_yolo_robot_state_trigger.csv
├── rollout_<episode>_<camera-key>.mp4
├── yolo_robot_state_episode_summary.csv
└── evaluation_yolo_robot_state.log
```

触发 CSV 包含视觉来源、信号来源、EE 数据年龄、玩具位移、目标框面积变化、EE 总速度/XY 速度/Z 速度、候选持续时间、触发原因和最终暂停标志。

### 4.2 Stage-2：YOLO trigger + VLM recovery

入口：

```bash
python kuavo_deploy/src/scripts/script_auto_test_yolo_vlm_recovery.py \
  --task auto_test_yolo_vlm_recovery \
  --config configs/deploy/deploy_lingbot_yolo_trigger_sim_task1.yaml \
  --stage2-python <qwen-python> \
  --stage2-model-path <qwen-model-path>
```

Stage-1 确认后，系统暂停机器人，将本次 head-camera 历史写成视频片段，并调用已有的 VLM verifier。Stage-2 返回失败类型和确定性的恢复动作：

- `RESUME`：视为误报，解除暂停，清空监控器的时序历史并继续当前 rollout；
- `RESET_AND_RETRY`：重置仿真，在允许的重试次数内重新执行该 episode；
- 其他停止动作：结束评测。

额外输出包括：

```text
<eval-output>/yolo_vlm_recovery_<timestamp>/
├── episode_<episode>_attempt_<attempt>_yolo_trigger.csv
├── episode_<episode>_attempt_<attempt>_<camera-key>.mp4
├── stage2_clips/*.mp4
├── stage2_failure_events.jsonl
├── yolo_vlm_recovery_episode_summary.csv
└── evaluation_yolo_vlm_recovery.log
```

## 5. 配置说明

通用默认值位于 `configs/deploy/total/deploy_total.yaml`，任务示例位于 `configs/deploy/deploy_lingbot_yolo_trigger_sim_task1.yaml`。

| 配置项 | 默认值 | 含义 |
| --- | ---: | --- |
| `failure_trigger_enabled` | `false` | 是否启用在线触发器 |
| `failure_trigger_ee_pose_topic_left/right` | 空 | 左右 EE `PoseStamped` topic |
| `failure_trigger_pause_topic` | `/kuavo/pause_state` | 暂停发布 topic |
| `failure_trigger_yolo_model` | `yolov8s-worldv2.pt` | YOLO-World 权重 |
| `failure_trigger_yolo_device` | `0` | 推理设备 |
| `failure_trigger_yolo_python` | 空 | 独立 YOLO Python；空值表示进程内推理 |
| `failure_trigger_yolo_conf` | `0.03` | 检测置信度阈值 |
| `failure_trigger_yolo_iou` | `0.5` | 检测 NMS IoU 阈值 |
| `failure_trigger_motion_window_s` | `0.5` | 计算运动差异的时间窗口 |
| `failure_trigger_ee_motion_min` | `0.02` | EE XY 速度或关节代理运动下限 |
| `failure_trigger_ee_vertical_motion_min` | `0.02` | 记录 Z 方向运动状态的阈值 |
| `failure_trigger_toy_motion_max` | `0.012` | 认定玩具静止的归一化图像位移上限 |
| `failure_trigger_duration_s` | `1.0` | 信号不一致的最短持续时间 |
| `failure_trigger_confirm_frames` | `2` | 最终暂停前的连续确认帧数 |
| `failure_trigger_start_after_s` | `0.0` | episode 开始后的保护时长 |
| `failure_trigger_pose_max_age_s` | `0.10` | EE pose 允许的最大数据年龄 |
| `failure_trigger_allow_joint_state_fallback` | `false` | 是否允许以关节状态替代 EE pose |

使用示例配置前应重点检查：

1. `failure_trigger_yolo_python` 是否存在且能导入 `ultralytics`；
2. GPU 编号和模型路径是否正确；
3. 仿真端是否发布左右 EE pose；若没有，必须显式开启 joint-state fallback；
4. `failure_trigger_start_after_s` 是否与任务首次抓取时刻匹配；
5. 暂停 topic 是否与仿真控制端一致。

## 6. 离线运动跟踪基准

目录 `yolo/motion_tracking_benchmark/` 提供统一的离线比较入口：

```bash
python yolo/motion_tracking_benchmark/run_benchmark.py \
  --backend yolo_csrt \
  --prompt toy \
  --output-dir outputs/motion_tracking/a_yolo_csrt
```

支持的组合为：

| Backend | 方法 |
| --- | --- |
| `yolo_csrt` | YOLO-World 初始化 + CSRT 跟踪 |
| `yolo_sam` | YOLO-World 初始化 + SAM 2.1 传播 |
| `grounding_dino_sam` | Grounding DINO 初始化 + SAM 2.1 传播 |
| `sam3` | SAM 3 文本提示视频分割/跟踪 |

所有 backend 统一输出 `tracks.csv`、`annotated.mp4` 和 `summary.json`。该基准只比较视觉后端，不读取机器人状态，也不会发布真实暂停消息。

完整参数和依赖见 `yolo/motion_tracking_benchmark/README.md`；遮挡与几何过滤规则见 `yolo/motion_tracking_benchmark/OCCLUSION_AND_GEOMETRY.md`。

## 7. 初始化器/跟踪器两阶段消融

本分支将端到端跟踪误差拆为两个层次：

```text
视频 ──> initializer ──> left/right 初始框
                         │
                         └──> tracker ──> 完整轨迹
```

- 初始化器比较：YOLO-World、Grounding DINO、SAM 3 text；
- 跟踪器比较：CSRT、SAM 2.1、SAM 3；
- 初始化指标：recall@IoU、左右侧 recall、中心误差、双目标 recall；
- 跟踪指标：recall@IoU、IoU、中心误差、双目标 recall。

为了公平比较，所有跟踪器必须使用同一份 seed manifest，不能为每个 tracker 选择不同的初始化结果。典型流程为：

```bash
# 1. 从 GT 生成公共种子
python yolo/motion_tracking_benchmark/two_stage_ablation/make_seed_manifest.py \
  --gt gt_boxes.csv \
  --output outputs/two_stage/seeds.csv

# 2. 使用公共种子运行 tracker
python yolo/motion_tracking_benchmark/two_stage_ablation/run_seeded_tracker.py \
  --backend csrt \
  --seeds outputs/two_stage/seeds.csv \
  --video-dir /path/to/videos \
  --output outputs/two_stage/csrt_predictions.csv

# 3. 统一评测
python yolo/motion_tracking_benchmark/two_stage_ablation/evaluate_two_stage.py \
  tracker \
  --gt gt_boxes.csv \
  --pred outputs/two_stage/csrt_predictions.csv \
  --name csrt \
  --output outputs/two_stage/csrt_tracker.json
```

完整初始化器命令、CSV schema 和 SAM 参数见 `yolo/motion_tracking_benchmark/two_stage_ablation/README.md`。

## 8. 关键文件索引

| 路径 | 作用 |
| --- | --- |
| `yolo/online_robot_state_trigger.py` | 在线视觉/机器人状态交叉触发器 |
| `yolo/yolo_world_worker.py` | 独立 Python 环境中的持久化 YOLO worker |
| `kuavo_deploy/src/eval/sim_auto_test_yolo_robot_state.py` | Stage-1 仿真闭环评测 |
| `kuavo_deploy/src/eval/sim_auto_test_yolo_vlm_recovery.py` | Stage-1 + VLM Stage-2 恢复评测 |
| `kuavo_deploy/src/scripts/script_auto_test_yolo_robot_state.py` | Stage-1 CLI 包装 |
| `kuavo_deploy/src/scripts/script_auto_test_yolo_vlm_recovery.py` | Stage-2 CLI 包装 |
| `configs/deploy/deploy_lingbot_yolo_trigger_sim_task1.yaml` | 当前任务的示例配置 |
| `yolo/motion_tracking_benchmark/run_benchmark.py` | 四类视觉 backend 的离线统一入口 |
| `yolo/motion_tracking_benchmark/two_stage_ablation/` | 初始化器与 tracker 的两阶段消融工具 |
| `yolo/eval_progress_trigger*.py` | 历史纯视觉触发实验 |
| `yolo/eval_failure_trigger_from_tracks.py` | 基于已有轨迹的离线触发评估 |
| `yolo/toy_annotations.json` | rollout 级失败标注 |
| `report/08_yolo_fast_trigger.md` | 快速触发实验记录 |
| `report/09_yolo_robot_state.md` | robot-state trigger 的设计背景与实验记录 |
| `report/10_MOTION_TRACKING` | 多视角、分阶段运动跟踪方案 |

## 9. 当前边界与注意事项

- 在线触发器当前只使用 head camera，`report/10_MOTION_TRACKING` 中规划的 wrist-camera 联合判断尚未完整落地。
- 左右玩具身份主要来源于初始图像侧别，不代表严格的机器人本体左右语义；目标交叉后应依赖持续跟踪保持身份。
- 关节状态回退仅能说明机器人发生了关节/夹爪运动，准确实验应优先接入同步的笛卡尔 EE pose。
- “机器人运动、玩具静止”主要覆盖抓取失败或传输不一致，不能独立覆盖所有放置错误、掉落和篮筐外释放情况。
- 当前阈值是任务相关参数，应在成功/失败 rollout 上同时评估；只提高 failure recall 可能显著增加成功任务的误暂停。
- 离线 motion-tracking benchmark 与在线 trigger 的目的不同：前者评估视觉后端，后者评估闭环失败触发，不应混用其指标。
- `gt_boxes.csv` 和示例路径中包含实验数据约定；在新数据集上运行前应检查 rollout 名称、视频后缀、帧号与左右目标身份是否一致。

## 10. 建议验证顺序

1. 使用 `--dry-run` 检查配置解析、模型路径和 CLI 参数；
2. 在单个短 episode 上运行 Stage-1，确认 YOLO worker、ROS topic、时间戳和 CSV 正常；
3. 检查成功 rollout 是否在预抓取/调整阶段误触发；
4. 使用人工标注的失败 rollout 评估触发时刻与失败时刻之间的延迟；
5. 固定 Stage-1 参数后再启用 Stage-2，分别验证 `RESUME` 和 `RESET_AND_RETRY`；
6. 最后扩大到完整 `eval_episodes`，同时报告成功率、失败检出率、成功任务误暂停率和平均触发延迟。

