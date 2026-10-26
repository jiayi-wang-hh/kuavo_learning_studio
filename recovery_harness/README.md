# Kuavo VLM Skill Routing / Recovery Harness

这个目录实现的是 Kuavo 包裹分拣的 VLM 技能路由与恢复控制框架。它把相机图像、可用的 CV 证据、触觉接触、机器人状态和 Skill Registry 一起交给 VLM；VLM 只提出技能建议，控制器再检查新鲜度、机器人状态、停止确认、目标定位、碰撞检查和 checkpoint 状态。

**当前支持离线 VLM 路由、ROS 只读 live dry-run、异步 VLM 调度和离线 mock 执行。没有任何模式会隐式发送机器人动作。** mock policy/action backend 仅用于软件闭环，不加载 VLA、不连接 ROS motion interface。真机动作仍被禁止：真实 Kuavo executor、安全状态/停止确认和 Recovery HF export 尚未接通。

## 整体工作流程

```mermaid
flowchart TD
    C[Head RGB] --> S[时间戳缓冲 / 新鲜度检查]
    LW[Left Wrist RGB 可选] --> S
    RW[Right Wrist RGB 可选] --> S
    T[DexHand tactile 可选] --> S
    J[Kuavo joint state] --> S
    S --> P[CV 证据与 head ROI 几何]
    S --> O[Multi-view Observation + 历史]
    P --> O
    P --> E[CV event candidates]
    T --> E
    E --> SCH[VLM event / periodic scheduler]
    R[Skill Registry] --> V[VLM 动态注入 Skills 并输出 JSON 建议]
    O --> SCH --> V
    V --> D{Controller 校验建议/安全状态}
    D -->|普通场景| N[NORMAL / normal_sort 建议]
    D -->|异常候选| F[FAILURE_CANDIDATE]
    F --> X[停止 normal policy / 取消未执行 chunk]
    X --> Q{停止确认与 Recovery 前置检查}
    Q -->|失败/未知| Z[STOPPED / stop_and_verify]
    Q -->|通过且有动作授权| A[RECOVERING / recover_package]
    A --> W[VERIFY_RECOVERY]
    W -->|视觉与机器人状态通过| U[RESUME_PRECHECK]
    U -->|Normal 起始状态已确认| N
    U -->|否则| Z
    M[当前配置: ActionExecutor disabled, 无真机后端] -. 阻断真实动作 .-> A
```

### 1. 读取配置和技能注册表

入口 [run.py](run.py) 从 [package_sorting.yaml](config/package_sorting.yaml) 读取运行配置，再加载 [skill_registry.yaml](config/skill_registry.yaml)。Registry 是 VLM 可推荐 Learned VLA Skills 的唯一来源，初始注册：

- `normal_sort`：一个完整的双臂 Pick–Flip–Place 任务。左臂从传送带取包裹并放到黄色平台；右臂按需翻转，再放入篮子。
- `recover_package`：左臂从黄色平台身体侧的异常位置重新抓取包裹，并放回平台。
- `stop_and_verify`：控制器动作，不是 VLA Skill。

每个 Skill 配置包含任务描述、task instruction、checkpoint 与 selector、base/source model 路径、normalization、pre/postconditions、支持场景、timeout 和 execution status。每次推理时 `SkillRegistry.prompt_descriptions()` 从当前 Registry 生成完整的 Available Skills 文本；Prompt 本身不维护第二份硬编码 Skill 清单。

启动前可运行不加载模型的路径检查：

```bash
PYTHONPATH="$PWD:$PWD/src/lingbotvla" python3 -m recovery_harness.run preflight
```

### 2. 收集和整理观测

`KuavoSensorBridge` 是只读 ROS bridge，不创建动作 publisher 或运动 service client。话题由配置决定，默认包括：

| 数据 | 默认 ROS topic | 用途 |
|---|---|---|
| Head RGB | `/cam_h/color/image_raw/compressed` | 全局任务场景、平台和篮子 |
| Left Wrist RGB | `/cam_l/color/image_raw/compressed` | 可选的身体侧目标和抓取/释放辅助视角 |
| Right Wrist RGB | `/cam_r/color/image_raw/compressed` | 可选；默认关闭 |
| DexHand tactile | `/dexhand/touch_state` | 左右手五指的多指接触摘要 |
| Kuavo joints | `/sensors_data_raw` | 当前 joint state；不等同于安全或停止状态 |
| Operator pause | `/kuavo/pause_state` | `true` 暂停 dry-run 的 VLM 调度；`false` 恢复 |
| Operator stop | `/kuavo/stop_state` | `true` 锁存停止请求、取消策略并令 FSM 进入 `STOPPED` |

图像放入各自的 `TimestampedBuffer`。ROS `header.stamp` 为 0 或无效时会标记为 receipt-time 来源。推理前会过滤过期图像/触觉；VLM 推理完成后会重新检查**实际用于该次推理的图像**以及最新机器人状态，防止耗时推理让旧建议直接生效。各相机分别取最近的新鲜帧，当前实现不是严格的跨相机硬件时间同步器。

近期历史只记录之前的 VLM 建议与感知结果，不把旧目标位置当作当前确认位置。Left/Right Wrist 图像作为独立视角输入；没有标定时不会把 wrist 像素转换成 head 坐标。

### 3. CV 与几何证据

`PerceptionAdapter` 负责合并可选 detector/tracker 输出和 head ROI 几何证据：

1. 可以通过 `perception.detector_plugin: "module:function"` 接入现有检测或分割代码；Detector 结果不是 VLM 推理的前置条件。
2. `CandidateTracker` 可维护短寿命 CSRT 轨迹，并记录来源、年龄、有效性和是否为部分可见对象。轨迹丢失或过期后不会无限沿用旧位置。
3. Head ROI JSON 中的 `platform_polygon`、`body_side_edge` 和像素坐标会作为 VLM 上下文提供。当前配置指向 `/media/data/jiayi/dataset/roi_config.json`。
4. 如果 detector 给出完整 package box，可计算 box 与 platform polygon 的近似重叠比例，以及 box 中心是否越过身体侧边缘。部分可见 box、CSRT-only box 或无效几何的数值为 `null`/`unknown`，不会冒充完整包裹 footprint。
5. Wrist 图像不套用 head ROI；只让 VLM 独立理解 wrist 画面。

仓库里曾有 YOLO-World/CSRT/SAM 实验的 `.pyc` 缓存，但对应源码当前不在 checkout 中，因此本 harness 没有声称复用了那些实现。未配置 detector 时，VLM 仍直接接收原始 RGB 图片和 ROI 坐标；检测不到候选框不会自动否决视觉推理。

感知上下文包括 `package_visible`、`package_partially_visible`、`package_occluded`、`package_location`、`body_side_violation`、`overlap_ratio`、`geometry_validity`、track age/source 和 ROI 像素坐标。证据不足时显式使用 `unknown`。

### 4. 触觉接触状态

`BinaryContactProcessor` 对每根手指把三个 normal-force channel 相加，按每指阈值转成二值接触；至少两根手指接触并持续 100 ms 才输出该手的 `multifinger_contact=1`。触觉仅表示接触，不能单独证明抓住了目标。

当前 `finger_force_thresholds` 与 `valid_status_values` 默认是 `null`。驱动源码只声明 `status` 是整数，没有给出其状态含义；真实传感器的无载基线也未标定。因此默认触觉状态是 `UNKNOWN`，不会猜阈值或 status 语义。离线 JSONL manifest 可以显式提供 `left_multifinger_contact` / `right_multifinger_contact`。

### 5. VLM 场景理解与技能推荐

`SceneReasoner` 每次根据当前 Registry 动态构造 Prompt，并输入：

- 当前可用的 Head RGB 和启用的 Wrist RGB；
- 可选 CV/tracking/ROI 几何证据；
- 最新触觉接触摘要；
- 简化后的 robot state、观测 freshness 和近期历史；
- 从 Registry 即时生成的 Available Skills 描述及合法输出集合。

`QwenVLMAdapter` 根据配置只加载一个模型：`qwen2_5_vl` 使用 Qwen2.5-VL model class；`qwen3_vl` 使用 Qwen3-VL class。Processor 和 chat template 从所选本地模型目录读取。VLM 输出 JSON 字段为 `scene_state`、`task_progress`、`recommended_skill`、`target_object`、`reason` 和 `requires_verification`。只接受 Registry 中的 Skill ID 或 `stop_and_verify`；JSON 错误、非法 Skill 或推理异常都会 fail-closed 到停止核验。

VLM 只是建议者。它不拥有动作执行权限，也不生成关节命令、IK 或碰撞检查结果。

Live dry-run 使用 `EventTrigger` 和单 worker `VLMInvocationScheduler`：稳定 CV 异常需满足配置的连续帧数；已校准的 `BinaryContactProcessor` 输出接触状态转变时可立即触发；独立周期推理作为低频兜底。事件只决定何时推理，不决定技能。相同事件在 cooldown 内去重；推理不会并行，且完成后按本次输入图像年龄拒绝过期结果。ROS 主循环每 50 ms 调度，不等待 VLM 返回。触发、推理耗时和结果有效性会写入日志/输出。

### 6. Controller FSM 与安全门

`RecoveryController` 区分 VLM 推荐、Controller 选择和实际 VLA 执行。FSM 状态为：

`NORMAL → FAILURE_CANDIDATE → STOPPING → RECOVERY_PRECHECK → RECOVERING → VERIFY_RECOVERY → RESUME_PRECHECK → NORMAL`

任何前置条件失败或状态未知都进入 `STOPPED`；全任务完成需另行确认全部六个包裹在篮子中及机器人安全。核心检查包括：

- 用于建议的图像和当前 robot state 是否新鲜、有效，`safety_ok` 是否明确为 true；
- Normal policy 是否停止、待执行 action chunk 是否取消、机器人是否有停止确认；
- Recovery 目标是否可靠定位、恢复 checkpoint 是否可加载、碰撞检查是否通过；
- Recovery 后目标是否仍可见并已回到平台、机器人是否安全。当前 `verify_recovery` 只在 tactile 明确为 `1` 时拒绝；tactile 为 `UNKNOWN` 不会证明夹爪已释放，所以真机接入前还需加上可信 gripper/release 确认；
- Normal full-task policy 是否支持 Recovery 后的初始状态。不能默认 Normal policy 能从任意状态恢复。

`ActionExecutor` 默认 `enabled=False` 且没有 backend。即使 VLM 推荐了 `recover_package`，没有停止确认、碰撞/定位检查和显式动作授权也不能开始恢复。Policy adapter 按需加载当前 Skill，切换时卸载旧模型并清理 pending chunk，避免两个 VLA 同时常驻。

### 7. 策略执行、恢复验证和继续分拣

代码中的 `LingBotPolicyAdapter` 复用仓库现有 `kuavo_server.adapters.lingbot_vla.LingBotVlaAdapter`，包括它的 LingBot inference、task prompt、norm stats 和 action chunk 入口。控制器提供 `start_normal`、`begin_recovery`、`step_active_policy`、`finish_recovery_execution`、`verify_recovery` 和 `approve_resume` 等门控接口。

完整真机序列预期为：停止 Normal 并确认机器人停稳 → 检查 Recovery 前置条件 → 加载 `recover_package` → 在授权的 executor 下逐步执行 action chunk → 等待独立的视觉/状态验证 → 仅在 Normal 起始状态支持性得到确认后重载并继续 `normal_sort`。如果任何一步不确定，停止并等待人工处理；不执行自动 Reset 或探索动作。

离线 `--mock-execution` 可将经 VLM 路由后的决定交给 FSM，再调用纯 Python mock policy/action backend。Recovery 流程需要 manifest 明确提供 `mock_prechecks`、`mock_verification` 和 `mock_resume`；这些字段是合成测试输入，不可表示真实安全信号。该模式不加载 LingBot VLA，不创建 ROS publisher/service，也不会下发动作。验证成功还必须提供明确 `release_confirmed=true`；未知 tactile 或 action chunk 完成不构成释放证明。

## Checkpoint 和资源状态

- Normal checkpoint：用户提供的 `task3_package_lora` 路径在当前服务器不存在；实际找到的是 `/media/data/jiayi/outputs/kuavo_lingbotvla/task3_pacakge_lora/global_step_66480/hf_ckpt`（目录确实拼作 `pacakge`）。这是含 `config.json` 与 safetensors 的 HF checkpoint，可由现有 adapter 解析。
- Recovery checkpoint：配置固定选择 `global_step_28300`。该目录为多 rank DCP (`model/*.distcp`、`.metadata`)，不是现有 HF-only inference adapter 可直接加载的导出格式。需先转换并验证 HF export，再在 `skill_registry.yaml` 设置 `exported_hf_checkpoint`；本 harness 不转换或修改模型权重。
- VLM 默认配置为本地 Qwen2.5-VL-7B；改为 `backend: qwen3_vl` 和 Qwen3 本地目录即可选择 Qwen3-VL-4B。运行时只加载所选的一个 VLM。

## 可运行模式

所有命令从仓库根目录执行，并在相应环境中安装 ROS、LingBot 和 Transformers 运行依赖。

### Preflight（不加载模型）

```bash
PYTHONPATH="$PWD:$PWD/src/lingbotvla" python3 -m recovery_harness.run preflight
```

此命令会核对 VLM、VLA base/source、norm stats、training config、robot config 和 checkpoint。当前预期是 Normal ready，Recovery 因 DCP 格式不能被 HF inference adapter 加载而报告失败。

### Offline 图片 replay（只运行 VLM，不执行 VLA）

JSONL 每行代表一个观测样本，图片路径相对当前工作目录。可选添加 wrist 图、timestamp、`robot_state`、`tactile` 和 label。结果默认打印；`--output` 可附加写入结果 JSONL。每次最多 10 帧：

```json
{"label":"recovery_sample","head_image":"vlm_test/recovery_rgb_tactile/images/frame_00004.jpg","left_wrist_image":null,"robot_state":{"valid":true,"safety_ok":true,"motion_state":"stopped"},"tactile":{"left_multifinger_contact":null,"right_multifinger_contact":null}}
```

```bash
PYTHONPATH="$PWD:$PWD/src/lingbotvla" python3 -m recovery_harness.run offline \
  --manifest recovery_harness/examples/replay.example.jsonl \
  --max-frames 10 \
  --output outputs/vlm_test/replay_results.jsonl
```

前一轮 Qwen2.5-VL-7B routing probe 使用 2 帧 Normal 和 3 帧 Recovery。模型没有区分 Recovery 初始异常位置，详情见 [测试结果 JSONL](../outputs/vlm_test/skill_routing_probe_qwen25_7b.jsonl) 和 [测试摘要](../outputs/vlm_test/skill_routing_probe_qwen25_7b_summary.md)。之后增加的 ROI 坐标上下文尚未重新做 VLM 推理。

### Live sensor-only dry-run（ROS 只读，VLM 异步调度，不运行 VLA/动作）

```bash
PYTHONPATH="$PWD:$PWD/src/lingbotvla" python3 -m recovery_harness.run live-dry-run \
  --max-observations 10
```

`--max-observations 0` 表示持续运行到 Ctrl-C。该模式订阅配置中的 ROS 图像、触觉和关节状态，加载一个所选 VLM，周期/事件触发推理并输出触发证据、推理 freshness、推荐与 controller decision。它不创建动作 publisher/service，也不加载 VLA。

它也订阅 `real_single_test.py` / `sim_auto_test.py` 使用的 pause/stop Bool topics：pause 为真时不调度新推理，暂停期间完成的推理结果会丢弃；stop 为真时锁存请求，调用 policy/executor 的 stop 请求并将 controller 置为 `STOPPED`。这些信号只是操作员请求。尤其 `/kuavo/stop_state` 不是机器人已物理停稳的确认，不能替代 recovery precheck 所需的 stopped acknowledgement。`sim_auto_test.py` 的 `/simulator/init`、`/simulator/success` 与 `/simulator/reset` 仅用于仿真 rollout，不适用于本 harness，也不会在这里调用。

### Offline VLM + mock policy/action orchestration

```bash
PYTHONPATH="$PWD:$PWD/src/lingbotvla" python3 -m recovery_harness.run offline \
  --manifest recovery_harness/examples/replay.example.jsonl --max-frames 10 --mock-execution
```

该命令调用配置的 VLM；VLA policy 和 Kuavo backend 均替换成显式 mock。要覆盖模拟 Recovery/Resume，manifest 中必须提供明确标为 mock 的前置条件、视觉/释放验证和 Normal 初始状态支持字段。输出 `robot_motion:false` 表示纯软件模拟。

### Real-robot preflight (currently refused)

```bash
PYTHONPATH="$PWD:$PWD/src/lingbotvla" python3 -m recovery_harness.run real-robot
```

当前命令 fail-closed 并列出缺失接口；不会尝试加载 VLA 或创建动作 client。

## 当前限制与真机前必须完成的事项

1. 选择的 Recovery DCP checkpoint 需要导出为现有 LingBot 推理入口支持的 HF safetensors，并验证权重/配置/base model 对应关系。
2. DexHand 每指阈值需用真实无载基线校准，`status` 合法值需从驱动定义确认；否则 tactile 保持 `UNKNOWN`。
3. `/sensors_data_raw` 在当前 bridge 中只证明有 joint state，不提供 `safety_ok`、机器人是否停稳、gripper 是否释放或 collision-free 状态。需接入可信状态提供器。
4. 需实现和验证 site-specific Kuavo `ActionExecutor`，覆盖 action 单位、限幅、控制模式、chunk 取消、停止确认和故障停机。当前代码不实例化会进入控制设置的 `KuavoRealEnv`。
5. 需在实机环境验证完整 Normal → Stop → Recovery → Verify → Resume 流程，并验证 Normal full-task policy 对 Recovery 后状态的支持性。

## 验证状态与剩余集成阻塞

- **实现并由单元测试覆盖**：Registry 描述/推荐合法性、CV/触觉事件候选、单 worker 事件/周期调度、重复事件合并、过期结果拒绝、FSM 路由与 fail-closed、Recovery 释放证据门、mock backend 无 ROS/motor 连接、真实 backend 缺失时报错。
- **本次实际验证**：只运行轻量 Python unittest 与语法编译；未加载 Qwen/VLA，没有 ROS 运行环境联调，也没有真机测试。
- **真实 ROS 观察**：只读订阅代码已存在，但本次没有启动 ROS，不能声称本次验证了 live observations。
- **真机未验证且 real-robot CLI 当前拒绝执行**：需要 Kuavo action backend（动作单位/限幅/模式）、活动 policy 停止确认、pending chunk drain/cancel 确认、可信 stopped acknowledgement、joint/end-effector validity、collision/safety checker、gripper release evidence、Recovery HF checkpoint export 和 deployment authorization。
- `/sensors_data_raw` 当前只提供 joint 数据；没有证据说明它包含安全、停稳、碰撞或夹爪释放状态。`/kuavo/stop_state` 也未验证为本 harness 停止确认，因此不会被当作安全信号。
- Pause/stop topic 的消息类型和语义按 eval 脚本使用的 `std_msgs/Bool` 接入；它们可停止 harness 的推理/控制流程，但不会单独发布或保证硬件急停。
- Detector plugin 默认仍为 `null`。若需 CV 触发，请配置已有且已验证的 detector 插件；无检测器时依靠周期 VLM 和触觉事件触发。CV geometry 是 image-space 候选证据，不是物理支撑判定。

本次没有执行模型训练、checkpoint 转换、完整数据集 replay 或真机动作。机器人运动保持禁用。
