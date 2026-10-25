# Codex Task: Implement Kuavo VLM-Guided VLA Skill Routing and Failure Recovery

## 0. 任务目标

请在现有仓库中实现完整的 VLM-Guided VLA Skill Routing + Failure Recovery Pipeline。

仓库：

`~/jiayi/kuavo_learning_studio`

目标是让 Kuavo 双臂机器人根据实时场景信息，通过 VLM 在两个已经训练好的 LingBot-VLA Policy 之间进行选择与安全切换：

- `normal_sort`
- `recover_package`

实现完整闭环：

Normal VLA → Failure Observation → VLM Reasoning → Skill Selection → Safe Policy Switch → Recovery VLA → Verify → Resume Normal VLA

**本任务必须实际修改仓库代码，不能只输出架构方案。**

注意：本地电脑不具备运行高负载模型或复杂测试的能力。因此禁止在本地运行 VLM/VLA 推理、模型训练、GPU 测试或完整 Offline Replay。

实现代码和配置，但将需要 GPU、ROS、真实模型或硬件资源的运行步骤留给用户在指定计算设备上执行。

不要自行假设本地具备足够计算资源。

---

## 1. Git 分支要求（必须首先执行）

所有代码修改必须在新分支：

`lingbot-vlm-vlm-skills`

中完成。

首先执行：

```bash
git status
git branch --show-current
git branch --list lingbot-vlm-vlm-skills
```

检查现有分支和未提交修改。

如果目标分支不存在，基于当前合适的开发基线创建：

```bash
git switch -c lingbot-vlm-vlm-skills
```

如果目标分支已经存在，检查它的内容后安全切换。

不得覆盖、删除或丢弃用户未提交的修改。

不要自动执行 `git reset --hard`、`git clean -fd`、强制推送或修改其他分支的历史。

必须确认当前分支为 `lingbot-vlm-vlm-skills` 后再修改代码。

禁止直接在 `master`、`main` 或现有其他开发分支上实现本任务。

不要自动 Merge 或 Push。

---

## 2. 现有任务

### 2.1 Normal Task

工作场景：

- 左侧：传送带，有六个 Package。
- 中间：黄色 Sorting Platform。
- 右侧：Basket。

Normal LingBot-VLA 执行完整的 Package Pick–Flip–Place：

1. 左手从传送带取包裹。
2. 左手将包裹放到黄色平台。
3. 右手根据包裹朝向进行 Flip。
4. 右手将包裹放入 Basket。
5. 重复直到分拣完成。

Normal 是一个完整任务 Policy，而不是多个独立的 Pick、Flip、Place Skills。

### 2.2 Failure Recovery

Failure 场景：

Package 掉到黄色操作台靠近机器人身体的一侧，或者部分悬出 Platform。

Recovery Policy：

左手重新抓取该 Package，将其放回黄色 Platform。

Recovery 训练数据以人工布置的异常包裹位置为初始状态，没有记录 Package 实际掉落过程。

因此，Failure Detection 的主要任务是识别异常位置，而不是必须观察到掉落运动。

---

## 3. 实际模型 Checkpoint

### Normal LingBot-VLA

```text
/media/data/jiayi/outputs/kuavo_lingbotvla/task3_package_lora
```

### Recovery LingBot-VLA

```text
/media/data/jiayi/outputs/kuavo_lingbotvla/task3_package_lora_failure/package_below_target_zone/vlm_action_bs16_ga1_vs8/checkpoints
```

检查以上路径的实际目录结构。

确认：

- LoRA Adapter 或完整 Model。
- 需要的 Base Model。
- Model Configuration。
- Normalization Statistics。
- Checkpoint Loading Procedure。
- Inference Entry Point。
- Prompt / Task Instruction。

Recovery 路径可能是包含多个训练 Checkpoint 的目录。需要检查并提供可配置的具体 Checkpoint 选择，不允许随意假设目录本身可直接加载。

不要训练新的 LingBot-VLA Model。

不要修改训练好的模型权重。

---

## 4. VLM 模型

支持两个可配置 VLM Backend：

### Backend A

`Qwen2.5-VL-7B-Instruct`

检查现有模型目录，找到本地可用的 7B Checkpoint。

不要自动下载大型模型。

### Backend B

`Qwen3-VL-4B-Instruct`

路径：

```text
/media/data/jiayi/hf_model/Qwen3-VL-4B-Instruct
```

实现统一的 VLM Adapter，允许用户通过配置切换模型。

注意：两个模型可能使用不同的 Processor、Chat Template 和 Model Class。

不能直接假设它们具有完全相同的加载代码。

优先复用已有 VLM 推理实现。

禁止默认同时加载两个 VLM。只加载用户配置选择的模型。

---

## 5. 核心要求：VLM 必须知道可调用的 VLA Skills

这是本项目最重要的功能之一。

不能让 VLM 仅仅输出一个预先写死的 Failure Boolean，再让 Python 根据固定规则选择 Recovery。

必须真正实现 **VLM-Guided Skill Routing**。

### 5.1 Skill Registry

建立一个可扩展的 Skill Registry。

初始包含两个 VLA Policy：

#### Skill 1 — normal_sort

```yaml
skill_id: normal_sort
skill_type: vla_policy
description: >
  Complete bimanual package sorting. The left arm picks packages
  from the conveyor and places them on the yellow sorting platform.
  The right arm flips packages if needed and transfers them
  into the basket.
checkpoint: /media/data/jiayi/outputs/kuavo_lingbotvla/task3_package_lora
```

#### Skill 2 — recover_package

```yaml
skill_id: recover_package
skill_type: vla_policy
description: >
  Recover a misplaced package from the robot-body side
  of the yellow sorting platform. The left arm grasps
  the misplaced package and returns it to the platform.
checkpoint_root: /media/data/jiayi/outputs/kuavo_lingbotvla/task3_package_lora_failure/package_below_target_zone/vlm_action_bs16_ga1_vs8/checkpoints
```

每个 Skill 必须记录：

- `skill_id`
- `description`
- `checkpoint`
- `task_instruction`
- `expected_preconditions`
- `expected_postconditions`
- `supported_scenarios`
- `execution_timeout`
- `execution_status`

Skill Registry 是唯一合法 Skill 来源。

### 5.2 Dynamic Skill Description Injection

每次 VLM Reasoning 时，必须从 Skill Registry 动态生成 Available Skills 信息。

例如：

```text
Available robot skills:

1. normal_sort
   Function: Perform complete package sorting.
   Appropriate when: Packages are in normal sorting positions,
   and normal task execution can continue.

2. recover_package
   Function: Pick up a misplaced package from the body-side
   platform region and return it to the yellow platform.
   Appropriate when: A misplaced package is observed,
   recovery geometry is supported, and the robot
   satisfies recovery execution preconditions.

You may recommend only one registered skill.
If the scene is ambiguous, return stop_and_verify.
```

不允许在 VLM Prompt 中维护一份与 Registry 无关的硬编码 Skill 列表。

### 5.3 VLM 输出

使用结构化 JSON：

```json
{
  "scene_state": "package_misplaced",
  "task_progress": "sorting_interrupted",
  "recommended_skill": "recover_package",
  "target_object": "package_near_body_side",
  "reason": "The package is misplaced outside the valid sorting area.",
  "requires_verification": false
}
```

上述只是输出格式示例。

实际状态由 VLM 根据观察结果生成。

合法输出：

- `normal_sort`
- `recover_package`
- `stop_and_verify`

`stop_and_verify` 是控制器动作，不属于 Learned VLA Skill。

必须检查 VLM 输出是否合法。

VLM 负责推荐 Skill，但不是最终执行权限的拥有者。

最终执行必须经过 Controller 的前置条件及安全检查。

---

## 6. 多视角视觉感知：Head + Wrist Cameras

### 6.1 重要现实问题

当前 Failure 场景存在严重的视觉困难。

Package 掉到黄色操作台身体侧之后：

- 可能不能完整出现在 Head Camera 画面中。
- 可能只有部分 Package 可见。
- Package 可能被 Platform 遮挡。
- 由于透视视角，Package 不再呈现规则矩形。
- 单张 Head RGB 图像中的 Axis-aligned Bounding Box 不一定能准确描述 Package 的几何状态。

**因此，不能把 YOLO Bounding Box + 固定 Platform ROI 当作唯一 Failure Detection 方法。**

### 6.2 Head Camera

负责提供全局场景观察：

- Conveyor。
- Yellow Platform。
- Package。
- 左右臂。
- Basket。
- 当前任务状态。

已有 Head Camera ROI：

```text
/media/data/jiayi/dataset/roi_config.json
```

可视化：

```text
/media/data/jiayi/dataset/roi_config_preview.jpg
```

使用 JSON 中的 `platform_polygon` 和 `body_side_edge`。

不要把 Preview JPG 当作当前实时观测。

### 6.3 Left Wrist Camera

将左腕相机作为可选但优先支持的第二个视角。

其作用：

- 辅助观察靠近机器人身体侧的 Package。
- 在 Head Camera 看不清 Package 时提供额外视觉证据。
- 辅助观察 Recovery Grasp 和 Release。

必须注意：左腕相机可能也无法看到目标。

不得假设它始终比 Head Camera 看得清楚。

由于腕部相机跟随机械臂运动，不能直接使用 Head Camera 的固定 ROI 坐标。

对于 Wrist Image，可以由 VLM 分别进行视觉理解；如果已有可靠的相机标定和坐标变换，也可以进行几何融合。

不允许编造 Wrist-to-Head 坐标转换。

### 6.4 Right Wrist Camera

保留接口。

可以用于 Normal Flip/Place 的辅助验证。

第一版不强制参与每次 VLM Reasoning。

所有相机由配置决定是否启用。

### 6.5 Multi-view VLM Input

设计统一的 Multi-view Observation：

```json
{
  "images": {
    "head": "head_rgb_frame",
    "left_wrist": "left_wrist_rgb_frame",
    "right_wrist": null
  },
  "available_views": [
    "head",
    "left_wrist"
  ]
}
```

上述为数据结构示意，不要求将图像编码成 JSON 字符串。

使用模型实际支持的多图片输入方式。

如果某个相机不可用，不能导致整个软件崩溃。

如果所有可用相机都不能明确观察目标，输出 `unknown` 或 `stop_and_verify`。

不要让机器人为了观察而自动执行未经验证的探索动作。

---

## 7. Package Perception：不要强制训练 YOLO

第一版禁止自动训练检测模型。

优先尝试仓库现有工具：

- YOLO。
- SAM / SAM2 / SAM3。
- Grounding-based Detection。
- Existing Object Tracking。
- Existing VLM-based Object Localization。

检查实际仓库环境与可用模型文件。

实现统一的 Perception Adapter。

### 7.1 几何感知

如果包裹检测可靠且可见，计算：

\[
R_{\mathrm{overlap}}
=
\frac{
\mathrm{Area}(B_{\mathrm{package}}\cap P_{\mathrm{platform}})
}{
\mathrm{Area}(B_{\mathrm{package}})
}
\]

并计算是否越过 Body-side Edge。

但是应当明确：

- Bounding Box 不是实际 Package Footprint。
- Package 透视变形可能导致计算误差。
- Package 被部分遮挡或出画时，2D Overlap 可能无效。
- 只有 Head Camera 的 ROI 能直接用于该视角的几何计算。

如果存在可靠的 Segmentation Mask，可以计算 Mask-based Overlap。

如果检测不到完整对象，允许使用可见区域、历史目标轨迹和多视角证据辅助判断，但不能把可见部分的 overlap 假装成完整物体的 overlap。

### 7.2 视觉状态

输出：

- `package_visible`
- `package_partially_visible`
- `package_occluded`
- `package_location`
- `body_side_violation`
- `overlap_ratio`
- `geometry_validity`
- `visual_evidence`

当几何计算无效时：

```json
{
  "package_visible": true,
  "package_partially_visible": true,
  "overlap_ratio": null,
  "geometry_validity": "unreliable",
  "body_side_violation": "unknown"
}
```

不要强行生成不可靠的几何值。

### 7.3 Temporal Reasoning

保留最近若干个时间点的 Observation 和 Perception Results。

如果 Package 曾经可见，随后被遮挡，可以将上一时刻目标信息作为历史证据。

但不能将历史位置当作当前已确认位置。

VLM 应能区分：

- Package 仍在正常搬运。
- Package 可能被放到异常位置。
- Package 当前不可见。
- Recovery 正在进行。
- Package 已经回到平台。

---

## 8. 触觉处理

ROS Topic：

`/dexhand/touch_state`

Message Type：

`kuavo_msgs/dexhandTouchState`

每只手五根手指。

每根手指包含三个 Normal Force Measurement。

采用简单 Binary Multi-finger Contact Rule：

\[
F_i=N_{i,1}+N_{i,2}+N_{i,3}
\]

\[
C_i=\mathbb{1}[F_i>\tau_i]
\]

\[
G=\mathbb{1}\left[\sum_i C_i\geq2\right]
\]

输出：

- `left_multifinger_contact`
- `right_multifinger_contact`

使用 100 ms 的初始时间持续条件。

阈值必须考虑真实传感器基线。

检查 ROS 驱动中 `status` 的含义。

缺失或无效数据输出 `UNKNOWN`。

不要根据 Aggregate Normal Force 直接判断 Grasp。

之前 Recovery 数据中左手法向力曾长时间维持 106，不应直接解释为有效抓取。

Contact 和 Grasp 是不同概念。

VLM 使用简化后的 Tactile State。

Robot Controller 可以使用额外的 Joint/Gripper State 判断执行条件。

---

## 9. Robot State 的职责

读取以下信息：

- 左右臂 Joint State。
- 左右 Gripper State。
- Active VLA Policy。
- Execution Status。
- Observation Freshness。
- Existing Safety Status。

Robot State 主要传给 Recovery Controller。

不需要把完整 14 维 Joint Values 直接传入 Qwen2.5-VL。

可以提供简化的 Symbolic State：

```json
{
  "active_skill": "normal_sort",
  "left_multifinger_contact": 0,
  "right_multifinger_contact": 0,
  "robot_motion_state": "running"
}
```

VLM 不能自行完成 IK 或碰撞检查。

必须复用真实的 Kuavo Robot Control Interfaces。

---

## 10. Observation Synchronization

整合：

- Head RGB。
- Optional Wrist RGB。
- Tactile。
- Robot State。

检查实际 ROS Timestamp。

已知 Head RGB 的 `header.stamp` 有时为 0。

Offline Replay 可以使用 ROS bag Record Timestamp 进行近似时间对齐。

Online 需要使用 Timestamped Buffers，并记录观测新鲜度。

VLM 推理可能需要数秒。

因此，VLM 给出的 Skill Recommendation 在执行前必须使用更新的 Observation 和 Robot State 再次确认。

尤其在 Normal VLA 正在运动时，不能直接使用过期的图像结果中断或切换机器人运动。

---

## 11. VLM-Guided Skill Router

这是核心系统。

VLM 必须接收：

1. Head RGB。
2. Optional Wrist RGB。
3. CV Perception Results。
4. Tactile Binary Contact。
5. Simplified Current Task State。
6. Available Skills from Skill Registry。
7. Recent Observation History when available。

VLM 输出 Scene Interpretation + Skill Recommendation。

例如：

```json
{
  "scene_state": "recoverable_misplacement",
  "recommended_skill": "recover_package",
  "reason": "A package appears to be displaced toward the robot-body side of the platform.",
  "target_object": "package_0",
  "requires_verification": true
}
```

这只是示例。

不要在 Python Controller 中直接用：

`if overlap < threshold: recovery`

来完全替代 VLM Skill Reasoning。

正确架构是：

Perception 产生证据 → VLM 根据场景与 Available Skills 推荐 → Controller 根据安全及前置条件决定是否执行。

Controller 可以基于确定性安全规则否决 VLM，但不能让 VLM 推荐成为一个永远不起作用的装饰性模块。

对于看不清楚的场景，应允许 VLM 选择 `stop_and_verify`。

---

## 12. Recovery Controller

实现状态机：

- `NORMAL`
- `FAILURE_CANDIDATE`
- `STOPPING`
- `RECOVERY_PRECHECK`
- `RECOVERING`
- `VERIFY_RECOVERY`
- `RESUME_PRECHECK`
- `STOPPED`
- `TASK_COMPLETED`

### Normal

运行 Normal Policy。

监测当前场景。

### Failure Candidate

出现异常候选。

VLM 分析当前状态，结合视觉及触觉证据推荐 Skill。

### Stopping

安全停止 Normal Policy。

取消尚未执行的 Action Chunks。

确认机器人已经进入允许切换策略的状态。

### Recovery Precheck

检查：

- 目标 Package 是否定位可靠。
- Recovery Policy 是否支持当前初始场景。
- 左右臂当前姿态。
- Gripper State。
- Potential Collision。
- Robot Safety Status。

### Recovering

执行 Recovery LingBot-VLA。

禁止 Normal Policy 同时发送机器人动作。

### Verify Recovery

使用新的 RGB、Tactile 和 Robot State 检查恢复结果。

不能单纯根据 Action Chunk 执行完毕判断恢复成功。

### Resume Precheck

判断 Normal VLA 是否可以从 Recovery 后的状态继续。

不能默认 Full-task Normal Policy 具备任意状态恢复能力。

### Stopped

停止所有新的动作提交。

等待人工介入。

### Task Completed

验证完整分拣任务已经完成。

---

## 13. VLA Policy Adapter 与真实模型切换

需要接入两个真实 LingBot-VLA Checkpoint。

检查现有推理和部署代码。

实现统一接口：

- `initialize`
- `get_observation`
- `infer_action`
- `execute_action`
- `request_stop`
- `wait_until_stopped`
- `get_execution_status`
- `shutdown`

接口名称可以根据现有项目调整。

关键要求：

- 防止两个 Policy 同时向 Robot 发送命令。
- 支持取消旧 Action Chunk。
- 验证机器人停止状态。
- 切换后更新 Observation。
- 验证新 Policy 的 Initial State。
- 不允许自动执行未经验证的 Reset Motion。

如果 GPU 资源不足，不能同时常驻两个模型，应实现安全的串行加载或模型切换。


---

## 15. 文件结构

建议创建：

```text
recovery_harness/
    config/
        package_sorting.yaml
        skill_registry.yaml
    perception/
        package_detector.py
        platform_geometry.py
        multi_view.py
        temporal_tracker.py
    tactile/
        binary_contact.py
    observation/
        synchronizer.py
    vlm/
        qwen_adapter.py
        scene_reasoner.py
        prompts.py
    policies/
        registry.py
        lingbot_adapter.py
    controller/
        recovery_fsm.py
        skill_router.py
    ros/
        kuavo_bridge.py
    tests/
    run.py
    README.md
```

根据现有仓库结构灵活调整。

优先复用已有实现。

不要重新开发完整的 Kuavo ROS SDK 或 LingBot-VLA Framework。

---

## 16. 可观测性和日志

实现结构化日志，包含：

- Current Controller State。
- Active Skill。
- Available Skills。
- Head/Wrist Image Timestamps。
- Tactile Contact。
- Robot State Validity。
- Package Location。
- Body-side Failure Evidence。
- VLM Output。
- Recommended Skill。
- Controller Selected Skill。
- Reason for Selection。
- Reason for Rejection。
- Skill Switching Status。
- Recovery Verification Result。
- Inference Latency。

区分：

- VLM Recommendation。
- Controller Decision。
- Actual VLA Execution。

不得把三者混为一谈。

---

## 17. ## Server Environment, Testing Budget and Token Efficiency

Codex is running directly on the GPU server, NOT on my local computer.

The server may have access to CUDA, ROS Noetic, the existing datasets, VLM checkpoints, and LingBot-VLA checkpoints.

You may perform limited tests to verify the implementation, but prioritize completing the entire VLM–VLA routing pipeline efficiently.

### 1. Allowed tests

You may perform:

- Repository inspection and checkpoint path verification.
- Python syntax checks and lightweight module imports.
- Small unit tests for geometry computation, tactile binary contact, Skill Registry, and FSM transitions.
- One short offline replay using a small number of frames.
- A small number of VLM inference calls to verify image input, dynamic skill descriptions, JSON output, and skill recommendations.
- One-time model loading checks if genuinely necessary and server resources permit.

### 2. Testing limits

By default:

- Maximum 6 real VLM inference calls in total.
- Maximum 1 short offline replay, with no more than 10 sampled frames.
- Do not load both VLM backends solely for comparison; select Qwen2.5-VL-7B as the default and keep Qwen3-VL-4B configurable.
- Avoid repeatedly loading the same VLM checkpoint.
- Do not run full-dataset evaluations.
- Do not train or fine-tune YOLO, VLM, or VLA models.
- Do not run large-scale benchmarks.
- Do not repeatedly run expensive failed commands without inspecting their cause.
- Do not run hardware motion tests automatically.

If additional high-cost tests are necessary, document them and leave commands for me to run manually.

### 3. Reuse previous perception work

I have already experimented with YOLO-World, CSRT tracking, geometric filtering, and segmentation.

YOLO-World package classification was not consistently reliable.

Inspect and reuse existing implementations instead of repeating detection experiments.

The primary perception design should support:

- Head RGB scene understanding.
- Optional left wrist RGB for partially visible recovery packages.
- Existing tracking and geometric evidence where reliable.
- Binary tactile contact.
- Explicit uncertainty when objects are occluded or out of view.

Do not make successful YOLO-World detection a mandatory prerequisite for VLM reasoning.

### 4. Prioritize implementation over testing

Work in this order:

1. Create or safely switch to branch `lingbot-vlm-vla-skills`.
2. Inspect existing inference, ROS, perception, and policy execution code.
3. Implement the complete Skill Registry, multi-view observation interface, VLM scene reasoner, and Skill Router.
4. Implement tactile processing, perception adapters, and Recovery Controller.
5. Integrate both real LingBot-VLA checkpoint configurations and policy adapters.
6. Implement offline, live dry-run, and gated real-robot modes.
7. Perform limited lightweight tests and at most the specified small VLM/offline smoke tests.
8. Report all integration limitations and provide real-robot validation commands.

Do not spend most of the work budget on debugging isolated perception examples.

### 5. Token efficiency

- Keep command output concise.
- Do not print complete ROS bags, model configurations, large source files, or full tracebacks unless necessary.
- Use targeted `rg`, `find`, `sed`, and bounded log excerpts.
- Avoid repeatedly searching the same directories.
- Prefer reusing existing modules.
- Do not produce long architectural explanations while implementing.
- Do not ask for confirmation after every implementation step.
- Work autonomously through the software implementation.
- Keep final reporting concise and practical.

### 6. Hardware safety

Real-robot motor execution remains disabled by default.

Do not start Kuavo motion, reset trajectories, or live policy execution without my explicit authorization.

Live sensor-only inspection is permitted when the existing ROS launch procedure is known to be safe and does not activate arm motion.

Never assume that a ROS launch command is sensor-only without verifying its actual contents.

### 7. Completion criteria

The goal is a complete and integrated software pipeline, not exhaustive testing.

Deliver:

- Implemented source code on `lingbot-vlm-vlm-skills`.
- Working configuration for both VLA checkpoint paths.
- Dynamic VLM Skill Registry and routing.
- Multi-view perception and tactile integration.
- Safe policy switching and Recovery FSM.
- Executable offline and live dry-run entry points.
- Real-robot deployment procedure with safety gates.
- Brief summary of tests actually performed.
- Remaining blockers and exact commands for manual validation.

Do not claim unexecuted tests or hardware integrations have passed.

---

## 18. 最终交付

在 `lingbot-vlm-vlm-skills` 分支中完成实现。

最后输出：

1. 当前 Git Branch，确认是 `lingbot-vlm-vlm-skills`。
2. 新建及修改的文件。
3. 复用的现有代码。
4. Normal VLA Checkpoint 解析情况。
5. Recovery VLA Checkpoint 解析情况。
6. VLM Model Adapter 实现情况。
7. Multi-view Perception 实现情况。
8. Tactile Binary Contact 实现情况。
9. Dynamic Skill Registry / VLM Routing 实现情况。
10. FSM 与 VLA Policy Switching 实现情况。
11. Offline Replay 启动命令。
12. Live Dry-run 启动命令。
13. Real Robot 部署步骤及安全限制。
14. 未执行的测试清单。
15. 尚未解决的 Integration Blockers。

**最终要求：不要只完成感知模块，也不要只写一个 VLM Prompt。必须把两个真实 VLA Policy、VLM Skill Registry、Skill Router、Controller 和 Kuavo ROS 接口集成为同一个 Pipeline。**

所有开发必须位于 `lingbot-vlm-vlm-skills` 分支。

不要在本地进行高负载测试。

不要自动执行真机动作。

不要修改训练好的 Checkpoint。

不要进行模型训练。

完成代码实现后，将需要在远程计算环境和真机上验证的部分明确列出。


## Additional requirement: Reuse existing YOLO-World + CSRT + Geometric Filtering

Previous experiments have already evaluated YOLO-World for package perception.

Observed limitations:
- YOLO-World frequently produces unreliable package class detections.
- Geometric filtering was added to reject irrelevant detections.
- CSRT tracking was used to maintain package tracks.
- Existing YOLO/SAM tracking and segmentation utilities may already be present in the repository.
- Recovery packages can be partially outside the head camera field of view, occluded, or heavily distorted by perspective.

Do not implement a new YOLO-World-only package detector and do not repeat the earlier detector experiments.

First inspect the repository for the existing YOLO-World, CSRT, geometric filtering, and segmentation/tracking implementations.

Reuse these components as optional visual evidence providers.

Implement a perception interface with:
- Candidate package localization from existing detectors.
- Tracking of previously confirmed package instances.
- Geometric filtering relative to the platform ROI.
- Tracking validity and age.
- Explicit uncertainty when detections or tracks become unreliable.
- Optional multi-view evidence from the left wrist camera.

Do not assume CSRT can recover a fully occluded or lost target indefinitely. Expired tracks must become UNKNOWN.

Do not use a bounding box of a partially visible package as if it represented its full geometry.

Do not make successful YOLO-World package classification a prerequisite for VLM scene reasoning.

The VLM should receive:
1. Original head camera image.
2. Optional left wrist image.
3. Available candidate detections and tracking history.
4. Platform geometry information.
5. Binary tactile contact.
6. Available VLA skills from the dynamic Skill Registry.

The VLM should reason about whether a package is misplaced, whether further observation is needed, and which available VLA skill should be recommended.

The deterministic Recovery Controller must still validate preconditions and prohibit unsupported or unsafe robot execution.

Do not train a new detector or run computationally expensive local tests.