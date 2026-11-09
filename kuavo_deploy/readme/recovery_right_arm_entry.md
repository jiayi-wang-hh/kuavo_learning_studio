# Recovery 前的右臂平滑让位

此步骤仅添加到自动 Demo（eval.py 选项9），不改变普通 real_single_test、不改变 VLA 权重或动作，也不修改 bag 和数据集。

## 参考姿态

配置位于 `configs/deploy/lingbot_skill_demo.yaml` 的 `recovery_right_arm_entry`，默认启用。

参考 bag：`/media/data/training_data/task3_package_pick_and_place_failure/package_below_target_zone/vr_record_2026-10-22-12-10-32.bag`。

取 bag 第一条 `/sensors_data_raw` 的**实测关节位置（弧度）**，按部署配置的关节映射读取：当前4pro左臂为 `[12:19]`、右臂为 `[19:26]`。不使用 `/kuavo_arm_traj` 的角度命令，不播放整个 bag，不读取手部或头部动作。第一帧无效、话题缺失或目标超出配置关节范围则退出，不悄悄换用后续帧。

此次只读检查得到右臂目标（rad）：

```text
[-0.070032, -0.127779, -0.090290, -1.803616, -0.101735, 0.388311, 0.279653]
```

这是用户指定 bag 的目标，不代表已验证为无碰撞姿态，也不代表它与 Recovery 训练入口分布相同。

## 执行顺序

1. 原有连续 Recovery 投票触发，停止 Normal、清空动作缓存、等待双臂稳定。
2. 原有停稳后复核通过，确认应进入 Recovery。
3. 记录 `RIGHT_ARM_PREPOSITIONING`。从当前实测右臂位置出发，使用五次平滑插值到参考位置，起止速度和加速度均为零。
4. 保持左臂停止时最后已发送的目标（不存在时用当前实测左臂位置）；整个让位步骤只调用现有 `safe_control_arm()`，不发送任何手部开合命令，不调用 VLA 推理。
5. 右臂反馈在目标误差范围内持续稳定，并通过原有双臂停稳检查后，记录 `RIGHT_ARM_READY`。
6. 清除让位前票数，记录 `VERIFY_AFTER_RIGHT_ARM_ENTRY`。只使用让位后新图像：连续3次 recovery 才清空 Recovery 模型状态、用最新真实三路图像和双臂反馈开始 Recovery；连续3次 normal 取消 Recovery；不稳定则按复核超时退出。

180秒 Recovery 计时从真正开始 Recovery 技能时计算，不包含右臂让位时间。取消进入不占 Recovery 次数。普通 Normal 启动时不自动让位。

## 参数和保护

- `duration: 3.0`：最短时间，不保证固定3秒；按最大关节距离自动延长，以满足速度和加速度上限。
- `rate_hz: 100`、`max_velocity: 0.25` rad/s、`max_acceleration: 0.5` rad/s²。实际线程/ROS延迟会使运动更慢，不跳过插值点追赶进度。
- `arrival_tolerance: 0.03` rad，`stable_seconds: 0.5` 秒：使用实测反馈确认到位，不以“发完命令”当作到位。
- `arrival_timeout: 10.0` 秒：最后插值点发送后等待反馈的上限。
- `left_hold_tolerance: 0.10` rad：左臂明显偏离保持目标则中止。
- 每次让位指令都经过原 SafetySupervisor，保留上一条命令历史及跟踪检查，不跳过、不放宽。使用原反馈时效检查，过期、停止信号、监控持续故障会阻止继续下发。
- 操作员暂停会停止发送后续插值点，恢复时继续下一个小步，不跳到按墙钟时间推算的较远位置。此处暂停不等同硬件急停，已经提交的目标可能继续执行。

平滑插值不提供碰撞检查，也不保证避开左臂、机器人身体、桌面或手中物体。实际路径尚未实机验证；不要通过关闭 SafetySupervisor 来绕过拒绝。

此改动只保证进入 Recovery **之前**让位，不冻结 Recovery checkpoint 随后输出的右臂动作。

## 使用和记录

保持已加载的 Qwen、Normal 和 Recovery 服务，无需重载权重；下一次启动 `eval.py` 选项9生效。不要同时运行两个控制客户端。

每次 Demo 的 `decisions.jsonl` 新增：`right_arm_entry_reference`、`right_arm_entry_started`、逐步 `right_arm_entry_command`、`right_arm_entry_arrived`。记录 bag 来源、目标、左臂保持值、实际插值时长、下发指令与最终反馈误差。

如不需要该流程，将 `recovery_right_arm_entry.enabled` 设为 false。测试仅运行离线单元测试、只读 bag 提取和无模型无动作 preflight，未启动机器人或模型推理。
