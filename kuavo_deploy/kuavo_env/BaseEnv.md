# KuavoBaseRosEnv



## 1. 类结构

脚本主要包含两个类：

```python
class KuavoBaseRosEnv(gym.Env):
```

负责：

- ROS / Kuavo SDK 初始化
- Observation 获取
- Action Space 定义
- Policy action 检查
- 双臂关节控制
- 强脑灵巧手控制
- Reset / Step / Close

脚本中还包含：

```python
class LejuClaw:
```


---

# 2. 整体数据流

```text
ROS / Kuavo Robot
        ↓
    ObsBuffer
        ↓
     get_obs()
        ↓
observation.state
observation.images.*
        ↓
      Policy
        ↓
      action
        ↓
      step()
        ↓
  check_action()
        ↓
    exec_action()
        ↓
 ┌───────────────┬────────────────┐
 │               │                │
 ▼               ▼                ▼
左臂7关节      右臂7关节      左右灵巧手
 │               │                │
 └───────┬───────┘                │
         ▼                        ▼
   14D joint command      DexterousHand.control()
         │                        │
         └──────────┬─────────────┘
                    ▼
                Kuavo Robot
```

---

# 3. `__init__`

```python
def __init__(self, config: KuavoConfig)
```

环境初始化入口。

主要执行：

```text
_set_config()
    ↓
ROSManager()
ControlSignalManager()
CvBridge()
    ↓
_set_observation_space()
_set_action_space()
    ↓
_init_kuavo_sdk()
    ↓
_set_ros_topics()
```

---

# 4. `_set_config`

```python
def _set_config(self, config_kuavo_env)
```

读取环境配置，并保存为成员变量。

当前双臂 + 强脑模式下最关键的参数包括：

```python
self.which_arm
self.eef_type
self.control_mode
self.ros_rate
self.direct_to_wbc
self.qiangnao_dof_needed
self.arm_state_keys
self.limits
```

---

# 5. Observation Space

## `_set_observation_space`

```python
def _set_observation_space(self)
```

用于构造模型输入的 Gym observation space。

当前双臂模式下，state 主要由：

```text
左臂 joint
左 gripper
右臂 joint
右 gripper
```

组成。

---

## 5.1 `observation.state`

双臂模式下：

```python
obs_low.extend(
    joint_min[:7]
    + grip_min[:1]
    + joint_min[7:14]
    + grip_min[1:2]
)
```

因此 state 顺序固定为：

```text
index 0~6    左臂 7 个关节
index 7      左手状态

index 8~14   右臂 7 个关节
index 15     右手状态
```

总维度：

```text
16D
```

即：

```text
[
    left_arm_0,
    left_arm_1,
    left_arm_2,
    left_arm_3,
    left_arm_4,
    left_arm_5,
    left_arm_6,

    left_gripper,

    right_arm_0,
    right_arm_1,
    right_arm_2,
    right_arm_3,
    right_arm_4,
    right_arm_5,
    right_arm_6,

    right_gripper
]
```

---

## 5.2 Image Observation

相机数据由：

```python
self.obs_key_map
```

决定。

如果 key 中包含：

```text
cam
```

则构造：

```text
observation.images.<camera_key>
```

RGB 图像空间为：

```text
shape = (3, H, W)
dtype = uint8
range = [0, 255]
```

---

# 6. Action Space

## `_set_action_space`

```python
def _set_action_space(self)
```

用于定义 policy action 的维度和合法范围。

当前：

```text
which_arm = both
```

且在 joint control 下，action 由：

```text
左臂 7 joint
左手 1 value
右臂 7 joint
右手 1 value
```

组成。

---

## 6.1 Action 结构

双臂 action：

```text
index 0~6    left arm joints
index 7      left gripper

index 8~14   right arm joints
index 15     right gripper
```

总维度：

```text
16D
```

结构为：

```text
[
    L0, L1, L2, L3, L4, L5, L6,
    left_gripper,

    R0, R1, R2, R3, R4, R5, R6,
    right_gripper
]
```

其中：

```text
14D
```

属于机械臂关节控制，

```text
2D
```

属于左右强脑灵巧手控制。

---

# 7. Kuavo SDK 初始化

## `_init_kuavo_sdk`

```python
def _init_kuavo_sdk(self)
```

首先初始化：

```python
KuavoSDK().Init()
```

然后创建：

```python
self.robot = KuavoRobot()
self.robot_state = KuavoRobotState()
```

其中：

```text
KuavoRobot
```

负责机器人控制，

```text
KuavoRobotState
```

负责机器人状态读取。

---

# 8. ROS / 强脑灵巧手初始化

## `_set_ros_topics`

```python
def _set_ros_topics(self)
```

在：

```yaml
eef_type: qiangnao
```

条件下：

```python
self.qiangnao = DexterousHand()
```

因端执行器控制最终会走：

```python
DexterousHand.control()
```

最后：

```python
self.obs_buffer.wait_buffer_ready()
```

等待机器人状态与传感器 observation 准备完成。

---

# 9. Reset

## `reset`

```python
def reset(self, **kwargs)
```

Reset 流程：

```text
_enter_external_control_mode()
        ↓
_reset_head()
        ↓
_reset_eef()
        ↓
_compute_average_state()
        ↓
初始化 cur_state
        ↓
初始化 cur_joint_angles_action
        ↓
get_obs()
```

---

# 10. `_enter_external_control_mode`

```python
def _enter_external_control_mode(self)
```

调用：

```python
self.robot.set_external_control_arm_mode()
```

将机械臂切换为外部控制模式。

之后根据：

```python
direct_to_wbc
```

决定是否进一步切换 WBC 控制模式。

---

# 11. `_reset_eef`

```python
def _reset_eef(self)
```
进入：

```python
self.qiangnao.control(
    target_positions=[
        0, 100, 0, 0, 0, 0,
        0, 100, 0, 0, 0, 0
    ],
    target_velocities=None,
    target_torques=None
)
```

这里共包含：

```text
左手 6 个控制量
+
右手 6 个控制量
```

用于将左右灵巧手恢复到预定义初始状态。

---

# 12. `_compute_average_state`

```python
def _compute_average_state(self, average_num=10)
```

Reset 时连续读取多次机器人状态。

每次：

```python
state = self.get_obs()
```

同时获取当前机械臂关节：

```python
fk_joint_angles = self._get_init_joint_angles(
    self.arm_state["joint_q"]
)
```

最后计算平均：

```python
avg_state
avg_joint
```

返回：

```python
{
    "state": avg_state,
    "joint_action": avg_joint,
}
```

---

# 13. `_get_init_joint_angles`

```python
def _get_init_joint_angles(self, joint_q)
```

由于：

```text
which_arm = both
```

直接返回：

```python
np.array(joint_q)
```

即完整的：

```text
14D 双臂关节状态
```

不需要补另一侧机械臂。

---

# 14. `cur_state`

Reset 后：

```python
self.cur_state = avg_data["state"]
```

表示当前环境保存的 observation state。

在双臂模式下，典型结构为：

```text
左臂 7D
左手 1D
右臂 7D
右手 1D
```

总计：

```text
16D
```

---

# 15. `cur_joint_angles_action`

Reset 后：

```python
self.cur_joint_angles_action = avg_data["joint_action"]
```

表示完整双臂关节角。

结构：

```text
左臂 7D
+
右臂 7D
```

总计：

```text
14D
```

不包含左右灵巧手控制量。

---

# 16. `check_action`

```python
def check_action(self, action, mode="default")
```

负责检查 policy action。

首先检查：

```text
action length
```

是否与：

```python
self.action_space
```

一致。


之后检查 action 是否超出：

```python
self.action_space.low
self.action_space.high
```

如果超限：

```python
action = np.clip(
    action,
    self.action_space.low,
    self.action_space.high
)
```

将 action 裁剪到合法范围。

---

# 17. `step`

```python
def step(self, action)
```

是整个环境的主要动作执行入口。

双臂 + 强脑模式下主流程为：

```text
16D action
    ↓
check_action()
    ↓
提取 14D arm joint
    ↓
exec_action()
    ↓
控制双臂
    ↓
控制双手
    ↓
get_obs()
    ↓
compute_reward()
    ↓
return
```

---

# 18. 双臂关节提取

在 `step()` 中：

```python
self.cur_joint_angles_action = np.concatenate(
    (action[:7], action[8:15]),
    axis=0
)
```

即：

```text
action[0:7]
→ 左臂 7D

action[8:15]
→ 右臂 7D
```

跳过：

```text
action[7]
action[15]
```

因为这两个位置属于左右灵巧手。

最终得到：

```text
14D
```

双臂关节 action。

---

# 19. 普通控制路径

当：

```python
direct_to_wbc == False
```

执行：

```python
self.exec_action(action)
self.rate.sleep()
```

即 policy action 直接进入实际控制函数。

---

# 20. Direct-to-WBC 路径

如果：

```python
direct_to_wbc == True
```

则会在 action 下发前进行插值和平滑。


# 21. `exec_action`

```python
def exec_action(self, action)
```

双臂 + 强脑模式下，首先拆分：

```python
left_joints = action[:7]
left_eef = action[7]

right_joints = action[8:15]
right_eef = action[15]
```

然后构造：

```python
target_position = np.concatenate(
    (left_joints, right_joints),
    axis=0
)
```

得到：

```text
14D
```

双臂 joint command。

---

# 22. 双臂关节下发

随后执行：

```python
self.safe_control_arm(target_position)
```

内部调用：

```python
self.robot.control_arm_joint_positions(target_position)
```

即：

```text
14D target_position
        ↓
KuavoRobot
        ↓
左右机械臂
```

---

# 23. 强脑灵巧手控制

机械臂控制完成后：

```python
self._control_eef(left_eef, right_eef)
```

当前：

```text
eef_type = qiangnao
```

因此进入强脑控制分支。

首先要求：

```python
self.qiangnao_dof_needed == 1
```

即 policy 每只手只提供一个控制量。

---

## 23.1 Policy Hand Action

输入：

```text
left_eef
right_eef
```

均为单个 scalar。

先执行：

```python
tem_left = left_eef * 100
tem_right = right_eef * 100
```

---

## 23.2 扩展到灵巧手控制量

然后构造：

```python
target_positions = np.array([
    tem_left,
    100,
    tem_left,
    tem_left,
    tem_left,
    tem_left,

    tem_right,
    100,
    tem_right,
    tem_right,
    tem_right,
    tem_right
])
```

共：

```text
12D
```

即：

```text
左手 6D
+
右手 6D
```

---

## 23.3 最终下发

调用：

```python
self.qiangnao.control(
    target_positions=target_positions
)
```

因此手部 action 数据流为：

```text
Policy
    ↓
2D hand action
[left_eef, right_eef]
    ↓
× 100
    ↓
扩展
    ↓
12D finger target
    ↓
DexterousHand.control()
    ↓
左右强脑灵巧手
```

---

# 24. `safe_control_arm`

```python
def safe_control_arm(self, target_position)
```

负责安全调用：

```python
self.robot.control_arm_joint_positions(
    target_position
)
```

如果机器人当前状态不允许手臂控制，则捕获特定 RuntimeError。

正常情况下输入为：

```text
14D 双臂 joint position
```

---

# 25. `get_obs`

```python
def get_obs(self)
```

负责获取模型 observation。

首先从：

```python
ObsBuffer
```

读取当前 ROS 数据。

如果启用：

```python
frame_alignment
```

则调用：

```python
get_aligned_obs()
```

否则：

```python
get_latest_obs()
```

---

# 26. State 拼接

状态数据首先进入：

```python
self.arm_state
```

然后按照：

```python
self.arm_state_keys
```

逐项处理。

双臂模式下：

```python
mid = len(data) // 2
```

将数据拆成：

```text
left
right
```

例如：

```text
joint_q
14D
→
left 7D
right 7D
```

以及：

```text
gripper
2D
→
left 1D
right 1D
```

---

# 27. 最终 Observation State

最终按照：

```python
arm_data["left"] + arm_data["right"]
```

拼接。

因此典型顺序为：

```text
left joint 7D
left gripper 1D
right joint 7D
right gripper 1D
```

得到：

```text
16D
```

最后：

```python
torch.from_numpy(...).float().unsqueeze(0)
```

转成：

```text
shape = [1, 16]
```

---


# 28. `close`

```python
def close(self)
```

关闭：

```text
ObsBuffer subscribers
ROSManager
ControlSignalManager
```

并清理缓存。

---



# 29. 核心函数关系

```text
__init__()
 ├── _set_config()
 ├── _set_observation_space()
 ├── _set_action_space()
 ├── _init_kuavo_sdk()
 └── _set_ros_topics()
        └── DexterousHand()

reset()
 ├── _enter_external_control_mode()
 ├── _reset_head()
 ├── _reset_eef()
 │      └── qiangnao.control()
 ├── _compute_average_state()
 └── get_obs()

step(action)
 ├── check_action()
 ├── 提取双臂14D joint
 ├── exec_action()
 │      ├── safe_control_arm()
 │      │      └── control_arm_joint_positions()
 │      │
 │      └── _control_eef()
 │             └── qiangnao.control()
 │
 ├── get_obs()
 └── compute_reward()
```
