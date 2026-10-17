# Kuavo 真机推理日志：STATE / Action 偏差标注

> 本文档仅基于已有日志进行整理，没有重新运行模型或机器人。  
> `both + qiangnao` 下 16D 顺序：左臂 7D、左手 1D、右臂 7D、右手 1D。  
> 本文将机械臂关节 `|Action - STATE| ≥ 0.5 rad` 标记为 **重大偏差**。

## 1. 关键结论

- **最显著的偏差集中在右臂。**
- **右臂 J4（action index 11）偏差最大**：第 1 步约 `+1.862 rad`。
- **右臂 J1（action index 8）第 1 步约 `+0.999 rad`**。
- **右臂 J7（action index 14）第 1 步约 `-0.523 rad`**，刚超过本文重大偏差阈值。
- 后续机器人 STATE 明显向这些 action 方向移动，因此 J1/J4 的偏差逐步下降，但在日志结束前仍然很大。
- 左右 gripper action 中出现微小负值，被 `check_action()` 裁剪为 `0`；这是越界裁剪，不是本文定义的机械臂重大偏差。

## 2. Step-by-step STATE → Action 偏差

### Step 1

| idx | 维度 | STATE | Action | Δ = Action - STATE | 标记 |
|---:|---|---:|---:|---:|---|
| 0 | 左臂 J1 / shoulder pitch | -0.133158 | -0.118787 | +0.014371 |  |
| 1 | 左臂 J2 / shoulder roll | 0.130083 | 0.086503 | -0.043580 |  |
| 2 | 左臂 J3 / arm yaw | -0.430687 | -0.285422 | +0.145264 |  |
| 3 | 左臂 J4 / forearm pitch | -1.874952 | -1.880802 | -0.005850 |  |
| 4 | 左臂 J5 / wrist yaw | -0.034716 | -0.241595 | -0.206879 |  |
| 5 | 左臂 J6 / wrist pitch | 0.124363 | -0.008262 | -0.132626 |  |
| 6 | 左臂 J7 / wrist roll | 0.363547 | 0.390498 | +0.026951 |  |
| 7 | 左手 gripper | 0.000000 | -0.000000 | -0.000000 |  |
| **8** | **右臂 J1 / shoulder pitch** | **-0.009060** | **0.989915** | **+0.998975** | **⚠️ 重大偏差** |
| 9 | 右臂 J2 / shoulder roll | -0.150683 | 0.000939 | +0.151622 |  |
| 10 | 右臂 J3 / arm yaw | -0.061035 | -0.000000 | +0.061035 |  |
| **11** | **右臂 J4 / forearm pitch** | **-1.861982** | **-0.000000** | **+1.861982** | **⚠️ 重大偏差** |
| 12 | 右臂 J5 / wrist yaw | -0.264361 | 0.000939 | +0.265300 |  |
| 13 | 右臂 J6 / wrist pitch | 0.022891 | -0.127889 | -0.150780 |  |
| **14** | **右臂 J7 / wrist roll** | **0.391772** | **-0.131277** | **-0.523049** | **⚠️ 重大偏差** |
| 15 | 右手 gripper | 0.000000 | -0.000448 | -0.000448 |  |

### Step 2

| idx | 维度 | STATE | Action | Δ = Action - STATE | 标记 |
|---:|---|---:|---:|---:|---|
| 0 | 左臂 J1 / shoulder pitch | -0.133159 | -0.119810 | +0.013349 |  |
| 1 | 左臂 J2 / shoulder roll | 0.130085 | 0.086399 | -0.043686 |  |
| 2 | 左臂 J3 / arm yaw | -0.430688 | -0.284994 | +0.145694 |  |
| 3 | 左臂 J4 / forearm pitch | -1.874953 | -1.879855 | -0.004903 |  |
| 4 | 左臂 J5 / wrist yaw | -0.034713 | -0.246866 | -0.212152 |  |
| 5 | 左臂 J6 / wrist pitch | 0.124365 | -0.007525 | -0.131890 |  |
| 6 | 左臂 J7 / wrist roll | 0.363546 | 0.393002 | +0.029455 |  |
| 7 | 左手 gripper | 0.000000 | -0.000000 | -0.000000 |  |
| **8** | **右臂 J1 / shoulder pitch** | **-0.009061** | **0.989915** | **+0.998976** | **⚠️ 重大偏差** |
| 9 | 右臂 J2 / shoulder roll | -0.150683 | -0.000000 | +0.150683 |  |
| 10 | 右臂 J3 / arm yaw | -0.061035 | -0.000000 | +0.061035 |  |
| **11** | **右臂 J4 / forearm pitch** | **-1.861984** | **-0.000000** | **+1.861984** | **⚠️ 重大偏差** |
| 12 | 右臂 J5 / wrist yaw | -0.264363 | -0.000000 | +0.264363 |  |
| 13 | 右臂 J6 / wrist pitch | 0.022891 | -0.130977 | -0.153868 |  |
| **14** | **右臂 J7 / wrist roll** | **0.391774** | **-0.131723** | **-0.523498** | **⚠️ 重大偏差** |
| 15 | 右手 gripper | 0.000000 | -0.004736 | -0.004736 |  |

### Step 3

| idx | 维度 | STATE | Action | Δ = Action - STATE | 标记 |
|---:|---|---:|---:|---:|---|
| 0 | 左臂 J1 / shoulder pitch | -0.125907 | -0.120833 | +0.005074 |  |
| 1 | 左臂 J2 / shoulder roll | 0.125597 | 0.086399 | -0.039199 |  |
| 2 | 左臂 J3 / arm yaw | -0.419322 | -0.284994 | +0.134328 |  |
| 3 | 左臂 J4 / forearm pitch | -1.871921 | -1.882696 | -0.010775 |  |
| 4 | 左臂 J5 / wrist yaw | -0.065422 | -0.248974 | -0.183553 |  |
| 5 | 左臂 J6 / wrist pitch | 0.101625 | -0.006788 | -0.108413 |  |
| 6 | 左臂 J7 / wrist roll | 0.367315 | 0.395818 | +0.028503 |  |
| 7 | 左手 gripper | 0.000000 | -0.000000 | -0.000000 |  |
| **8** | **右臂 J1 / shoulder pitch** | **0.162057** | **0.989915** | **+0.827858** | **⚠️ 重大偏差** |
| 9 | 右臂 J2 / shoulder roll | -0.130436 | 0.000939 | +0.131375 |  |
| 10 | 右臂 J3 / arm yaw | -0.060243 | -0.000000 | +0.060243 |  |
| **11** | **右臂 J4 / forearm pitch** | **-1.658499** | **-0.000000** | **+1.658499** | **⚠️ 重大偏差** |
| 12 | 右臂 J5 / wrist yaw | -0.238298 | 0.003756 | +0.242054 |  |
| 13 | 右臂 J6 / wrist pitch | 0.016532 | -0.129433 | -0.145965 |  |
| 14 | 右臂 J7 / wrist roll | 0.290745 | -0.132393 | -0.423138 |  |
| 15 | 右手 gripper | 0.000000 | -0.003128 | -0.003128 |  |

### Step 4

| idx | 维度 | STATE | Action | Δ = Action - STATE | 标记 |
|---:|---|---:|---:|---:|---|
| 0 | 左臂 J1 / shoulder pitch | -0.101857 | -0.119299 | -0.017442 |  |
| 1 | 左臂 J2 / shoulder roll | 0.109085 | 0.086608 | -0.022477 |  |
| 2 | 左臂 J3 / arm yaw | -0.391714 | -0.285422 | +0.106292 |  |
| 3 | 左臂 J4 / forearm pitch | -1.870374 | -1.881749 | -0.011375 |  |
| 4 | 左臂 J5 / wrist yaw | -0.113770 | -0.249818 | -0.136048 |  |
| 5 | 左臂 J6 / wrist pitch | 0.070525 | -0.005314 | -0.075839 |  |
| 6 | 左臂 J7 / wrist roll | 0.371947 | 0.396444 | +0.024497 |  |
| 7 | 左手 gripper | 0.000000 | 0.001878 | +0.001878 |  |
| **8** | **右臂 J1 / shoulder pitch** | **0.414737** | **0.989915** | **+0.575178** | **⚠️ 重大偏差** |
| 9 | 右臂 J2 / shoulder roll | -0.099098 | -0.000000 | +0.099098 |  |
| 10 | 右臂 J3 / arm yaw | -0.059503 | -0.000000 | +0.059503 |  |
| **11** | **右臂 J4 / forearm pitch** | **-1.311685** | **0.001878** | **+1.313563** | **⚠️ 重大偏差** |
| 12 | 右臂 J5 / wrist yaw | -0.169651 | -0.000000 | +0.169651 |  |
| 13 | 右臂 J6 / wrist pitch | -0.045469 | -0.129433 | -0.083964 |  |
| 14 | 右臂 J7 / wrist roll | 0.195990 | -0.132616 | -0.328607 |  |
| 15 | 右手 gripper | 0.000000 | -0.003664 | -0.003664 |  |

## 3. 重大偏差汇总

| Step | index | 关节 | STATE | Action | Δ |
|---:|---:|---|---:|---:|---:|
| 1 | 8 | **右臂 J1 / shoulder pitch** | -0.009060 | 0.989915 | **+0.998975** |
| 1 | 11 | **右臂 J4 / forearm pitch** | -1.861982 | -0.000000 | **+1.861982** |
| 1 | 14 | **右臂 J7 / wrist roll** | 0.391772 | -0.131277 | **-0.523049** |
| 2 | 8 | **右臂 J1 / shoulder pitch** | -0.009061 | 0.989915 | **+0.998976** |
| 2 | 11 | **右臂 J4 / forearm pitch** | -1.861984 | -0.000000 | **+1.861984** |
| 2 | 14 | **右臂 J7 / wrist roll** | 0.391774 | -0.131723 | **-0.523498** |
| 3 | 8 | **右臂 J1 / shoulder pitch** | 0.162057 | 0.989915 | **+0.827858** |
| 3 | 11 | **右臂 J4 / forearm pitch** | -1.658499 | -0.000000 | **+1.658499** |
| 4 | 8 | **右臂 J1 / shoulder pitch** | 0.414737 | 0.989915 | **+0.575178** |
| 4 | 11 | **右臂 J4 / forearm pitch** | -1.311685 | 0.001878 | **+1.313563** |

### 偏差随 Step 的变化

- **右臂 J1 / shoulder pitch (index 8)**： Step 1: `+0.999` → Step 2: `+0.999` → Step 3: `+0.828` → Step 4: `+0.575`
- **右臂 J4 / forearm pitch (index 11)**： Step 1: `+1.862` → Step 2: `+1.862` → Step 3: `+1.658` → Step 4: `+1.314`
- **右臂 J7 / wrist roll (index 14)**： Step 1: `-0.523` → Step 2: `-0.523` → Step 3: `-0.423` → Step 4: `-0.329`

## 4. Action 越界裁剪

日志显示 gripper 合法范围为 `[0, 1]`。模型多次输出非常小的负数，因此被裁剪到 `0`。例如第 1 步：
```text
left gripper  action[7]  = -4.80785065e-11  -> clip -> 0
right gripper action[15] = -4.48252307e-04 -> clip -> 0
```
这类 gripper 越界与右臂关节的 STATE / Action 大幅跳变是两个不同问题。

## 5. 原始日志

```text
/opt/ros/noetic/lib/python3/dist-packages/rosbag/bag.py:1568: SyntaxWarning: invalid escape sequence '\d'
  matches = re.match("#ROS(.*) V(\d).(\d)", version_line)

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.5.3 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "/home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py", line 23, in <module>
    from kuavo_deploy.kuavo_env.KuavoBaseRosEnv import KuavoBaseRosEnv
  File "/home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/kuavo_env/__init__.py", line 2, in <module>
    from kuavo_deploy.kuavo_env.KuavoSimEnv import KuavoSimEnv
  File "/home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/kuavo_env/KuavoSimEnv.py", line 4, in <module>
    from cv_bridge import CvBridge
  File "/opt/ros/noetic/lib/python3/dist-packages/cv_bridge/__init__.py", line 6, in <module>
    from cv_bridge.boost.cv_bridge_boost import cvtColorForDisplay, getCvType
AttributeError: _ARRAY_API not found
[33mWarning: /opt/ros/noetic/lib/python3/dist-packages/actionlib/goal_id_generator.py:46 - invalid escape sequence '\p'[0m
[33mWarning: /opt/ros/noetic/lib/python3/dist-packages/actionlib/goal_id_generator.py:56 - invalid escape sequence '\p'[0m
[33mWarning: /opt/ros/noetic/lib/python3/dist-packages/tf/transformations.py:59 - invalid escape sequence '\*'[0m
[33mWarning: /opt/ros/noetic/lib/python3/dist-packages/tf/transformations.py:869 - invalid escape sequence '\*'[0m
[33mWarning: /home/kuavo/miniforge3/envs/kls_jiayi/lib/python3.12/site-packages/kuavo_humanoid_sdk/kuavo/robot_climbstair.py:9 - A NumPy version >=1.23.5 and <2.3.0 is required for this version of SciPy (detected version 2.5.3)[0m
2026-09-25 11:16:14,200 - 🦾 ROBOT | INFO: 使用配置文件: configs/deploy/deploy.yaml | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:14,200 - 🦾 ROBOT | INFO: 执行任务: run | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:14,205 - 🦾 ROBOT | INFO: 📡 信号处理器已设置: | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:14,205 - 🦾 ROBOT | INFO:   SIGUSR1 (kill -USR1): 暂停/恢复机械臂运动 | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:14,205 - 🦾 ROBOT | INFO:   SIGUSR2 (kill -USR2): 停止机械臂运动 | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:14,205 - 🦾 ROBOT | INFO: 🆔 当前进程ID: 84764 | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:14,205 - 🦾 ROBOT | INFO: 💡 使用以下命令控制机械臂运动: | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:14,205 - 🦾 ROBOT | INFO:    暂停/恢复: kill -USR1 84764 | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:14,205 - 🦾 ROBOT | INFO:    停止运动: kill -USR2 84764 | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:14,603 - 🦾 ROBOT | INFO: Subscribed to /cam_h/color/image_raw/compressed for key 'head_cam_h' | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:14,605 - 🦾 ROBOT | INFO: Subscribed to /cam_l/color/image_raw/compressed for key 'wrist_cam_l' | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:14,606 - 🦾 ROBOT | INFO: Subscribed to /cam_r/color/image_raw/compressed for key 'wrist_cam_r' | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:14,607 - 🦾 ROBOT | INFO: Subscribed to /sensors_data_raw for key 'joint_q' | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:14,608 - 🦾 ROBOT | INFO: Subscribed to /dexhand/state for key 'gripper' | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
[33mWarning: /home/kuavo/miniforge3/envs/kls_jiayi/lib/python3.12/site-packages/gymnasium/spaces/box.py:231 - [33mWARN: Box low's precision lowered by casting to float32, current low.dtype=float64[0m[0m
[33mWarning: /home/kuavo/miniforge3/envs/kls_jiayi/lib/python3.12/site-packages/gymnasium/spaces/box.py:297 - [33mWARN: Box high's precision lowered by casting to float32, current high.dtype=float64[0m[0m
2026-09-25 11:16:19,866 - INFO - [WheelArmROSControl] ROS接口初始化成功
2026-09-25 11:16:19,866 - INFO - [WheelArmROSControl] 轮臂ROS控制模块初始化完成
2026-09-25 11:16:19,866 - INFO - Using standard mode, subscribing to: /sensors_data_raw
2026-09-25 11:16:20,414 - 🦾 ROBOT | INFO: ✅ All buffers ready! | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:20,414 - 🦾 ROBOT | INFO: Inializing done! | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:20,415 - 🦾 ROBOT | INFO: 机械臂初始化成功 | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:20,415 - 🦾 ROBOT | INFO: 开始执行任务: run | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
kuavo-humanoid-sdk log_file: ./log/kuavo_humanoid_sdk/kuavo_humanoid_sdk.log
[LoggerClient] 使用 localhost 构造连接地址: ws://localhost:8889
[LoggerClient] 连接失败: [Errno 111] Connect call failed ('127.0.0.1', 8889)
[LoggerClient] 连接日志服务器超时
[35m当前为双足模型[0m
[35m当前为双足模型[0m
Inializing done!

Evaluating model:   0%|          | 0/10 [00:00<?, ?episode/s]2026-09-25 11:16:22,827 - 🦾 ROBOT | INFO: STATE: contained ['joint_q', 'gripper'], concated value: [-0.13316158  0.13008375 -0.43068744 -1.87495211 -0.03471571  0.12436243
  0.36354624  0.         -0.00905886 -0.15068186 -0.06103489 -1.86198269
 -0.2643612   0.02288906  0.39177411  0.        ] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:22,830 - 🦾 ROBOT | INFO: STATE: contained ['joint_q', 'gripper'], concated value: [-0.13315764  0.13008359 -0.43068726 -1.87495234 -0.03471447  0.12436249
  0.3635447   0.         -0.00906291 -0.15068232 -0.06103537 -1.8619826
 -0.26436046  0.02288879  0.39177362  0.        ] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:22,831 - 🦾 ROBOT | INFO: STATE: contained ['joint_q', 'gripper'], concated value: [-0.13315766  0.13008356 -0.43068695 -1.87495247 -0.03471443  0.12436195
  0.36354506  0.         -0.00905774 -0.15068258 -0.06103536 -1.86198268
 -0.26436116  0.02288901  0.39177399  0.        ] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:22,832 - 🦾 ROBOT | INFO: STATE: contained ['joint_q', 'gripper'], concated value: [-0.13315772  0.13008352 -0.43068654 -1.87495265 -0.03471438  0.12436122
  0.36354555  0.         -0.00905    -0.15068294 -0.06103536 -1.86198278
 -0.26436208  0.02288932  0.39177448  0.        ] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:22,834 - 🦾 ROBOT | INFO: STATE: contained ['joint_q', 'gripper'], concated value: [-0.13315774  0.13008351 -0.43068641 -1.87495271 -0.03471405  0.12436089
  0.36354602  0.         -0.00904543 -0.15068295 -0.06103515 -1.86198302
 -0.26436311  0.02288889  0.39177516  0.        ] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:22,835 - 🦾 ROBOT | INFO: STATE: contained ['joint_q', 'gripper'], concated value: [-0.13315776  0.1300835  -0.43068631 -1.87495275 -0.03471435  0.12436082
  0.36354582  0.         -0.0090436  -0.15068314 -0.06103536 -1.86198283
 -0.26436259  0.0228895   0.39177475  0.        ] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:22,836 - 🦾 ROBOT | INFO: STATE: contained ['joint_q', 'gripper'], concated value: [-0.13315777  0.13008349 -0.43068623 -1.87495278 -0.03471462  0.12436078
  0.36354563  0.         -0.00904252 -0.15068329 -0.06103554 -1.86198267
 -0.26436211  0.02289001  0.39177438  0.        ] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:22,837 - 🦾 ROBOT | INFO: STATE: contained ['joint_q', 'gripper'], concated value: [-0.13315778  0.13008359 -0.43068593 -1.87495293 -0.03471477  0.1243598
  0.36354476  0.         -0.00903977 -0.15068381 -0.06103609 -1.86198234
 -0.26436147  0.02289112  0.39177398  0.        ] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:22,839 - 🦾 ROBOT | INFO: STATE: contained ['joint_q', 'gripper'], concated value: [-0.13315779  0.13008348 -0.43068611 -1.87495284 -0.0347151   0.12436074
  0.36354531  0.         -0.00904286 -0.15068356 -0.06103586 -1.86198237
 -0.26436126  0.02289091  0.39177371  0.        ] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:22,840 - 🦾 ROBOT | INFO: STATE: contained ['joint_q', 'gripper'], concated value: [-0.13315779  0.13008338 -0.43068627 -1.87495276 -0.03471537  0.12436156
  0.36354581  0.         -0.00904667 -0.15068333 -0.06103564 -1.86198241
 -0.26436111  0.02289071  0.3917735   0.        ] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:22,841 - 🦾 ROBOT | INFO: STATE: contained ['joint_q', 'gripper'], concated value: [-0.1331578   0.13008305 -0.43068659 -1.87495244 -0.03471625  0.12436349
  0.36354678  0.         -0.00906041 -0.15068272 -0.06103545 -1.86198236
 -0.2643608   0.02289106  0.39177249  0.        ] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
set_external_control_arm_mode KuavoArmCtrlMode.ExternalControl


Episode 1:   0%|          | 0/200 [00:00<?, ?step/s][A2026-09-25 11:16:23,603 - 🧠 MODEL | INFO: action infer time: 0.762s | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,604 - 🦾 ROBOT | INFO: action: [-1.18787155e-01  8.65034834e-02 -2.85422236e-01 -1.88080227e+00
 -2.41595075e-01 -8.26238282e-03  3.90498072e-01 -4.80785065e-11
  9.89914775e-01  9.39032645e-04 -4.80785065e-11 -4.80785065e-11
  9.39032645e-04 -1.27889290e-01 -1.31276771e-01 -4.48252307e-04] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,605 - 🦾 ROBOT | WARNING: action out of range, action: [-1.18787155e-01  8.65034834e-02 -2.85422236e-01 -1.88080227e+00
 -2.41595075e-01 -8.26238282e-03  3.90498072e-01 -4.80785065e-11
  9.89914775e-01  9.39032645e-04 -4.80785065e-11 -4.80785065e-11
  9.39032645e-04 -1.27889290e-01 -1.31276771e-01 -4.48252307e-04], low: [-3.14 -3.14 -3.14 -3.14 -3.14 -3.14 -3.14  0.   -3.14 -3.14 -3.14 -3.14
 -3.14 -3.14 -3.14  0.  ], high: [3.14 3.14 3.14 3.14 3.14 3.14 3.14 1.   3.14 3.14 3.14 3.14 3.14 3.14
 3.14 1.  ] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,605 - 🦾 ROBOT | INFO: clip action: [-1.18787155e-01  8.65034834e-02 -2.85422236e-01 -1.88080227e+00
 -2.41595075e-01 -8.26238282e-03  3.90498072e-01  0.00000000e+00
  9.89914775e-01  9.39032645e-04 -4.80785065e-11 -4.80785065e-11
  9.39032645e-04 -1.27889290e-01 -1.31276771e-01  0.00000000e+00], check time: 0.001s | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,613 - 🦾 ROBOT | INFO: rate.sleep time: 0.007s | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,614 - 🦾 ROBOT | INFO: STATE: contained ['joint_q', 'gripper'], concated value: [-0.13315868  0.13008525 -0.4306878  -1.87495273 -0.03471347  0.12436456
  0.36354633  0.         -0.00906134 -0.15068278 -0.06103509 -1.86198433
 -0.26436273  0.02289076  0.39177426  0.        ] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,617 - 🦾 ROBOT | INFO: get obs time: 0.002s | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,620 - 🧠 MODEL | INFO: Step 1 time: 0.778s | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 


Episode 1:   0%|          | 0/200 [00:00<?, ?step/s, Reward=0.000, Status=Running, Total Reward=0.000][A

Episode 1:   0%|          | 1/200 [00:00<02:35,  1.28step/s, Reward=0.000, Status=Running, Total Reward=0.000][A2026-09-25 11:16:23,651 - 🧠 MODEL | INFO: action infer time: 0.030s | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,656 - 🦾 ROBOT | INFO: action: [-1.19809978e-01  8.63987952e-02 -2.84993976e-01 -1.87985539e+00
 -2.46865913e-01 -7.52521120e-03  3.93001676e-01 -4.80785065e-11
  9.89914775e-01 -4.80785065e-11 -4.80785065e-11 -4.80785065e-11
 -4.80785065e-11 -1.30976781e-01 -1.31723315e-01 -4.73618740e-03] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,658 - 🦾 ROBOT | WARNING: action out of range, action: [-1.19809978e-01  8.63987952e-02 -2.84993976e-01 -1.87985539e+00
 -2.46865913e-01 -7.52521120e-03  3.93001676e-01 -4.80785065e-11
  9.89914775e-01 -4.80785065e-11 -4.80785065e-11 -4.80785065e-11
 -4.80785065e-11 -1.30976781e-01 -1.31723315e-01 -4.73618740e-03], low: [-3.14 -3.14 -3.14 -3.14 -3.14 -3.14 -3.14  0.   -3.14 -3.14 -3.14 -3.14
 -3.14 -3.14 -3.14  0.  ], high: [3.14 3.14 3.14 3.14 3.14 3.14 3.14 1.   3.14 3.14 3.14 3.14 3.14 3.14
 3.14 1.  ] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,669 - 🦾 ROBOT | INFO: clip action: [-1.19809978e-01  8.63987952e-02 -2.84993976e-01 -1.87985539e+00
 -2.46865913e-01 -7.52521120e-03  3.93001676e-01  0.00000000e+00
  9.89914775e-01 -4.80785065e-11 -4.80785065e-11 -4.80785065e-11
 -4.80785065e-11 -1.30976781e-01 -1.31723315e-01  0.00000000e+00], check time: 0.013s | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,713 - 🦾 ROBOT | INFO: rate.sleep time: 0.042s | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,714 - 🦾 ROBOT | INFO: STATE: contained ['joint_q', 'gripper'], concated value: [-0.12590674  0.12559739 -0.41932192 -1.87192101 -0.06542161  0.10162451
  0.36731514  0.          0.16205657 -0.13043594 -0.06024342 -1.65849871
 -0.23829791  0.01653183  0.29074477  0.        ] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,714 - 🦾 ROBOT | INFO: get obs time: 0.001s | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,714 - 🧠 MODEL | INFO: Step 2 time: 0.092s | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 


Episode 1:   0%|          | 1/200 [00:00<02:35,  1.28step/s, Reward=0.000, Status=Running, Total Reward=0.000][A2026-09-25 11:16:23,735 - 🧠 MODEL | INFO: action infer time: 0.020s | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,735 - 🦾 ROBOT | INFO: action: [-1.20832801e-01  8.63987952e-02 -2.84993976e-01 -1.88269603e+00
 -2.48974234e-01 -6.78804005e-03  3.95818233e-01 -4.80785065e-11
  9.89914775e-01  9.39032645e-04 -4.80785065e-11 -4.80785065e-11
  3.75613081e-03 -1.29433036e-01 -1.32393137e-01 -3.12821195e-03] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,736 - 🦾 ROBOT | WARNING: action out of range, action: [-1.20832801e-01  8.63987952e-02 -2.84993976e-01 -1.88269603e+00
 -2.48974234e-01 -6.78804005e-03  3.95818233e-01 -4.80785065e-11
  9.89914775e-01  9.39032645e-04 -4.80785065e-11 -4.80785065e-11
  3.75613081e-03 -1.29433036e-01 -1.32393137e-01 -3.12821195e-03], low: [-3.14 -3.14 -3.14 -3.14 -3.14 -3.14 -3.14  0.   -3.14 -3.14 -3.14 -3.14
 -3.14 -3.14 -3.14  0.  ], high: [3.14 3.14 3.14 3.14 3.14 3.14 3.14 1.   3.14 3.14 3.14 3.14 3.14 3.14
 3.14 1.  ] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,736 - 🦾 ROBOT | INFO: clip action: [-1.20832801e-01  8.63987952e-02 -2.84993976e-01 -1.88269603e+00
 -2.48974234e-01 -6.78804005e-03  3.95818233e-01  0.00000000e+00
  9.89914775e-01  9.39032645e-04 -4.80785065e-11 -4.80785065e-11
  3.75613081e-03 -1.29433036e-01 -1.32393137e-01  0.00000000e+00], check time: 0.000s | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,813 - 🦾 ROBOT | INFO: rate.sleep time: 0.077s | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,815 - 🦾 ROBOT | INFO: STATE: contained ['joint_q', 'gripper'], concated value: [-0.10185698  0.10908548 -0.3917141  -1.87037377 -0.11376995  0.07052508
  0.37194688  0.          0.41473674 -0.09909844 -0.0595028  -1.31168522
 -0.16965111 -0.04546932  0.19599037  0.        ] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,815 - 🦾 ROBOT | INFO: get obs time: 0.001s | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,815 - 🧠 MODEL | INFO: Step 3 time: 0.101s | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 


Episode 1:   1%|          | 2/200 [00:00<02:34,  1.28step/s, Reward=0.000, Status=Running, Total Reward=0.000][A

Episode 1:   2%|▏         | 3/200 [00:00<00:54,  3.65step/s, Reward=0.000, Status=Running, Total Reward=0.000][A2026-09-25 11:16:23,855 - 🧠 MODEL | INFO: action infer time: 0.038s | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,855 - 🦾 ROBOT | INFO: action: [-1.19298562e-01  8.66081715e-02 -2.85422236e-01 -1.88174915e+00
 -2.49817565e-01 -5.31369727e-03  3.96444142e-01  1.87806529e-03
  9.89914775e-01 -4.80785065e-11 -4.80785065e-11  1.87806529e-03
 -4.80785065e-11 -1.29433036e-01 -1.32616416e-01 -3.66420369e-03] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,855 - 🦾 ROBOT | WARNING: action out of range, action: [-1.19298562e-01  8.66081715e-02 -2.85422236e-01 -1.88174915e+00
 -2.49817565e-01 -5.31369727e-03  3.96444142e-01  1.87806529e-03
  9.89914775e-01 -4.80785065e-11 -4.80785065e-11  1.87806529e-03
 -4.80785065e-11 -1.29433036e-01 -1.32616416e-01 -3.66420369e-03], low: [-3.14 -3.14 -3.14 -3.14 -3.14 -3.14 -3.14  0.   -3.14 -3.14 -3.14 -3.14
 -3.14 -3.14 -3.14  0.  ], high: [3.14 3.14 3.14 3.14 3.14 3.14 3.14 1.   3.14 3.14 3.14 3.14 3.14 3.14
 3.14 1.  ] | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:23,855 - 🦾 ROBOT | INFO: clip action: [-1.19298562e-01  8.66081715e-02 -2.85422236e-01 -1.88174915e+00
 -2.49817565e-01 -5.31369727e-03  3.96444142e-01  1.87806529e-03
  9.89914775e-01 -4.80785065e-11 -4.80785065e-11  1.87806529e-03
 -4.80785065e-11 -1.29433036e-01 -1.32616416e-01  0.00000000e+00], check time: 0.000s | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:26,767 - 🦾 ROBOT | INFO: 🔔 收到信号: 12 | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:26,768 - 🦾 ROBOT | INFO: �� 执行停止 | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:26,768 - 🦾 ROBOT | INFO: ⏹️ 机械臂运动已停止 | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540 
2026-09-25 11:16:26,769 - 🦾 ROBOT | INFO: 🔔 信号处理完成，当前状态 - 暂停: False, 停止: True | /home/kuavo/jiayi/kuavo_learning_studio/kuavo_deploy/src/scripts/script.py:540
```

## 6. 开环测试补充说明（Open-loop Evaluation）

该图对 16 个 action 维度分别绘制了：

- `gt action`（数据集真值动作）
- `pred action`（模型预测动作）
- `inference point`（红点，表示每次重新推理的时刻）

在 `both + qiangnao` 条件下，16 个维度顺序为：

```text
0~6   左臂 7 个关节
7     左手 gripper
8~14  右臂 7 个关节
15    右手 gripper
```

### 6.1 与真机日志中的“重大偏差关节”对应关系

前文基于真机在线日志标出的主要“重大偏差”关节为：

- `index 8`：右臂 J1 / shoulder pitch
- `index 11`：右臂 J4 / forearm pitch
- `index 14`：右臂 J7 / wrist roll

从用户提供的开环测试图看，这几个维度在开环评估中均**没有出现明显异常发散**，而是整体呈现出：

- `pred action` 与 `gt action` 基本重合或高度贴合；
- 推理刷新点（红点）处没有明显的跨 chunk 跳变；
- 轨迹趋势连续、平滑，没有出现与真机日志中同等级别的突变偏差。

### 6.2 关节级观察

#### index 8（右臂 J1）

开环图中该维度整体接近常值，预测曲线与真值曲线基本重合，表现稳定。与真机在线日志中第 1 步约 `+0.999 rad` 的大偏差相比，开环结果说明该关节**模型预测本身并不异常**。

#### index 11（右臂 J4）

开环图中该维度同样呈稳定重合状态，没有出现真机日志中第 1 步约 `+1.862 rad` 的大幅错位。说明该维度在离线开环评测中**模型能够正常跟踪数据分布**。

#### index 14（右臂 J7）

开环图中该维度随时间变化较明显，但 `pred action` 与 `gt action` 贴合良好，整体趋势一致，没有出现明显失配。对应到真机日志中第 1 步约 `-0.523 rad` 的偏差，可以认为该关节在开环预测里也是**正常的**。

