# GR00T 部署问题与修改记录

## 1. Execution Step 设置

### 可能存在的问题

当前 GR00T 模型预测的 action chunk 长度为 16：

```python
delta_indices = list(range(16))
```

Learning Studio 中 `execution_horizon` 默认也设置为 16。这意味着模型观察一次环境并预测 16 步动作后，会连续执行完整的 16 步动作，期间不会根据最新图像和机器人状态重新推理。

当控制频率为 10 Hz 时，完整执行 16 步相当于约 1.6 秒的开环控制。在抓取、接近目标等需要频繁视觉修正的阶段，可能出现横向偏移、越过目标或无法及时纠正动作的问题。

需要区分：

- `action_horizon`：模型一次预测的 action chunk 长度，本项目为 16。
- `execution_horizon`：从预测 chunk 中实际执行多少步后重新观察和规划。

官方 Isaac GR00T N1.7 仿真 rollout 默认使用 `n_action_steps = 8`。官方代码也给出了预测 16 步、执行 8 步的 receding-horizon 示例。因此，保留 16 步预测 chunk、将实际执行步数改为 8，更接近官方部署方式。

### 建议修改

修改：

```text
configs/server/launcher.yaml
```

将：

```yaml
groot:
  args:
    execution_horizon: 16
```

改为：

```yaml
groot:
  args:
    execution_horizon: 8
```

也可以在启动时临时覆盖：

```bash
python kuavo_server/launch.py groot \
  --checkpoint /path/to/checkpoint \
  --execution_horizon 8 \
  --strict
```

### 对照实验

建议分别测试：

| execution_horizon | 10 Hz 下的重规划周期 | 用途 |
|---:|---:|---|
| 4 | 0.4 秒 | 更频繁地根据视觉反馈修正 |
| 8 | 0.8 秒 | 官方 N1.7 仿真默认，建议优先测试 |
| 16 | 1.6 秒 | 完整 action chunk 开环执行 |

测试时保持 checkpoint、任务提示词、相机输入、初始状态和随机种子一致，只修改 `execution_horizon`，对比目标接近轨迹和任务成功率。

### 判断标准

- 如果 8 步或 4 步明显改善目标接近和抓取轨迹，说明完整执行 16 步导致反馈修正不及时。
- 如果 4、8、16 步的行为基本一致，说明 `execution_horizon` 不是主要原因，应继续检查模型离线预测、数据与部署状态顺序、相机输入和归一化统计。

### 修改状态

- [ ] 修改 `execution_horizon`
- [ ] 完成 4/8/16 步对照实验
- [ ] 记录轨迹、成功率和结论

---

## 2. Simulator Head Init 设置

### 是否影响部署

有影响。

在 simulator 环境每次执行 `env.reset()` 时，部署代码都会调用：

```python
self.robot.control_head(self.head_init[0], self.head_init[1])
```

因此，`head_init` 会直接改变机器人头部姿态和头部相机视角。GR00T 使用 `head`、`wrist_left`、`wrist_right` 三路图像进行推理，如果头部相机视角与训练数据不一致，就会产生视觉域偏移，可能影响目标定位、接近方向和抓取动作。

### 可能存在的问题

原始 `kuavo_data_challenge` simulator 配置和 Learning Studio 部署文档使用：

```yaml
head_init: [0, 0.209]
```

其中 `0.209 rad` 约等于 `12°`。

但 Learning Studio 当前在加载 sim 配置时强制设置：

```python
env_cfg["head_init"] = [0, 12]
```

随后将该值直接传给 `control_head()`。如果 SDK 接口使用弧度，那么这里传入的是 `12 rad`，而不是预期的 `0.209 rad`，头部可能被转到错误位置或被关节限位裁剪，导致部署图像与训练图像明显不一致。

此外，该值是在配置加载代码中直接赋值的，因此即使在 `configs/deploy/deploy.yaml` 中手动填写 `head_init`，也会被覆盖。

### 建议修改

修改：

```text
kuavo_deploy/config.py
```

将：

```python
env_cfg["head_init"] = [0, 12]
```

改为：

```python
env_cfg.setdefault("head_init", [0, 0.209])
```

这样默认值与旧 simulator 部署保持一致，同时允许用户在 `deploy.yaml` 中显式覆盖。

如果希望暂时保持强制默认值，也至少应改为：

```python
env_cfg["head_init"] = [0, 0.209]
```

### 验证方法

1. 分别使用 `[0, 12]` 和 `[0, 0.209]` 重置 simulator。
2. 保存 `observation.images.head_cam_h` 的第一帧。
3. 与训练数据中相同初始阶段的 `head` 图像对比相机俯仰角和目标位置。
4. 在 SDK 文档或运行日志中确认 `control_head()` 的单位。
5. 保持 checkpoint、seed、prompt 和 execution horizon 不变，对比任务轨迹及成功率。

### 判断标准

- 如果 `[0, 0.209]` 的头部图像明显更接近训练数据，并改善目标接近轨迹，则错误的 head init 是部署问题之一。
- 如果两种设置得到完全相同的实际头部姿态，需要检查 SDK 是否自动进行角度转换或限位裁剪。
- 即使 head init 修复后模型仍存在动作噪声，也不能排除该问题；head init 影响视觉输入，而模型离线预测噪声属于另一条问题链。

### 修改状态

- [ ] 确认 `control_head()` 参数单位
- [ ] 修正 sim 默认 `head_init`
- [ ] 保存并对比 simulator 与训练数据头部图像
- [ ] 对比修改前后的任务成功率

---

## 3. State 归一化及数据语义对齐

### 可能存在的问题

GR00T 不会直接使用机器人原始 state。Checkpoint 中的 processor 会根据训练数据保存的 `statistics.json` 对关节和夹爪状态进行归一化；模型输出的 action 随后再经过反归一化，恢复为机器人控制量。

完整过程为：

```text
Simulator 原始 state
→ 按 modality key 拆分
→ 使用 checkpoint statistics 归一化
→ GR00T 推理
→ action 反归一化
→ 拼接成 Kuavo 16 维 action
→ env.step()
```

当前 Kuavo 双臂配置预期的 state/action 顺序为：

```text
left_arm[0:7]
left_gripper[7]
right_arm[8:15]
right_gripper[15]
```

| 数据 | 维度 | 预期单位/范围 |
|---|---:|---|
| `left_arm` | 7 | 关节角，rad |
| `left_gripper` | 1 | `[0, 1]` |
| `right_arm` | 7 | 关节角，rad |
| `right_gripper` | 1 | `[0, 1]` |

可能存在以下不一致：

- 训练数据使用弧度，而部署 state 使用角度。
- 训练与部署的左右臂或夹爪拼接顺序不同。
- 训练夹爪范围为 `[0, 1]`，部署输入为 `[0, 100]`。
- 部署 checkpoint 使用了其他数据集生成的 `statistics.json`。
- Checkpoint 根目录和 `processor/` 目录中的 processor/statistics 文件不匹配。
- Simulator 的关节零位定义与训练数据不同。
- Adapter 在维度不一致时自动截断或补零，使错误没有立即暴露。

这类问题通常不会直接报错，但模型会误判当前机器人姿态。对于绝对关节位置 action，可能表现为持续纠偏、左右摆动、动作幅度异常或无法朝目标接近。

### 建议修改

首先在部署链路中增加严格校验，不允许 state 被静默截断或补零：

```python
state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
if state.shape != (16,):
    raise ValueError(f"Expected Kuavo state shape (16,), got {state.shape}")
if not np.isfinite(state).all():
    raise ValueError("Kuavo state contains NaN or Inf")
```

启动 GR00T server 时开启严格模式：

```bash
python kuavo_server/launch.py groot \
  --checkpoint /path/to/checkpoint \
  --strict
```

同时记录以下数据：

1. 训练数据中某个 episode 的原始 `observation.state` 和 `action`。
2. Simulator 部署前 10 步发送给 adapter 的原始 16 维 state。
3. Adapter 按 modality 拆分后的四组 state。
4. GR00T processor 反归一化后的四组 action。
5. 最终传入 `env.step()` 的 16 维 action。

检查 checkpoint 中下列文件是否来自同一次训练：

```text
checkpoint-xxxxx/
├── config.json
├── statistics.json
├── processor_config.json
└── processor/
```

不要使用其他 run 或其他数据集的 `statistics.json` 覆盖当前 checkpoint。

### 验证方法

部署时打印前 10 步 state：

```python
state = np.asarray(obs["observation.state"]).reshape(-1)
print(
    "state_shape=", state.shape,
    "left_arm=", state[:7],
    "left_gripper=", state[7],
    "right_arm=", state[8:15],
    "right_gripper=", state[15],
)
```

重点检查：

- 手臂关节是否大致位于 `[-3.14, 3.14]`。
- 夹爪是否位于 `[0, 1]`。
- 左右臂运动时，对应 state 区段是否正确变化。
- 静止时是否存在明显跳变、NaN 或 Inf。
- 训练数据与 simulator state 是否具有相同的维度、顺序、单位、夹爪定义和零位定义。

还应使用训练 episode 运行 open-loop eval：

```bash
uv run python gr00t/eval/open_loop_eval.py \
  --model-path /path/to/checkpoint \
  --dataset-path /path/to/lerobot \
  --embodiment-tag NEW_EMBODIMENT \
  --traj-ids 0 \
  --action-horizon 16
```

- 如果训练 episode 上预测曲线也无法跟随 ground truth，应优先检查训练、modality 和 statistics。
- 如果训练 episode 上预测正常，但 simulator 中异常，应优先检查部署 state、图像输入和机器人控制接口。

### 判断标准

- 手臂 state 出现几十或上百度数值：可能将角度误当成弧度。
- 夹爪 state 出现 `50` 或 `100`：未转换到 `[0, 1]`。
- 活动左臂时右臂区段变化：state 拼接顺序错误。
- Checkpoint statistics 与当前训练数据范围明显不符：processor/checkpoint 可能不匹配。
- 所有数据语义一致且 open-loop eval 正常：可以基本排除 state 归一化是主要原因。

### 修改状态

- [ ] 校验部署 state 必须为 16 维且全部有限
- [ ] 确认关节角单位为 rad
- [ ] 确认夹爪范围为 `[0, 1]`
- [ ] 确认左右臂和夹爪顺序
- [ ] 确认 checkpoint processor/statistics 来自同一次训练
- [ ] 对比训练帧和 simulator 首帧 state
- [ ] 完成训练 episode 的 open-loop eval

---

## 4. 

### 可能存在的问题



### 建议修改



### 验证结果
---

## 5. 复用 Data Challenge 部署运行 GR00T N1.7

### 结论

可以复用 `kuavo_data_challenge` 中已经验证过的仿真环境、ROS 接口和机器人控制逻辑，但不建议把整个 `kuavo_deploy` 目录软链接到 Learning Studio。

推荐架构：

```text
Data Challenge inference 仿真与机器人控制
                    |
                    | ZeroMQ，localhost:5555
                    v
Learning Studio kuavo_server
                    |
                    v
Isaac GR00T N1.7 checkpoint
```

在这种架构中：

- Data Challenge 负责环境观测、相机数据、机器人状态和动作执行。
- Learning Studio 的 `kuavo_server` 负责加载 GR00T N1.7 并进行 GPU 推理。
- 两者通过 ZeroMQ 传递 observation 和 action。
- GR00T server 可以运行在独立的 Python/uv 环境中，不需要和 ROS 仿真环境使用同一套依赖。

### “server 和机器人端不在同一台机器”的含义

该说法主要适用于真机或分布式部署。例如，机器人控制运行在机器人计算机上，而 GR00T 推理运行在另一台 GPU 工作站上，此时 client 必须连接 GPU 工作站的 IP，并放通 TCP 5555。

当前使用 inference 仿真平台时，如果仿真、Data Challenge deploy 和 GR00T server 都在同一台主机或同一网络命名空间中，则直接使用：

```python
host = "localhost"
port = 5555
```

即可，不需要额外配置服务器 IP 或防火墙。

如果仿真和 server 分别运行在不同 Docker 容器中，`localhost` 只指向容器自身，此时应使用容器名、宿主机地址或 host network，而不能继续使用 `localhost`。

### 为什么不建议使用整个目录的 soft link

软链接只能替换代码路径，不能自动解决以下差异：

- Python/uv/conda 环境和依赖版本不同。
- 配置文件路径与工作目录不同。
- Data Challenge 与 Learning Studio 的 `kuavo_deploy` 已经发生代码分叉。
- 两个项目都使用名为 `kuavo_deploy` 的 Python package，容易受到 `PYTHONPATH` 顺序影响。
- Learning Studio 的 client 已增加 task prompt、server reset、action chunk 和错误处理，而原版 Data Challenge client 没有完整支持这些能力。

因此，整个目录软链接容易造成“实际导入了哪个版本”不明确，后续调试也比较困难。

### Data Challenge 原版代码目前的兼容缺口

虽然其基础 ZeroMQ 序列化协议与 Learning Studio server 基本一致，但不能原封不动地用于 GR00T：

1. `kuavo_deploy/config.py` 的配置校验只允许 `diffusion` 和 `act`，会拒绝 `policy_type: client`。
2. 原版 `PolicyClient` 缺少与部署流程兼容的 `reset()`、`eval()` 和 `to()` 行为。
3. 原版 client 不会向 GR00T server 添加任务提示词 `task_prompt`。
4. 原版 client 只支持 `select_action`，没有完整的 `select_action_chunk` 与 server reset 支持。
5. client 模式不应继续执行本地模型的 processor 加载逻辑；GR00T 的输入映射应由 server adapter 完成。

### 推荐修改方式

保留 Data Challenge 的部署主体，仅移植 Learning Studio 中很薄的远程推理适配层：

- `kuavo_deploy/kuavo_service/client.py` 中新版 `PolicyClient`。
- `real_single_test.py` 或 `real_async_test.py` 中 `policy_type == "client"` 的分支。
- 配置校验对 `client` 的支持。
- 配置项 `task_prompt`。

Data Challenge 侧的部署配置应包含类似内容：

```yaml
inference:
  policy_type: client
  task_prompt: "put the object into the target"
```

Learning Studio 侧配置：

```yaml
common:
  host: "*"
  port: 5555

adapters:
  groot:
    adapter: isaac_gr00t_n17
    args:
      checkpoint: /path/to/gr00t/checkpoint
      embodiment_tag: NEW_EMBODIMENT
      which_arm: both
      execution_horizon: 8
      device: cuda
```

启动 GR00T server：

```bash
cd /path/to/kuavo_learning_studio
python kuavo_server/launch.py groot
```

然后在 Data Challenge 环境中启动 inference deploy。这里所谓“挂上 server”，就是 deploy 以 client 模式把 observation 发到 `localhost:5555`，并将返回的 action 交给原有仿真控制代码执行。

### 验证步骤

1. 启动 server，确认终端显示监听 `tcp://*:5555`。
2. 在 deploy 环境调用 `ping`，确认 server 可达。
3. 发送单帧 observation，确认 server 没有报告缺少图像、state 或 prompt。
4. 检查返回 action 的维度、左右臂顺序和夹爪范围。
5. 先使用单步或较短 `execution_horizon` 低速测试，再运行完整任务。

### 最终建议

采用“Data Challenge 仿真部署 + Learning Studio GR00T server”的进程级组合，而不是目录级 soft link。这既能复用已经验证过的 Data Challenge 部署，也能让 GR00T N1.7 保持在自己的依赖环境和 server 进程中运行。
