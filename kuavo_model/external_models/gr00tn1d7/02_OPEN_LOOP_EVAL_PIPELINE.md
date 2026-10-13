# Kuavo Open-Loop Evaluation Pipeline

本文档说明如何使用 `gr00t/eval/open_loop_eval.py`，在离线 LeRobot 数据集上比较 GR00T N1.7 的预测动作与真实动作（Ground Truth）。Open-loop evaluation 不会驱动机器人，只用于快速检查 checkpoint 是否收敛、动作维度是否正确，以及不同 checkpoint 之间的相对表现。

## 1. Pipeline 概览

```text
LeRobot episode
  -> 按 checkpoint 中的 modality config 读取图像、状态和语言
  -> 每隔 action_horizon 步构造一次 observation
  -> GR00T policy 预测一个 action chunk
  -> 拼接预测动作并与数据集 GT action 对齐
  -> 计算未归一化的 MSE / MAE
  -> 保存 state / GT action / predicted action 曲线图
```

评估指标在原始动作空间中计算：

- `MSE`：对大误差更敏感，适合发现发散或个别动作维度异常。
- `MAE`：更直观地反映平均绝对动作误差。
- 曲线图：用于判断时间延迟、系统性偏差、动作抖动或部分关节失效。

MSE/MAE 只能衡量模仿数据的拟合程度，不能替代仿真或真机 closed-loop success rate。

## 2. 前置条件

从本目录执行以下命令：

```bash
cd /root/bayes-tmp/jiayi/kuavo_learning_studio/kuavo_model/external_models/gr00tn1d7
uv sync --all-extras
```

准备以下内容：

1. GR00T LeRobot 格式的数据集，例如：
   `/root/bayes-tmp/kuavo_dataset/package_pick_and_place_demo_lerobot_jiayi/lerobot`
2. 待评估 checkpoint，例如：
   `/root/bayes-tmp/jiayi/kuavo_learning_studio/outputs/grootn17-package-pick-and-place-ac32/checkpoint-50000`
3. checkpoint 内应包含模型权重及推理所需的 `processor_config.json`、`statistics.json`、`embodiment_id.json` 等配置文件。

当前仓库的 `kuavo_config.py` 使用：

- embodiment：`NEW_EMBODIMENT`
- cameras：`head`、`wrist_left`、`wrist_right`
- state/action：`left_arm`、`left_gripper`、`right_arm`、`right_gripper`
- action chunk：`delta_indices=list(range(0, 16))`，即 16 步

数据集字段、checkpoint 中保存的 modality config 和训练时配置必须一致。

## 3. 推荐方式：直接加载本地 checkpoint

先用单条 episode 做 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path /root/bayes-tmp/kuavo_dataset/package_pick_and_place_demo_lerobot_jiayi/lerobot \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /root/bayes-tmp/jiayi/kuavo_learning_studio/outputs/grootn17-package-pick-and-place-ac32/checkpoint-50000 \
  --traj-ids 0 \
  --steps 200 \
  --action-horizon 16 \
  --save-plot-path /tmp/open_loop_eval/checkpoint-50000_traj-0.jpeg
```

Smoke test 正常后再评估多条 episode：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path /root/bayes-tmp/kuavo_dataset/package_pick_and_place_demo_lerobot_jiayi/lerobot \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /root/bayes-tmp/jiayi/kuavo_learning_studio/outputs/grootn17-package-pick-and-place-ac32/checkpoint-50000 \
  --traj-ids 0 1 2 3 4 \
  --steps 400 \
  --action-horizon 16
```

不指定 `--save-plot-path` 时，每条轨迹分别保存到：

```text
/tmp/open_loop_eval/traj_0.jpeg
/tmp/open_loop_eval/traj_1.jpeg
...
```

注意：`--save-plot-path` 是完整的图片文件路径。多条 trajectory 共用一个显式路径时，后生成的图片会覆盖前一张。因此，多轨迹评估建议省略该参数，或者逐条运行并为每条轨迹指定不同文件名。

## 4. Server-Client 方式

模型常驻 GPU 服务端时，可分两个终端运行。

Terminal 1，启动 policy server：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/eval/run_gr00t_server.py \
  --model-path /root/bayes-tmp/jiayi/kuavo_learning_studio/outputs/grootn17-package-pick-and-place-ac32/checkpoint-50000 \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5555
```

Terminal 2，运行 open-loop client；关键点是不要传 `--model-path`：

```bash
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path /root/bayes-tmp/kuavo_dataset/package_pick_and_place_demo_lerobot_jiayi/lerobot \
  --embodiment-tag NEW_EMBODIMENT \
  --host 127.0.0.1 \
  --port 5555 \
  --traj-ids 0 1 2 \
  --steps 400 \
  --action-horizon 16
```

跨机器运行时，将 client 的 `--host` 改为服务端 IP，并确保端口可访问。

## 5. 主要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset-path` | `demo_data/cube_to_bowl_5/` | LeRobot 数据集根目录 |
| `--embodiment-tag` | `new_embodiment` | Kuavo 使用 `NEW_EMBODIMENT`，大小写不敏感 |
| `--model-path` | `None` | 本地 checkpoint 或 Hugging Face 模型 ID；省略时使用 server-client 模式 |
| `--traj-ids` | `0` | episode ID 列表，以空格分隔 |
| `--steps` | `200` | 每条 episode 最多评估的时间步；自动截断到 episode 实际长度 |
| `--action-horizon` | `16` | 每次推理消费的预测动作数，应与模型 action chunk 能力对齐 |
| `--host` / `--port` | `127.0.0.1` / `5555` | client 连接的 policy server 地址 |
| `--save-plot-path` | `None` | 单张输出图片的完整文件路径 |
| `--modality-keys` | `None` | 仅评估指定 action modality，例如 `left_arm left_gripper` |
| `--denoising-steps` | `4` | 当前 `open_loop_eval.py` 尚未将该值传给 `Gr00tPolicy`，修改此参数目前不会影响推理 |

如只检查双臂、不检查夹爪，可使用：

```bash
--modality-keys left_arm right_arm
```

## 6. 结果检查

日志中应重点检查：

```text
Current modality config: ...
Dataset length: ...
Unnormalized Action MSE across single traj: ...
Unnormalized Action MAE across single traj: ...
Average MSE across all trajs: ...
Average MAE across all trajs: ...
```

建议固定同一组 `traj_ids`、`steps` 和 `action_horizon` 比较多个 checkpoint。除平均 MSE/MAE 外，还应逐维查看曲线图：

- GT 与 prediction 是否大致同相位；
- 左右臂与夹爪是否都产生有效输出；
- 是否存在固定偏置、饱和、尖峰或明显延迟；
- 后期 checkpoint 是否相对早期 checkpoint 稳定改善。

如果数据集包含训练集与验证集，优先在未参与训练的验证 episodes 上报告最终结果，避免把训练集拟合误当作泛化能力。

## 7. 常见问题

### GT 与 prediction shape 不一致

通常由以下原因导致：

- checkpoint modality config 与数据集 action 字段不一致；
- `--modality-keys` 名称或顺序错误；
- `--action-horizon` 超过模型实际返回的 action chunk 长度。

先确认 `kuavo_config.py`、checkpoint 的 processor config 和数据集 schema 使用相同的 action keys。当前配置应优先使用 `--action-horizon 16`。

### 找不到图像或 action/state 字段

检查 LeRobot 数据集中是否存在以下逻辑字段：

```text
video.head
video.wrist_left
video.wrist_right
state.left_arm
state.left_gripper
state.right_arm
state.right_gripper
action.left_arm
action.left_gripper
action.right_arm
action.right_gripper
annotation.human.task_description
```

实际存储映射以数据集 metadata 和 checkpoint modality config 为准。

### CUDA OOM

- 确认只在目标 GPU 上运行：`CUDA_VISIBLE_DEVICES=0`；
- 关闭同一 GPU 上不需要的训练或推理进程；
- 先减少 `--steps` 和 trajectory 数量做 smoke test；
- 必要时改用 server-client 模式，让多个评估任务复用同一个模型进程。

### 端口被占用

服务端和客户端同时换用其他端口，例如 `--port 5556`。

### 指定多个 traj 后只剩一张图

显式传入同一个 `--save-plot-path` 会反复覆盖文件。省略该参数即可按 `traj_{id}.jpeg` 分别保存。

## 8. 推荐验收顺序

1. 用 `traj_ids=0`、`steps=32` 验证数据和 checkpoint 能被正确加载。
2. 用单条完整 episode 检查输出 shape、MSE/MAE 和动作曲线。
3. 固定一组验证 episodes，对 `checkpoint-5000` 至 `checkpoint-50000` 做横向比较。
4. 选出 open-loop 指标和曲线较好的 checkpoint。
5. 进入仿真或真机 closed-loop 评估，最终以任务成功率、安全性和动作稳定性为准。
