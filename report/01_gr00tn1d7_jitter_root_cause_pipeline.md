# GR00T N1.7 真机 Jitter 根因检测 Pipeline

## 1. 目标与结论

本文用于定位 Kuavo 真机闭环运行 GR00T N1.7 时出现的可见关节或末端抖动（jitter），暂不直接给出平滑或 RTC 的实现方案。

NVIDIA 的部署文档给出了明确的检测思路：保存全部预测 `Action Chunk`，必要时用正运动学（FK）将关节轨迹转换成 TCP 轨迹并做 3D 可视化，再分别检查 chunk 内部、相邻 chunk 边界以及机器人实际执行链路。文档还给出三个定量指标：chunk 内平均加速度幅值、chunk 边界位置跳变、chunk 边界速度方向余弦相似度。

最终应把问题归入以下至少一类：

| 类别 | 直接证据 | 优先检查方向 |
| --- | --- | --- |
| A：chunk 内部抖动 | 原始预测 chunk 自身二阶差分异常，FK 后 TCP 轨迹也不平滑 | 数据质量、训练不足、动作表示、归一化/反归一化 |
| B：chunk 边界抖动 | 单个 chunk 平滑，但前一 chunk 的最后执行点与新 chunk 起点不连续 | `execution_horizon`、观测时刻、相对动作、chunk 拼接、RTC |
| C：控制/硬件抖动 | 预测与下发命令平滑，但反馈关节/TCP 抖动 | 控制频率、插值、限幅、通信、驱动器、机械结构 |
| D：时序/供给不足 | action 队列耗尽、重复 hold、控制周期明显超期 | 推理延迟、网络延迟、异步队列、线程调度；这更接近 stop-and-go，但可能表现为抖动 |

## 2. 本仓库中的观测点

当前 GR00T N1.7 服务端入口为 `kuavo_server/adapters/isaac_gr00t_n17.py`：

- `_Gr00tRuntime.infer()` 的原始输出是按 action key 分组的 `[1, T, D]` 模型输出。
- `_convert_action_chunk()` 将其转换为 Kuavo 顺序的 `[T, 16]`（双臂）或 `[T, 8]`（单臂）命令。
- `execution_horizon` 会截断每次真正返回/缓存的 chunk；诊断记录必须同时保存模型完整 horizon 和实际 execution horizon，不能把两者混为一谈。
- `select_action_chunk()` 供 `kuavo_deploy/src/eval/real_async_test.py` 使用。当前异步实现将新 chunk 直接放入 FIFO buffer，没有 RTC/重叠融合；因此 chunk 边界是重点嫌疑点。
- `real_async_test.py` 在 buffer 超时后可能 hold 上一个 action。必须记录 `buffer.qsize()`、hold 次数和每次推理耗时，以排除队列欠载。
- 同步路径 `kuavo_deploy/src/eval/real_single_test.py` 已记录推理时间和 step 时间，但尚不足以区分模型命令与机器人反馈。

## 3. 安全与实验控制

1. 首轮在低速、空载、远离奇异位形和关节限位的姿态运行，准备物理急停。
2. 固定 checkpoint、任务 prompt、初始姿态、相机位置、控制频率和随机种子；每组至少重复 3 次。
3. 先做短轨迹和单臂实验，再扩展到双臂与完整任务。
4. 日志使用单调时钟纳秒时间戳。图像采集、观测完成、请求发送、推理开始/结束、chunk 收到、命令下发和反馈收到均要独立打点。
5. 不以单次肉眼观察作为结论；阈值必须由无 jitter 的基线轨迹或训练数据分布确定。

## 4. 最小日志格式

每个 episode 建议保存一个目录：

```text
jitter_run/<run_id>/
├── metadata.yaml
├── chunks.npz
├── control_trace.parquet
├── events.jsonl
├── video_head.mp4
└── video_wrist.mp4
```

`metadata.yaml` 至少包含：git commit/branch、checkpoint、embodiment、`which_arm`、模型 action horizon、`execution_horizon`、同步/异步模式、控制频率、buffer 配置、action 表示（绝对/相对）、归一化配置和机器人控制模式。

`chunks.npz` 至少包含：

- `raw_model_chunks[N,T,D_model]`：反归一化前后最好各保存一份；
- `kuavo_chunks[N,T,D_cmd]`：`_convert_action_chunk()` 后、截断前；
- `executed_chunk_mask[N,T]`：每个预测点是否真正执行；
- `chunk_obs_state[N,D_state]` 与 `chunk_timestamps[N]`；
- 若支持 FK，保存左右臂 `predicted_tcp[N,T,6或7]`。

`control_trace.parquet` 每个控制 tick 一行，至少记录：

- `t_command`、`t_feedback`、`chunk_id`、`step_in_chunk`、`buffer_size`、`held_last_action`；
- `command_joint`、`feedback_joint`、`feedback_joint_velocity`；
- `command_tcp`、`feedback_tcp`（可离线 FK 生成）；
- 图像/状态时间戳、推理往返耗时、实际控制周期 `dt`。

## 5. 检测流程

### Stage 0：建立可复现基线

分别采集以下三组，保持相同控制频率与初始姿态：

1. **静态保持**：持续发送固定关节目标 10–20 秒；若仍抖动，优先进入类别 C。
2. **确定性平滑轨迹**：绕过模型，下发限速的正弦或样条轨迹；若命令平滑而反馈抖动，优先进入类别 C。
3. **模型闭环轨迹**：同步和异步各运行一组；用于判断 jitter 是否与 chunk 边界、队列欠载或推理时延相关。

### Stage 1：验证数据与坐标语义

逐项核对：

- 模型 action 是绝对关节位置还是相对当前 state 的增量；部署端是否做了完全一致的反归一化和积分。
- 训练与部署的关节顺序、左右臂顺序、角度单位、夹爪单位是否一致。本 adapter 的 Kuavo 双臂顺序为：左臂 7、左夹爪 1、右臂 7、右夹爪 1。
- 观测 state 与相机帧是否来自接近同一时刻；统计 `max(image_ts)-min(image_ts)` 以及 `state_ts-image_ts`。
- 是否在接近关节限位、奇异位形或控制器限速/限加速度饱和区运行。

任何单位、顺序或绝对/相对动作错误都应先修正，再计算后续指标。

### Stage 2：检查 chunk 内部平滑性

在固定采样周期 `dt` 下，对每个关节分别计算：

```python
velocity = np.diff(chunks, axis=1) / dt
acceleration = np.diff(velocity, axis=1) / dt
intra_accel = np.linalg.norm(acceleration, axis=-1)
```

报告 mean、P95、max，并同时输出每个关节的 P95。NVIDIA 原文使用未除以 `dt` 的二阶位置差分；除以 `dt²` 后才是物理加速度，因此跨控制频率比较时必须使用带 `dt` 的版本。

将关节 chunk 通过与真机一致的 FK 转为左右臂 TCP，分别绘制：

- 3D 位置轨迹，并按 `chunk_id` 着色；
- `x/y/z` 与姿态随时间曲线；
- 速度、加速度或 jerk 随时间曲线。

若原始预测在单个 chunk 内已明显锯齿化，判为类别 A 候选。进一步对比训练数据相同指标的分布；模型 P95 显著高于训练数据 P95 才能较有把握地指向欠训练或输出分布异常。

### Stage 3：检查相邻 chunk 边界

边界必须比较“前一 chunk 最后一个**实际执行**点”和“后一 chunk 第一个实际执行点”，其中 `E=execution_horizon`：

```python
q_prev = chunks[:-1, E - 1, :]
q_next = chunks[1:, 0, :]
boundary_jump = np.linalg.norm(q_next - q_prev, axis=-1)

v_prev = chunks[:-1, E - 1, :] - chunks[:-1, E - 2, :]
v_next = chunks[1:, 1, :] - chunks[1:, 0, :]
momentum_cos = np.sum(v_prev * v_next, axis=-1) / (
    np.linalg.norm(v_prev, axis=-1) * np.linalg.norm(v_next, axis=-1) + 1e-8
)
```

同时在 TCP 空间计算位置跳变、旋转角跳变和速度方向变化。报告 mean、P95、max，并在视频和时间曲线上标出 top-k 异常边界。

若 chunk 内指标正常，但 jitter 视频时间点与 boundary jump 峰值一致，判为类别 B。注意：NVIDIA 示例假设所有 chunk 等长；真实日志若长度不同，应保存 list/offset，而不是强行堆叠成规则数组。

### Stage 4：比较命令与机器人反馈

对 command 与 feedback 先按时间戳对齐，并估计最佳跟踪延迟，再计算：

- 跟踪误差 `q_feedback(t+lag)-q_command(t)` 的 RMSE、P95 和频谱；
- command 与 feedback 的速度、加速度和 jerk；
- 控制 tick 的 `dt` mean/P95/max、deadline miss 比例；
- 反馈中存在而命令中不存在的高频峰值；
- 限速、限加速度、限位裁剪或控制器饱和事件。

若 command 平滑但 feedback 出现周期性高频振荡，判为类别 C 候选。然后分别检查插值方式、位置环增益、阻尼、驱动状态、编码器噪声、通信丢包和机械松动。

### Stage 5：排除时序和异步队列问题

对每次推理和控制 tick 计算：

- observation capture、预处理、ZMQ 往返、模型推理、后处理各阶段延迟；
- action buffer 最小深度和空队列次数；
- `held_last_action=True` 的次数及其后一次新 action 的跳变；
- 控制频率的 P50/P95/P99 和 deadline miss 比例。

若 jitter 只在 buffer 为空、hold 后恢复或控制周期超期时出现，判为类别 D；此时不要误判为模型 chunk 内抖动。当前异步代码对 chunk 采用直接 FIFO 拼接，诊断时还需区分“队列未空但 chunk 边界不连续”和“队列耗尽后恢复”两种事件。

### Stage 6：A/B 实验确认因果

一次只改变一个因素，每组重复至少 3 次：

| 实验 | 对照变量 | 能验证的假设 |
| --- | --- | --- |
| 模型输出离线回放 vs 原模型在线推理 | 相同 action、相同控制链路 | 抖动是否由观测闭环/模型随机性触发 |
| 同步 vs 异步 | checkpoint、任务和控制频率相同 | 队列与时序是否是主因 |
| `execution_horizon` 取 16/8/4 | 其余不变 | chunk 边界频率与 jitter 是否相关 |
| 模型轨迹 vs 样条轨迹 | 相同幅值/速度范围 | 模型命令与底层控制的责任边界 |
| 原始预测 vs 仅离线平滑后回放 | 其余不变 | 高频模型输出是否驱动 jitter |
| 静态/远离限位 vs 原任务姿态 | 控制参数相同 | 奇异位形、限位和硬件负载影响 |

不建议在定位阶段同时开启平滑、修改控制器增益并调整 chunk 参数，否则无法建立因果关系。

## 6. 判定规则与交付物

每次分析输出：

1. `summary.md`：运行配置、是否复现、类别 A/B/C/D 及置信度；
2. `metrics.csv`：按 episode、chunk、boundary、joint 汇总的 mean/P95/max；
3. `joint_command_feedback.png`：关节命令/反馈及 chunk 边界；
4. `tcp_trajectory_3d.html`：左右 TCP 交互式轨迹；
5. `latency_buffer.png`：推理延迟、控制周期和 buffer 深度；
6. top-k 异常事件对应的视频时间戳和原始日志索引。

推荐采用相对基线判定，不写死跨机器人通用阈值。例如：

- 指标超过无 jitter 基线 P99，标记为异常；
- boundary jump 峰值与肉眼 jitter 时间在一个控制周期内对齐，支持类别 B；
- feedback 高频能量显著高于 command，且平滑样条也能复现，支持类别 C；
- jitter 与 buffer empty/hold 恢复高度重合，支持类别 D。

## 7. 推荐实施顺序

1. 在 adapter 记录完整 raw/converted action chunks、`chunk_id` 和 `execution_horizon`。
2. 在真机 env/控制循环记录每 tick 的 command、feedback、单调时钟、buffer 和 hold 状态。
3. 先跑静态保持与样条基线，再跑同步模型闭环，最后跑异步模型闭环。
4. 编写离线分析脚本，生成三项 NVIDIA 指标、TCP FK 图、command-feedback 图和 latency/buffer 图。
5. 根据 A/B/C/D 分类做单变量 A/B 实验；确认根因后再进入 smoothing、相对动作、RTC 或底层控制器的解决阶段。

## 8. 无真机时的 Adapter Server + 数据集实验

### 8.1 能回答与不能回答的问题

使用本地 LeRobot 数据集逐帧提供 observation，并通过真实 ZMQ adapter server 获取 action chunk，可以验证：

- 模型预测 chunk 内部是否平滑（类别 A）；
- 在数据集状态推进条件下，相邻预测 chunk 是否连续（类别 B 的离线证据）；
- adapter 的 action key 映射、关节顺序、维度和 `execution_horizon` 是否正确；
- ZMQ + 模型推理的耗时分布，以及不同图像/状态输入对预测稳定性的影响；
- 第一个预测 action 与数据集示教 action 的偏差。

该实验不能验证类别 C，即电机、编码器、机器人控制器、真实通信周期或机械结构振动。它也不是真正的模型闭环：下一次 observation 来自示教数据，而不是上一次预测 action 作用于机器人后的状态。因此结论应写成“模型/adapter 侧发现或未发现抖动证据”，不能写成“已排除真机 jitter”。

### 8.2 新增诊断接口与工具

- Adapter server 新增可选端点 `diagnose_action_chunk`。
- GR00T N1.7 的该端点返回完整、未被 `execution_horizon` 截断的 Kuavo action chunk，同时返回模型原始 action-key 输出和输入 state16。
- 离线工具：`kuavo_server/tools/offline_jitter_diagnostic.py`。
- 输出：`chunks.npz`、`per_chunk.csv`、`per_boundary.csv` 和 `summary.json`。

之所以不能只调用 `select_action_chunk`，是因为它会按 server 的 `execution_horizon` 截断，从而无法判断抖动原本存在于完整模型输出中，还是由部署时截断/拼接产生。

### 8.3 基础运行方法

先启动 GR00T adapter server（按实际 checkpoint 修改）：

```bash
python kuavo_server/launch.py groot \
  --checkpoint /absolute/path/to/checkpoint \
  --execution_horizon 8 \
  --port 5555
```

在 GR00T N1.7 的 uv 环境中运行离线实验。工具复用 NVIDIA
`gr00t.data.dataset.LeRobotEpisodeLoader`，与 `gr00t/eval/open_loop_eval.py`
使用相同的数据加载路径，因此可以直接读取当前的 LeRobot v2.1 数据集，
不需要安装 Hugging Face `lerobot` 包，也不需要将数据转换为 v3.0：

```bash
uv run --project kuavo_model/external_models/gr00tn1d7 \
  python kuavo_server/tools/offline_jitter_diagnostic.py \
  --dataset-root /absolute/path/to/lerobot \
  --episode 0 \
  --max-frames 100 \
  --host localhost \
  --port 5555 \
  --prompt "Pick and Place" \
  --output-dir outputs/jitter/offline_ep0_h8
```

工具默认令 dataset `stride = execution_horizon`。例如执行 horizon 为 8 时，第 0 帧产生 chunk 0，第 8 帧产生 chunk 1，并比较 `chunk0[7]` 与 `chunk1[0]`。这近似 receding-horizon 的边界，但下一观测仍来自示教轨迹。若要显式覆盖，可传 `--stride`；此时应确保 stride 与待模拟的执行步数语义一致。

如果数据集元数据没有可靠 FPS，显式传入 `--fps 10`。如果 server metadata 无法提供 execution horizon，传入 `--execution-horizon 8`。

若当前目录已经是 `kuavo_model/external_models/gr00tn1d7`，脚本路径应改为仓库根目录下的绝对路径，或先回到仓库根目录执行上述命令。运行前必须重启 adapter server，使新增的 modality metadata 和 `diagnose_action_chunk` 端点生效。

### 8.4 实验矩阵

先固定 checkpoint、episode、prompt 和采样区间，然后依次执行：

| 实验编号 | 改变量 | 建议取值 | 目的 |
| --- | --- | --- | --- |
| E0 | 重复性 | 同一 observation 连续请求 10 次 | 检查 flow/diffusion 推理随机性是否导致输出方差 |
| E1 | episode | 训练集 3 条、验证集 3 条 | 判断问题是普遍存在还是场景相关 |
| E2 | execution horizon/stride | 4、8、16 | 判断边界频率和预测远期长度对不连续性的影响 |
| E3 | prompt | 正确 prompt、空 prompt、错误 prompt | 检查语言条件敏感性 |
| E4 | 图像消融 | 原图、冻结首帧、仅头部、仅腕部 | 定位视觉输入变化导致的 action 不稳定 |
| E5 | state 消融 | 原 state、冻结首帧、加入小幅受控噪声 | 判断 proprioception 敏感性 |
| E6 | checkpoint | 当前 checkpoint、较早 checkpoint、训练更久版本 | 判断欠训练或过拟合趋势 |
| E7 | adapter 对照 | 直接 `Gr00tPolicy` 与 ZMQ adapter | 排除 adapter 映射/序列化引入的差异 |

当前脚本直接支持 E1/E2/E3 和 checkpoint 切换；E0/E4/E5/E7 可在其保存的 observation/chunk 机制上扩展。第一轮建议先完成 E1 + E2，因为它们最直接回答 chunk 内和 chunk 边界问题。

### 8.5 第一轮执行计划

1. 选择一条肉眼平滑、训练时实际使用过的 episode，运行 horizon 4/8/16 三组。
2. 每组采集至少 50 个 chunks；保持对应 stride 等于 horizon。
3. 对同一设置运行 3 次，观察指标是否稳定。
4. 再选择两条不同任务阶段/物体位置的 episode 重复测试。
5. 汇总以下指标：
   - `intra_chunk_acceleration` mean/P95/max；
   - `boundary_position_jump` mean/P95/max；
   - `boundary_velocity_cosine` mean/P95/min（当前 summary 输出 mean/P95/max，逐边界 CSV 可检查最小值）；
   - `first_action_vs_dataset_l2`；
   - `request_latency_ms` P95/max。
6. 从 `per_boundary.csv` 选 position jump 最大和 velocity cosine 最小的 top-10 边界，读取 `chunks.npz` 检查具体关节。

不要直接比较不同 FPS 下未归一化的二阶差分。本工具的加速度已除以 `dt²`；关节角若为弧度，单位为 `rad/s²`，夹爪维度由于量纲不同应与手臂关节分开解释。

### 8.6 离线判读规则

- **chunk 内指标在多数 episode 都高**：支持类别 A；继续对比训练数据 action 的同类指标，并检查归一化/反归一化。
- **chunk 内平滑，但 boundary jump 高且 cosine 低**：支持类别 B；比较不同 horizon，并重点检查 state-relative action、观测推进与 RTC/融合方案。
- **只在少量 episode/任务阶段异常**：更像数据分布或视觉/state 输入敏感性问题，执行 E4/E5。
- **相同 observation 的重复输出差异大**：先固定随机种子并检查推理采样参数，再讨论平滑。
- **所有离线指标正常**：只能说明未发现模型/adapter 侧证据；需要真机后用 command-feedback 日志继续检查类别 C/D。

## 9. 参考资料

- NVIDIA Isaac-GR00T：`getting_started/real_world_deployment.md`，章节 “Common Issues: Jittering and Stop-and-Go”。
- 本仓库镜像文档：`kuavo_model/external_models/gr00tn1d7/getting_started/real_world_deployment.md`。
- 本仓库 GR00T N1.7 adapter：`kuavo_server/adapters/isaac_gr00t_n17.py`。
- 本仓库真机异步执行：`kuavo_deploy/src/eval/real_async_test.py`。
- 本仓库真机同步执行：`kuavo_deploy/src/eval/real_single_test.py`。
