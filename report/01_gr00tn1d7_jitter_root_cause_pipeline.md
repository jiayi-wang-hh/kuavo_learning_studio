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

### 8.7 已完成实验与结果（Bottle + Apple）

实验日期：2026-08-14。两项任务均使用 episode 0、10 FPS、模型 action horizon 16，并分别测试 execution horizon 4、8、16。所有比较均使用 adapter server 的完整未截断预测；边界按实际 execution horizon 比较。Bottle 使用 checkpoint-20000，Apple 使用 checkpoint-30000，因此跨任务只比较趋势，不把绝对指标差异解释为任务本身的因果影响。

必须将 14 维手臂与 2 维夹爪分开统计。夹爪值域约为 `[0,1]`，关节单位为 rad，二者直接放入同一个 L2/加速度会让开闭事件支配结果。

#### 手臂指标

| 任务 | Horizon | Chunk 数 | chunk 内加速度均值 | 边界跳变均值 | 边界跳变 P95 | 边界跳变最大值 | 边界速度 cosine 均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bottle | 4 | 39 | 7.002 | 0.092 | 0.147 | 0.229 | 0.211 |
| Bottle | 8 | 20 | 7.157 | 0.101 | 0.164 | 0.176 | 0.108 |
| Bottle | 16 | 10 | 7.130 | 0.127 | 0.202 | 0.216 | 0.201 |
| Apple | 4 | 34 | 4.875 | 0.062 | 0.144 | 0.206 | 0.471 |
| Apple | 8 | 17 | 4.843 | 0.079 | 0.183 | 0.230 | 0.412 |
| Apple | 16 | 9 | 4.852 | 0.116 | 0.252 | 0.291 | 0.371 |

结果解释：

- 同一任务内，h4/h8/h16 的手臂 chunk 内加速度基本不变。当前没有证据表明执行 horizon 是 chunk 内抖动的主因。
- 两个任务中，h16 的手臂边界跳变均值和 P95 都最大；缩短 horizon 能改善平均边界位置连续性。
- Apple 的 h16 最大异常位于 frame 96 → 112：右臂 J4 跳变 `-0.2305 rad`（约 `-13.2°`），右臂 J5 跳变 `-0.1379 rad`（约 `-7.9°`）。
- Bottle 的 h16 主要异常位于 frame 48 → 64、96 → 112 和 32 → 48，边界 L2 分别约为 `0.216`、`0.181`、`0.181`，多次涉及左右臂 J4。
- Bottle 的边界速度方向一致性整体比 Apple 差，但两个任务都存在低或负 cosine 边界，说明新 chunk 可能改变甚至反转运动方向。

#### 夹爪事件

原始 summary 中约 `0.99–1.38` 的边界极值以及约 `95–136` 的二阶差分峰值主要来自夹爪开闭，不是手臂 jitter：

- Bottle h8 的 frame 80 → 88：左右夹爪分别跳变约 `0.975` 和 `0.963`。
- Apple h4/h8 的 frame 80 → 88：右夹爪跳变约 `0.986/0.982`。
- h16 有时不出现夹爪边界峰值，是因为开闭转换落在单个 chunk 内部，而不是因为夹爪预测更平滑。

后续报告必须默认输出 arm-only、left/right arm 和 gripper-only 指标；夹爪应采用离散状态切换指标，不应用关节加速度阈值判定。

#### 推理时延与实时裕量

| 任务 | Horizon | 稳态推理均值 | 稳态 P95 | Chunk 可执行时间 | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| Bottle | 4 | 约 365 ms | 约 394 ms | 400 ms | 裕量不足，部分请求超期 |
| Bottle | 8 | 约 346 ms | 约 374 ms | 800 ms | 有约 450 ms 平均裕量 |
| Bottle | 16 | 约 406 ms | 约 424 ms | 1600 ms | 裕量充分，但边界跳变更大 |
| Apple | 4 | 约 364 ms | 约 404 ms | 400 ms | P95 已超过 chunk 时长 |
| Apple | 8 | 约 334 ms | 约 385 ms | 800 ms | 有约 466 ms 平均裕量 |
| Apple | 16 | 约 363 ms | 约 386 ms | 1600 ms | 裕量充分，但边界跳变更大 |

首次请求约需 `1.75–1.87 s`，必须在开始动作前完成 warm-up。h4 虽然平均边界指标最好，但在 10 Hz 下只有 400 ms 动作供给，无法可靠覆盖推理尾延迟，因此可能把 jitter 问题转化为 buffer empty 或 stop-and-go。

#### 阶段性结论

现有离线证据在两个任务上方向一致：

1. 单个预测 chunk 内的手臂平滑性对 execution horizon 不敏感；
2. 跨 chunk 边界存在显著位置跳变和速度方向不一致，支持类别 B；
3. h16 的边界幅度偏大；h4 的实时裕量不足；
4. `execution_horizon=8 + asynchronous inference` 是目前最合理的真机起点；
5. 夹爪必须从手臂轨迹平滑中拆出，使用独立状态机处理。

该结论仍是 teacher-forced 离线证据：下一 observation 来自数据集，而不是预测 action 作用后的真实状态。它不能排除类别 C/D。

### 8.8 下一步决策：先做两项确认，再实施边界修正

不建议继续无目标地遍历所有 E0–E7，也不建议立即接入完整 RTC。当前 GR00T N1.7 模型 horizon 为 16，而 NVIDIA 文档中的标准 RTC 建议 action chunk 至少 32，且当前 server-client 路径没有现成 RTC 集成。直接上 RTC 会同时改变调度和轨迹融合，难以验证因果。

按以下顺序推进：

#### Step 1：完成两个高价值确认实验

1. **E0 重复性**：Bottle 和 Apple 各选择正常帧、最大边界前帧和夹爪切换前帧，共 3 个 observation；每个 observation 重复推理 10 次。报告每个 action step/关节的预测方差。若同一输入方差很大，先固定随机种子或推理噪声，再做 chunk 融合。
2. **E7 adapter 对照**：对同一批 observation，使用 `diagnose_action_chunk` 同一 response 中的原始 `Gr00tPolicy.get_action()` action dict 重建 Kuavo 顺序，并与 adapter 转换后的完整 chunk 比较。目标是逐元素误差接近浮点容差，从而排除 action key 映射、序列化和 Kuavo 维度组合错误。

这两项不需要真机，且会直接决定修正应落在模型采样、adapter 还是 chunk executor。

仓库提供 `kuavo_server/tools/offline_jitter_controls.py` 将两项控制实验合并执行。E0 对每个指定 observation 重复请求；E7 则用同一 response 中的 GR00T 原始 action dict 按 `left_arm, left_gripper, right_arm, right_gripper` 重建 Kuavo chunk，并与 adapter 输出逐元素比较。这样不需要在同一 GPU 上加载第二份模型，也不会把 diffusion 的两次独立采样误差误判成 adapter 映射误差。

Bottle 推荐帧：正常帧 0、最大 h8 手臂边界前帧 48、夹爪转换前帧 80：

```bash
uv run --project kuavo_model/external_models/gr00tn1d7 \
  python kuavo_server/tools/offline_jitter_controls.py \
  --dataset-root /root/bayes-tmp/kuavo_dataset/task1_bottle_pick_lerobot/lerobot_v2.1 \
  --episode 0 \
  --frames 0,48,80 \
  --repeat-count 10 \
  --prompt "Pick up the bottles from the table." \
  --output-dir outputs/jitter/bottle/controls_e0_e7
```

Apple 推荐帧：正常帧 0、最大 h8 手臂边界前帧 24、夹爪转换前帧 80：

```bash
uv run --project kuavo_model/external_models/gr00tn1d7 \
  python kuavo_server/tools/offline_jitter_controls.py \
  --dataset-root /root/bayes-tmp/kuavo_dataset/task2_pick_apple_messy_lerobot/lerobot_v2.1 \
  --episode 0 \
  --frames 0,24,80 \
  --repeat-count 10 \
  --prompt "Pick up the apple from the table." \
  --output-dir outputs/jitter/apple/controls_e0_e7
```

每个任务运行前必须启动对应 checkpoint 的 adapter server。输出 `summary.json` 和 `controls.npz`。E7 的 `pass_at_1e-7` 应为 true；E0 重点比较正常帧、异常边界帧和夹爪转换帧的 `repeat_deviation_l2`、`per_element_std` 与 `first_action_per_element_std`。

#### E0/E7 实际结果

Bottle 实际测试帧为 `0,48,80`；Apple 实际测试帧为 `0,48,80`。每帧重复推理 10 次，execution horizon 为 8。

E7 结果：

| 任务 | 最大逐元素映射误差 | `pass_at_1e-7` | 结论 |
| --- | ---: | --- | --- |
| Bottle | 0.0 | true | action key 顺序、拼接和 wire response 一致 |
| Apple | 0.0 | true | action key 顺序、拼接和 wire response 一致 |

因此现有 jitter 证据不能归因于 adapter 的 `left_arm/left_gripper/right_arm/right_gripper` 拼接错误。E7 比较的是同一次模型采样的 raw 与 converted 输出，可以避免 diffusion 独立采样导致的假差异。

E0 手臂与夹爪分离结果：

| 任务/帧 | 语义 | 手臂逐元素 std 均值 | 手臂 std P95 | 手臂 std 最大值 | 夹爪 std 最大值 |
| --- | --- | ---: | ---: | ---: | ---: |
| Bottle 0 | 普通基线 | 0.0105 | 0.0171 | 0.0214 | 0.0038 |
| Bottle 48 | h8 手臂异常边界前 | 0.0120 | 0.0203 | 0.0273 | 0.0103 |
| Bottle 80 | 夹爪转换前 | 0.0118 | 0.0199 | 0.0269 | 0.3731 |
| Apple 0 | 普通基线 | 0.0070 | 0.0154 | 0.0341 | 0.0041 |
| Apple 48 | 普通/非目标边界帧 | 0.0061 | 0.0104 | 0.0150 | 0.0006 |
| Apple 80 | 夹爪转换前 | 0.0069 | 0.0138 | 0.0329 | 0.3559 |

解释：

- Bottle frame 48 的手臂 std 比 frame 0 略高，但只约高 14%，没有数量级差异；Apple frame 48 反而低于基线。现有证据不支持“异常边界主要由该 observation 的采样方差突然增大”这一假设。
- 两个任务的普通手臂预测都存在非零 diffusion 随机性（逐元素 std 均值约 `0.006–0.012 rad`），可能贡献一部分边界变化，但不是异常边界的唯一解释。
- frame 80 的巨大整体重复偏差由夹爪支配。Bottle 的最大夹爪 std 为 `0.3731`，Apple 为 `0.3559`；高方差集中在预测未来第 9–11 步，表示模型对“何时闭合”的时序呈现近似多模态，而不是当前第一步手臂输出不稳定。
- 两个任务在 frame 80 的 first-action std 仍接近普通水平。使用 h8 时，不执行最不确定的第 9–11 步远期预测，这进一步支持 h8 而不是 h16。

Apple 原计划用于最大 h8 手臂边界的前帧是 24，首轮实际上传结果使用了 frame 48，因此补测 frame 24：

```bash
uv run --project kuavo_model/external_models/gr00tn1d7 \
  python kuavo_server/tools/offline_jitter_controls.py \
  --dataset-root /root/bayes-tmp/kuavo_dataset/task2_pick_apple_messy_lerobot/lerobot_v2.1 \
  --episode 0 \
  --frames 24 \
  --repeat-count 10 \
  --prompt "Pick up the red apple from the table." \
  --output-dir outputs/jitter/apple/controls_e0_frame24_h8
```

补测结果：

- 全 chunk `per_element_std` mean/P95/max 为 `0.0182/0.0871/0.1246`，表面上明显高于 Apple 普通帧；
- 但 first-action std mean/P95/max 仅为 `0.0050/0.0085/0.0094`，与普通帧接近；
- arm-only std 从 step 0 的约 `0.0055` 逐步增长，在 step 8 约为 `0.0205`，step 12–16 约为 `0.0366–0.0412`；
- 最大方差来自远期 step 12–16 的右臂 J1/J4/J5，其中 RJ4 最大 std 为 `0.1246 rad`；
- 夹爪 std 最大仅 `0.0041`，本帧的高方差与夹爪无关；
- E7 映射误差仍为 `0.0`。

因此 frame 24 的异常不是“当前第一步随机”，而是预测越远、右臂轨迹分歧越大。h8 只执行前 8 步，可避免直接执行方差最高的 step 9–16；执行到第 8 步前重新观测和推理是合理的 receding-horizon 策略。

至此 E0/E7 均可关闭：模型存在随预测距离增长的 diffusion 不确定性，但 h8 已截掉最不稳定的远期部分；代码修正不应继续针对 adapter 映射，也不必先尝试完全消除模型随机性。下一实现重点转向 h8 的跨 chunk 手臂融合和夹爪独立状态机。

#### Step 2：实施低风险配置和夹爪修正

1. 默认设置 `execution_horizon=8`。
2. 使用异步推理，并在动作开始前完成一次 warm-up。
3. 记录 buffer 深度、空队列、hold-last-action 和 deadline miss；队列目标至少保有一个待执行 chunk。
4. 夹爪使用独立的阈值 + 迟滞 + 状态锁存，不对夹爪做与手臂相同的连续轨迹滤波。

#### Step 3：离线验证轻量跨 chunk 融合

模型仍预测 16 步，但只执行 8 步。在新 chunk 到达时，将“上一 chunk 尚未执行的后 8 步”与“新 chunk 的前 8 步”做可配置的线性/cosine ramp 融合；先离线重算边界位置跳变和速度 cosine，再决定是否进入执行链路。融合不得修改已下发动作，也必须经过关节限位、速度和加速度检查。

验收条件应相对未融合 h8 baseline 定义：

- 两个任务的 arm-only boundary jump mean/P95 均下降；
- velocity cosine 的 P05/median 上升；
- chunk 内加速度 P95 不恶化；
- first-action 与数据集 action 的误差没有显著增大；
- h8 推理/融合总耗时仍显著低于 800 ms。

仓库工具：`kuavo_server/tools/offline_chunk_blending_eval.py`。该工具只读取已有 `chunks.npz`，不需要 GPU、checkpoint 或运行中的 adapter server。它输出：

- `summary.json`：融合前后 arm-only 指标和相对变化；
- `boundary_comparison.csv`：逐边界位置跳变与速度 cosine；
- `blended_chunks.npz`：baseline 与 blended 的实际执行 chunks。

融合规则：模型仍预测 16 步、执行 8 步；新 chunk 的前 N 步由上一预测未执行 tail 与新预测 head 做 linear/cosine ramp。`--new-chunk-start-weight` 控制第一个融合点采用多少新预测：0 表示完全延续旧 tail，1 表示完全采用新 chunk。只融合14维手臂；夹爪可选择 passthrough 或 `[0.35,0.65]` 迟滞锁存。

Bottle 示例：

```bash
python kuavo_server/tools/offline_chunk_blending_eval.py \
  --input-dir outputs/jitter/bottle/offline_ep0_h08 \
  --execution-horizon 8 \
  --blend-steps 8 \
  --blend-mode cosine \
  --new-chunk-start-weight 0.25 \
  --gripper-mode hysteresis \
  --output-dir outputs/jitter/bottle/blended_h08_cosine_w025
```

Apple 示例：

```bash
python kuavo_server/tools/offline_chunk_blending_eval.py \
  --input-dir outputs/jitter/apple/offline_ep0_h08 \
  --execution-horizon 8 \
  --blend-steps 8 \
  --blend-mode cosine \
  --new-chunk-start-weight 0.25 \
  --gripper-mode hysteresis \
  --output-dir outputs/jitter/apple/blended_h08_cosine_w025
```

#### 离线融合 pilot 结果

首先测试 cosine、8 个融合点、new-chunk start weight 0.0。Bottle 的 boundary jump mean/P95 分别下降 `47.4%/54.1%`，Apple 分别下降 `53.1%/63.4%`；两个任务的速度 cosine P05/median 均改善。但 Apple first-action error mean 从 `0.0898` 上升至 `0.0948`、P95 从 `0.1194` 上升至 `0.1468`，说明完全延续旧 tail 会减弱新观测修正。

随后扫描 start weight，cosine、8 步、start weight 0.25 在两个任务上取得更均衡结果：

| 任务 | 指标 | h8 baseline | cosine-8-w0.25 | 变化 |
| --- | --- | ---: | ---: | ---: |
| Bottle | arm boundary jump mean | 0.1013 | 0.0507 | -49.9% |
| Bottle | arm boundary jump P95 | 0.1644 | 0.0859 | -47.7% |
| Bottle | velocity cosine P05 | -0.5784 | -0.4719 | +0.1064 |
| Bottle | velocity cosine median | 0.1027 | 0.3540 | +0.2513 |
| Bottle | intra-chunk acceleration P95 | 9.4148 | 8.2040 | -12.9% |
| Bottle | first-action error mean | 0.0973 | 0.0824 | -15.3% |
| Apple | arm boundary jump mean | 0.0789 | 0.0417 | -47.1% |
| Apple | arm boundary jump P95 | 0.1827 | 0.0817 | -55.3% |
| Apple | velocity cosine P05 | -0.1510 | -0.1074 | +0.0436 |
| Apple | velocity cosine median | 0.4031 | 0.5495 | +0.1464 |
| Apple | intra-chunk acceleration P95 | 7.6497 | 6.7590 | -11.6% |
| Apple | first-action error mean | 0.0898 | 0.0879 | -2.1% |

夹爪迟滞状态机在 Bottle 左右夹爪各产生一次状态切换，在 Apple 仅右夹爪切换一次，没有出现 chatter。pilot 数据支持 `cosine + blend_steps=8 + new_chunk_start_weight=0.25` 作为下一候选，但离线 teacher-forced 指标不能替代真机安全验证。

2026-08-14 将服务器输出重新传回本地后进行了复核。上传目录为 `outputs/jitter/jitter/{bottle,apple}`（传输时额外嵌套了一层 `jitter`）；其中 E0/E7、h4/h8/h16 的 `summary.json` 与本地已有结果逐字节一致。服务器生成的 `blended_h08_cosine_w025` 与本地重算结果除 `experiment.input_dir` 的绝对路径不同外，所有数值指标一致。因此上述 Bottle/Apple pilot 结果已经通过跨机器复现，目录层级重复不影响结论。此次上传仍只有 episode 0 和候选 B，不应被视为多 episode 参数验证。

#### 下一轮离线实验

为避免对单个 episode 过拟合，下一轮不再扫描所有组合，只比较以下候选，并扩展到每个任务至少3个 episodes：

| 候选 | Blend mode | Blend steps | Start weight | 目的 |
| --- | --- | ---: | ---: | --- |
| A | cosine | 8 | 0.00 | 最强连续性基线 |
| B（推荐） | cosine | 8 | 0.25 | 平滑与新观测响应折中 |
| C | cosine | 8 | 0.50 | 更偏重新预测，检查任务误差 |
| D | linear | 8 | 0.25 | 检查 ramp 形状敏感性 |

每个 episode 选择同一 checkpoint、相同 prompt 和相同 h8 chunks。先运行 `offline_jitter_diagnostic.py --episode N` 生成输入，再运行本工具。若候选 B 在 Bottle/Apple 的至少3个 episodes 上持续满足上述验收条件，则进入执行链路实现；否则根据 first-action error 与 boundary cosine 选择 B/C，而不是盲目增加融合强度。

工具现支持 `--candidate-matrix` 批量模式。`--input-dir` 后可一次传入多个由 `offline_jitter_diagnostic.py` 生成的目录；工具会对每个输入运行 A/B/C/D，在 `<output-dir>/<输入标签>/<候选>/` 保存单次结果，并在 output 根目录生成 `aggregate_summary.csv` 和 `aggregate_summary.json`。旧的单输入、单候选命令保持兼容。

服务器上完成 episode 0/1/2 的 h8 diagnostic 后，可一次运行：

```bash
uv run --project kuavo_model/external_models/gr00tn1d7 \
  python kuavo_server/tools/offline_chunk_blending_eval.py \
  --input-dir \
    outputs/jitter/bottle/offline_ep0_h08 \
    outputs/jitter/bottle/offline_ep1_h08 \
    outputs/jitter/bottle/offline_ep2_h08 \
    outputs/jitter/apple/offline_ep0_h08 \
    outputs/jitter/apple/offline_ep1_h08 \
    outputs/jitter/apple/offline_ep2_h08 \
  --candidate-matrix \
  --execution-horizon 8 \
  --gripper-mode hysteresis \
  --output-dir outputs/jitter/blending_matrix_h08
```

该命令不连接 adapter server，也不重新推理，只消费各目录中的 `chunks.npz`。输入标签由“父目录名 + 输入目录名”生成，例如 `bottle_offline_ep1_h08`，因此 Bottle 和 Apple 不会发生输出覆盖。汇总表每行包含 boundary mean/P95、velocity cosine 改变量、acceleration P95 和 first-action error，可直接用于跨 episode 比较。若尚未生成 episode 1/2，工具会明确报告缺失的 `chunks.npz`，不会静默跳过。

#### 三 episode A/B/C/D 结果（2026-08-14）

Bottle 和 Apple 的 episode 0/1/2 均已完成 h8 diagnostic，矩阵输出位于 `outputs/jitter/blending_matrix_h08`，共包含 24 组实验。下表为三个 episode 的相对变化/改变量的算术平均；负的 jump、acceleration 和 error 表示改善，正的 cosine 表示改善。

| 任务 | 候选 | Boundary mean | Boundary P95 | Cosine P05 | Cosine median | Acceleration P95 | First-action error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bottle | A | -44.8% | -43.6% | +0.104 | +0.061 | -11.2% | +3.9% |
| Bottle | B | -48.8% | -44.9% | +0.046 | +0.100 | -15.2% | -5.7% |
| Bottle | C | -40.2% | -35.3% | +0.081 | +0.098 | -13.9% | -9.9% |
| Bottle | D | -48.8% | -44.9% | +0.054 | +0.059 | -19.9% | -5.7% |
| Apple | A | -45.7% | -50.1% | -0.044 | +0.014 | -9.9% | +12.0% |
| Apple | B | -42.9% | -46.5% | -0.013 | +0.067 | -13.3% | +3.0% |
| Apple | C | -32.8% | -35.0% | +0.022 | +0.122 | -12.7% | -2.7% |
| Apple | D | -42.9% | -46.5% | -0.010 | +0.092 | -18.7% | +3.0% |

候选 B 在六个 episode 上都降低 boundary mean、boundary P95 和 acceleration P95，证明跨 chunk 融合对位置跳变的改善不是 episode 0 偶然现象。不过严格验收尚未全部通过：Bottle episode 1 的 cosine P05 下降 `0.199`，Bottle episode 2 的 cosine median 下降 `0.070`；Apple episode 1 的 cosine P05/median 分别下降 `0.240/0.040`，且 first-action error 增加 `14.6%`。B 的六 episode 逐项结果如下：

| Episode | Boundary mean | Boundary P95 | Cosine P05 | Cosine median | Acceleration P95 | First-action error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bottle 0 | -49.9% | -47.7% | +0.106 | +0.251 | -12.9% | -15.3% |
| Bottle 1 | -50.3% | -38.4% | -0.199 | +0.118 | -16.7% | +8.1% |
| Bottle 2 | -46.2% | -48.5% | +0.231 | -0.070 | -15.9% | -10.0% |
| Apple 0 | -47.1% | -55.3% | +0.044 | +0.146 | -11.6% | -2.1% |
| Apple 1 | -42.3% | -40.6% | -0.240 | -0.040 | -12.0% | +14.6% |
| Apple 2 | -39.2% | -43.7% | +0.158 | +0.095 | -16.2% | -3.4% |

结论：B 仍是强位置连续性候选，但不能按原标准直接进入默认实时配置。C 对新观测响应更强，在两个任务的平均 cosine 与 first-action error 上更稳定，代价是 boundary 降幅较小；D 与 B 的边界和 first-action 指标相同，并取得更低的 acceleration P95，但 Apple 的 cosine P05 仍略微退化。下一步应针对失败最明显的 Bottle episode 1、Apple episode 1 检查逐边界 CSV，判断 cosine 退化是少数静止/低速度边界造成的数值敏感性，还是实际方向反转；确认后再在 B/C/D 中选择实时候选。

#### 逐边界速度与方向审计

新增工具 `kuavo_server/tools/offline_blending_boundary_audit.py`。它读取矩阵中保存的 `blended_chunks.npz`，重新计算每个边界两侧的14维手臂速度范数和 cosine，并输出：

- `boundary_audit.csv`：逐输入、候选、边界的速度、位置跳变、cosine delta、方向反转和严重退化标记；
- `summary.json`：每个 episode 及每个任务的有效边界数、退化数、`delta <= -0.2` 严重退化数和方向反转数。

默认仅当边界两侧速度范数均不低于 `0.01 rad/step` 时才解释 cosine。对现有矩阵运行后，104 个任务边界（Bottle 57、Apple 47）在 B/C/D 下全部超过该阈值，因此此前 cosine 退化不是静止边界的除零或低速数值噪声。

| 任务 | 候选 | 有效边界 | Delta mean | Delta median | Delta P05 | 严重退化 | 方向反转 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bottle | B | 57 | +0.080 | +0.043 | -0.424 | 8 | 2 |
| Bottle | C | 57 | +0.081 | +0.090 | -0.183 | 3 | 1 |
| Bottle | D | 57 | +0.046 | +0.053 | -0.422 | 10 | 3 |
| Apple | B | 47 | +0.034 | -0.006 | -0.322 | 7 | 2 |
| Apple | C | 47 | +0.057 | +0.018 | -0.193 | 2 | 1 |
| Apple | D | 47 | +0.027 | +0.013 | -0.343 | 7 | 4 |

失败最明显的两个 episode 也证实是真实方向变化：Bottle episode 1 的 B boundary 1 cosine 从 `0.123` 变为 `-0.291`，同时位置跳变从 `0.2110` 降到 `0.0746`；Apple episode 1 的 boundary 14 从 `0.309` 变为 `-0.082`，位置跳变从 `0.0666` 降到 `0.0271`。融合在降低位置不连续的同时，个别边界会牺牲速度方向连续性。

因此排除“低速 cosine 假退化”假设。固定 B 不再作为唯一推荐，D 也因严重退化和反转更多而排除；C 在两个任务上都只有较少的严重退化和方向反转，同时平均 first-action error 下降，作为进入实时链路前的首选保守候选。下一 pipeline 是先以 `cosine8-w0.50` 实现可开关的 arm-only runtime blending，保留未融合 h8 作为 safety baseline；真机低速验证必须同时记录 boundary position、velocity direction、command-feedback、deadline miss 和 buffer empty。若 C 仍出现可见停顿，再实现基于新旧 chunk 分歧的自适应 start weight，而不是继续使用固定更强融合。

审计命令：

```bash
uv run --project kuavo_model/external_models/gr00tn1d7 \
  python kuavo_server/tools/offline_blending_boundary_audit.py \
  --matrix-dir outputs/jitter/blending_matrix_h08 \
  --candidates B_cosine8_w025 C_cosine8_w050 D_linear8_w025 \
  --min-velocity-norm 0.01 \
  --output-dir outputs/jitter/blending_boundary_audit_h08
```

#### Step 4：有真机后再检查类别 C/D

恢复真机后，先用 `h8 + async + 夹爪状态机` 跑低速实验，同时记录 command-feedback、控制 tick 和 buffer。若预测/下发命令平滑但反馈仍抖动，再进入驱动控制、插值、增益、编码器和机械结构排查；若 jitter 与 buffer empty/hold 恢复重合，则处理类别 D，而不是继续平滑模型输出。

## 9. 参考资料

- NVIDIA Isaac-GR00T：`getting_started/real_world_deployment.md`，章节 “Common Issues: Jittering and Stop-and-Go”。
- 本仓库镜像文档：`kuavo_model/external_models/gr00tn1d7/getting_started/real_world_deployment.md`。
- 本仓库 GR00T N1.7 adapter：`kuavo_server/adapters/isaac_gr00t_n17.py`。
- 本仓库真机异步执行：`kuavo_deploy/src/eval/real_async_test.py`。
- 本仓库真机同步执行：`kuavo_deploy/src/eval/real_single_test.py`。
