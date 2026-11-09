# Passive VLM Monitor：只观察、记录

脚本 `vlm_test/passive_vlm_monitor.py` 独立运行，不启动 VLA，不调用服务端推理接口，不构造机器人环境或 SDK，不创建动作/暂停/切换 publisher 或 service client。`recovery` 只产生日志和终端提示。Ctrl+C 只结束 Monitor，会等待自己的当前推理完成以保存审计记录。

本阶段仅提供脚本和用户操作步骤，未运行 Monitor、ROS 探测、模型或单元测试。不要把这份说明当作已经通过运行验证。

## 人工摆放：只开头相机、不运行 VLA

使用新增的 `--camera-only`。无需启动 LingBot server、eval、运控或任何 VLA 推理；只要求已有头相机话题持续发布。本脚本不会替你启动或重启相机。

```bash
cd /home/kuavo/jiayi/kuavo_learning_studio
source /opt/ros/noetic/setup.bash
export PYTHONPATH="$PWD/outputs/qwen_quant_deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  /home/kuavo/miniforge3/envs/qwen35/bin/python -u -m vlm_test.passive_vlm_monitor \
  --deploy-config configs/deploy/deploy_lingbot_package.yaml \
  --camera-only
```

此模式：

- 只订阅配置中的头相机（以及ROS自身需要的时钟），不订阅 pause/stop，不读取 VLA 执行日志。`VLA=not_monitored` 是正常状态。
- 默认使用 `vlm_test/prompts/package_front_edge_tolerant_v4.json`，有/无 VLA 都采用同一修订规则：用户确认的轻微边角悬出仍为 normal；只有明显进入受训近侧低位恢复姿态才为 recovery。v2/v3文件和旧结果保留，可用 `--prompt` 显式复现旧定义。
- 每次输入**仍只有当前一张 RGB**。在这张图中让 VLM 重新辨认黄色平台的实体边界、朝机器人身体的前缘和整个包裹轮廓；不把运单矩形、桌边或黄色胶带边界当成平台前缘。
- 默认读取 `/media/data/jiayi/dataset/roi_config.json` 作为历史标定审计快照，记录路径、SHA256和原坐标。**坐标不发送给模型，不用于当前图的裁剪/越界判断，也不把预览图作为第二张模型输入。** 移动机器人或桌子后，不沿用旧的 y=437 线。
- 这是 VLM 隐式视觉辨边，未实现或声称已实现几何跟踪、动态分割或可靠自动重标定。v4尚未实测，不能保证比v3更准；旧结果不能直接当作修订标签下的准确率。
- 图像、时间戳、原始输出、解析、时效门控和日志保持原实现。`session.json` 明确记录 `camera_only=true`、`geometry_mode` 和 `roi_reference.used_as_current_pixel_roi=false`。

建议人工依次摆放，每个稳定位置保持3–5秒、摆完后将手移开：

| 场景 | 参考标签 |
|---|---|
| 平台无包裹，或包裹完全处于平台内 | normal |
| 粉色/蓝色包裹主体仍在正常平台抓取位，仅边角/末端轻微悬出（用户确认示例） | normal |
| 包裹主体/正常抓取部位明显滑到机器人近侧较低恢复区域，后部可以仍有部分支撑 | recovery |
| 包裹在左侧待处理区或右侧篮中，没有近侧前缘异常包裹 | normal |
| 多包裹，其中一个明确处于上述滑落低位恢复姿态 | recovery |
| 将前缘包裹放回平台内部 | normal |

再改变桌子/相机相对位置，重复“平台内部→允许的轻微悬出→明确滑入近侧恢复区→放回”，检查预测是否跟随**实际平台**而不是旧图坐标变化。不要在遮挡/搬动瞬间直接评估稳定摆放效果；可单独标记这些过渡帧。运动中人手持包裹与留在前缘的放置状态也应分开复核。

使用保存的 `input_image` 对照预测；新测试不会自动复制上一轮人工标签或计算准确率。依然不发送任何机器人动作或切换指令。

## 1. 两类配置与实际相机链路

- Server checkpoint：仓库现有路径是 `configs/server/launcher.yaml`，`lingbotvla.args.checkpoint` 等由你设置，仍使用现有 `kuavo_server/launch.py lingbotvla` 流程。Monitor 不解析、验证或修改服务端 checkpoint。
- Deploy 相机配置：`python kuavo_deploy/eval.py configs/deploy/deploy_lingbot_package.yaml` 所用的同一份 YAML。Monitor 的 `--deploy-config` 必须指向它，**不是 server launcher YAML**。若你使用其他 deploy profile，两个进程传同一个路径。
- 现有调用链：`eval.py` → `src/scripts/script.py --task run` → `real_single_test.py`（`async_inference=false`）；异步部署 profile 则进入现有 `real_async_test.py`，Monitor 同样只旁路订阅。
- `load_kuavo_config` 合并 deploy simple/total 配置；本仓库 package profile 的 `head_cam_h` 来自 `configs/deploy/total/deploy_total.yaml`：`/cam_h/color/image_raw/compressed`，`sensor_msgs/CompressedImage`。左右腕分别是 `/cam_l/color/image_raw/compressed`、`/cam_r/color/image_raw/compressed`。
- `ObsBuffer.rgb_callback` 解码为 RGB，并把 `msg.header.stamp.to_sec()` 存为 VLA 图像时间。Monitor 用同一配置解析 topic，复用 `ROSManager.register_subscriber` 的 `queue_size=1` / `tcp_nodelay`，复用 `decode_compressed_rgb` 解码 RGB。

以上是静态代码确认，未连接你的实际 ROS feed。用户运行 `--probe-only` 后，日志才会确认实际 publisher、`header.frame_id`、`seq`、`stamp_ns`、ROS/墙钟/monotonic 接收时刻和图像 format。Monitor 与 VLA 使用同一来源，但独立采样，**不保证选择了同一条 ROS 消息或同一 VLA step**；通过 header 时间分析对应关系。

## 2. 环境与只读探测

从项目根目录运行。Qwen 环境需要 Torch、Transformers（含 Qwen3.5）、accelerate、Pillow、PyYAML、termcolor，以及能导入 ROS1 `rospy` / `sensor_msgs` / `std_msgs`。不需要在 Qwen 环境中导入 cv_bridge 或 Kuavo SDK。NF4 的 bitsandbytes 0.50.2 已直接安装到 `qwen35` 环境，无需额外的 quant-deps PYTHONPATH。`outputs/qwen_quant_deps` 是此前离线实验的独立安装目录，保留兼容。

```bash
cd /home/kuavo/jiayi/kuavo_learning_studio
source /opt/ros/noetic/setup.bash
export PYTHONPATH="$PWD/outputs/qwen_quant_deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"

/home/kuavo/miniforge3/envs/qwen35/bin/python -m vlm_test.passive_vlm_monitor \
  --deploy-config configs/deploy/deploy_lingbot_package.yaml \
  --probe-only
```

该命令只订阅5秒，不导入 Torch、不加载 VLM。检查 `camera_probe` / `camera_source_confirmed`：header 必须正数，header→当前 ROS 时间和接收后 monotonic 年龄均应小于默认0.5秒。ROS publisher 与 Monitor 必须使用相同的时钟域；不同机器需对时。零 header 不回退到“刚收到所以新鲜”。没有可用帧时退出非零，不输出 normal。

## 3. 启动独立 VLM Monitor

### 已确认的本机头相机时间域问题

2026-10-28 只读检查确认：`kuavo-up` 是上位机 `ubuntu`，同时有 `192.168.0.187` 和 `192.168.26.12` 两个地址。头相机是 Orbbec，左右腕是 RealSense。上位机运行的启动文件为 `/home/leju_kuavo/kuavo_data_pilot/src/kuavo_data_pilot_bin/launch/start_camera.launch`。

源图像 header 曾为0，随后为约11863秒，而 Monitor ROS系统时间约1793181xxx秒。新版直接记录原始 `secs/nsecs`，这种非Unix头时间与Unix ROS时间会明确报告 `header_clock_domain_mismatch`，不会被当作当前画面。

远端 Orbbec 源码 `OBCameraNode::getFrameTimestampUs` 支持 `device/global/system`，`system` 分支读取 `frame->systemTimeStampUs()`。现有 head include 未显式设置时间域，其被包含的 launch 默认 `global`。最小候选修复是在**头相机 include 内**加入：

```xml
<include file="$(find orbbec_camera)/launch/gemini_330_series.launch">
    <arg name="camera_name" value="cam_h"/>
    <arg name="time_domain" value="system"/>
    <!-- 其余原有参数保持原样 -->
</include>
```

对应补丁保存在 `vlm_test/patches/orbbec_head_system_time.patch`。SSH随后被远端反复关闭，**远端文件未成功修改，也未重启相机**。用户在上位机保存备份并修改后，按原有方式重启相机启动服务；不要重复启动第二套相机进程。仅修改launch不会改变已经运行的节点。重新检查实际header、系统时间与Monitor probe，确认驱动的系统帧时间确实可用，才继续监控。无需修改 `/use_sim_time`，也不要在接收端把时间戳重写成当前时间。

在用户已选择正确 ROS master、相机来源的情况下，单独打开终端：

```bash
cd /home/kuavo/jiayi/kuavo_learning_studio
source /opt/ros/noetic/setup.bash
export PYTHONPATH="$PWD/outputs/qwen_quant_deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  /home/kuavo/miniforge3/envs/qwen35/bin/python -u -m vlm_test.passive_vlm_monitor \
  --deploy-config configs/deploy/deploy_lingbot_package.yaml \
  --camera-key head_cam_h \
  --precision nf4 \
  --vla-log log/kuavo_deploy/kuavo_deploy.log
```

默认模型 `/media/data/jiayi/hf_model/Qwen3.5-9B`，加载本地权重，NF4 double quant + BF16 compute、SDPA、关闭 thinking、greedy、最多24 tokens。默认 Prompt 是 `vlm_test/prompts/package_front_edge_tolerant_v4.json` 的 `skill_prompt`，无 Skill Registry、检测器或多帧历史。

这里假定 GPU1 空闲且 VLA 使用其他 GPU；按实际分配修改 `CUDA_VISIBLE_DEVICES`。独立进程保证没有同步等待关系，**共用 GPU/CPU/磁盘仍可能影响 VLA 时延**，因此优先分离 GPU。Monitor 不会自动选择或抢占设备。

正常 VLA 的 server/deploy 仍由用户按原流程启动，不需要把 Monitor 导入 `eval.py` 或 `real_single_test.py`。以上命令不启动任何机器人动作。`policy_type=client` 时，deploy 中的 `pretrained_path` 不代表服务端实际 checkpoint，session 会明确记录此限制。

终端 stdout 仅输出被接受且当时未过期的：

```json
{"skill":"normal"}
{"skill":"recovery"}
```

stderr 每秒显示 `CURRENT`、health、busy 和 VLA 观测状态；出现 recovery 只提示。失败时没有伪造的二分类输出，stderr 显示 `UNAVAILABLE`。stdout 是历史判断流，不应把最后一行永久当作当前状态。

## 4. 调度和时效规则

- ROS callback 只更新容量1的压缩图像槽；1Hz 选择最新的一条，解码、PNG保存和 GPU 推理都在独立单 worker 中。
- 模型加载或推理占用 worker 时，采样 tick 记录 `sample_skipped`，不会排队旧图。若推理超过1秒，实际预测率会低于1Hz；不并发叠加 GPU 请求。
- 输入选择、预处理后、进入 GPU generation 前均要求 header 年龄与接收后 monotonic 年龄 ≤ `--max-frame-age`，默认0.5秒。
- 推理返回后要求两种年龄 ≤ `--max-result-age`，默认1.5秒，同时相机最新帧也必须可用。超时结果保留原文及 parsed 历史记录，但 `accepted_current=false`，不显示为当前分类。
- 50ms watchdog 检查当前结果与相机时效，过期/断流清空 `current_skill`。这是软件轮询，不是硬实时保证。
- `latest_status.json` 有 `current_valid_until_ros_s` 和0.25秒的 `lease_expires_wall_s`。任何读取它的 UI 都必须检查 lease，避免 Monitor 被杀死或 IO 卡住后继续显示旧 normal；当前结果存在时状态文件约10Hz刷新。
- 零 header、未来 header（容差0.05秒）、重复/乱序 header、暂停的仿真时钟和 rosbag 回绕都会拒绝/失效处理。不将墙钟、ROS 仿真时间、monotonic 时间混算。
- 默认仅对 VLM 将解码 RGB 等比例缩放到宽640（848×480来源→640×362），保存的 PNG 就是交给 Processor 的输入，无标注覆盖。`--resize-width 0` 可保留源尺寸，均不修改 VLA 预处理。

## 5. 日志与 VLA 关联

每次启动自动新建 `outputs/passive_vlm_monitor/<UTC时间>/`，不覆盖历史：

| 文件 | 内容 |
|---|---|
| `session.json` | deploy 配置、解析后 topic、VLA 配置、完整 Prompt及SHA256、模型和阈值 |
| `images/*.png` | 每次实际判断的单帧 RGB 输入，文件名含时钟epoch、接收序号和 header纳秒 |
| `events.jsonl` | probe、跳帧、模型就绪、每次判断、VLA日志行和heartbeat |
| `latest_status.json` | 有有效期的当前分类；缺失/过期/失败为 `current_skill:null` |
| `monitor.log` | 可直接 tail 的人类可读终端日志 |

`judgment` 包含：`frame.stamp_s/stamp_ns`、frame_id、publisher、接收的 ROS/墙钟/monotonic 时刻，`input_image`、原始尺寸与输入尺寸，`completed_at_utc/completed_wall_s/completed_ros_s`、`raw_output`、严格解析的 `parsed`、`status`、`accepted_current`、`vla_at_sample`、`vla_at_completion`。

延迟：`header_to_completion_ros_seconds` 是 header→完成的端到端 ROS 时间；`receipt_to_completion_seconds` 是 Monitor 收到图像→完成的 monotonic 时间；`timing.generation_seconds` 单独记录GPU prefill+decode；`completion_to_log_seconds` 记录 worker 完成到主线程处理的时间。ROS header 是否精确对应曝光时刻取决于相机驱动，不能把网络延迟当作已知零。

VLA 关联复用已有 `/kuavo/pause_state`、`/kuavo/stop_state` 只读订阅，并增量读取 eval 已有的 `log/kuavo_deploy/kuavo_deploy.log`。观察 `Step N time` / `action infer time` / pause / episode完成等日志；Monitor 启动前的历史行跳过，日志截断/轮换清空旧证据。日志没有近期活动时状态为 `unknown`，**不把没有 pause 请求等同于 VLA 正在运行**。操作请求不是物理停机确认，日志写入可能有缓冲，当前未修改 VLA 增加精确逐step telemetry。直接运行部署脚本而不经 eval 重定向时，用 `--vla-log` 指向实际输出文件，否则 VLA 状态会 unknown。

示例只读查看（将目录替换为本次启动打印的真实路径）：

```bash
tail -f outputs/passive_vlm_monitor/<本次UTC目录>/monitor.log
tail -f outputs/passive_vlm_monitor/<本次UTC目录>/events.jsonl
```

## 6. 用户执行的离线验证

以下步骤尚未执行。先不连接控制器，只在隔离 ROS master 回放已有图像/状态 bag；**不要把整条含动作命令的 bag 回放到机器人 master**。

1. 可先执行无ROS、无模型的单元检查：
   ```bash
   /home/kuavo/miniforge3/envs/qwen35/bin/python -m unittest vlm_test.test_passive_vlm_monitor
   ```
   覆盖旧header新接收、零/未来时间、推理结果超时、重复/乱序、仿真时钟冻结/回绕、严格JSON、推理失败不回退normal、输入保存、日志截断、入口无动作API调用。
2. 新终端 `source /opt/ros/noetic/setup.bash` 后启动 `roscore -p 11312`。其他离线终端设置 `export ROS_MASTER_URI=http://127.0.0.1:11312`，不要沿用实际机器人 master。
3. 在此隔离 master 执行 `rosparam set /use_sim_time true`，然后启动 Monitor。先等 `VLM ready`（模型加载可能数分钟，没有图像也可加载）。
4. 同一隔离 master 仅回放图像和只读请求话题：
   ```bash
   rosbag play --clock /path/to/package_observations.bag --topics \
     /cam_h/color/image_raw/compressed /kuavo/pause_state /kuavo/stop_state
   ```
   若使用 `--pause`，等模型就绪后按空格开始；暂停时应显示时钟停滞，不产生当前标签。不能在原视频MP4上直接运行本脚本；它需要带 header 的 ROS 图像来源。
5. 用 `--probe-only` 确认实际 topic/header，再用 `--max-samples 5` 做短测试；核对每个 judgment 都有输入PNG、原始输出、parsed、时间和延迟。没有VLA运行日志时，VLA状态为unknown是正确行为。
6. 停止图像回放：默认约0.5秒后应显示 `UNAVAILABLE`、`current_skill:null`；恢复回放只取最新帧。回绕 bag 应改变epoch，回绕前在途结果不得作为当前标签。
7. 用 `--max-result-age 0.5 --max-frame-age 0.5` 或在单元 fake adapter 中模拟慢推理，检查晚到结果为 `stale_result`，但仍保留原图和原文。未知/额外字段/Markdown输出应为 `invalid_output`，不输出normal。
8. 离线验证 `/kuavo/pause_state`、`/kuavo/stop_state` 的记录可由带这些消息的 bag 产生；这里只接收。另可用自己的日志副本并传 `--vla-log /tmp/vla_test.log`，手工追加 `Step 1 time: 0.1s` 检查关联，不需运行 VLA 或发动作。
9. 使用 `rosnode list` 找到匿名 `passive_vlm_monitor_...`，`rosnode info <节点名>` 核对没有机器人动作 publishers；只有配置RGB、pause/stop以及仿真时钟等只读订阅。停止 Monitor 不应改变其他进程。

不要用 `src/scripts/script.py --dry_run` 代替此离线验证：现有部署入口在检查该flag之前已构造 `ArmMove`/环境。本 Monitor 不调用这个入口。

## v4标签修订：正常操作可容忍的悬出

用户确认 `vlm_test/prompts/references/tolerated_overhang_normal.png` 为 normal 正例。该图片按原始字节归档，作为任务规则的审计依据，没有作为第二张模型输入。v4用文字描述这个反例：包裹主体和常规抓取部位仍处于平台上的正常操作姿态，只有边角或末端伸出，不应仅凭越线判recovery。

与之区分的正例是先前 `roi_config_preview.jpg` 中明显向近侧下方滑移的恢复姿态。参考图片、ROI坐标都不发送给当前单帧模型；这轮仍然没有标定实际IK可达边界，也没有设定百分比/像素阈值。中间状态需要继续人工标注。

旧v3结果必须保留原定义；例如旧报告中的“单次normal疑似漏检”，如果图像符合这次确认的可容忍悬出，应重新标注为正确normal，而不是强行按旧定义计错。不能仅凭这一张反例批量重标整段视频。v4日志会保存自己的definition_version和完整Prompt，重启Monitor后才会加载新默认；正在运行的旧进程不会热更新。
