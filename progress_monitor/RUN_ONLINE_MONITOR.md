# LingBot package：被动 Process 监控

只记录 `P1 / P2 / P2a / P2b / P3 / P4` 及概率，不控制动作、技能或 SafetySupervisor。
默认关闭。服务端复用动作推理的最终归一化 prefix，按离线代码做 vision-token
masked mean，并模拟离线 float16 特征存储，再使用 probe 内保存的训练集均值/标准差。

已核实的 **全量训练 probe**：
`/media/data/jiayi/outputs/lingbot_progress_pilot/full_probes/vision_unweighted/best.pt`

**已确认允许的差异：** 现有动作推理采用 PIL Resize，而离线提取采用 torchvision
tensor Resize；当前 package 提示词也不同。下面启用命令显式允许这些差异，每条预测
记录 `input_exact_match:false` 和 `input_mismatches`。不会为了监控修改动作预处理或提示词，
因此离线分数不能直接当作在线精度。不加允许参数时，监控遇到差异会停用，动作照常。

## 1. 设置路径

在仓库根目录的同一个终端执行：

```bash
cd /home/kuavo/jiayi/kuavo_learning_studio
PYTHON_BIN=/home/kuavo/miniforge3/envs/lingbotvla/bin/python
PROBE=/media/data/jiayi/outputs/lingbot_progress_pilot/full_probes/vision_unweighted/best.pt
CKPT=/media/data/jiayi/outputs/kuavo_lingbotvla/task3_pacakge_lora/global_step_66480/hf_ckpt
MONITOR_ROOT=/media/data/jiayi/outputs/lingbot_progress_pilot/online_monitor
```

## 2. 先做离线回放（CPU，无 ROS、无 VLA 加载）

```bash
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -B -m progress_monitor.check_online_monitor \
  --mode replay --probe "$PROBE" \
  --features /media/data/jiayi/outputs/lingbot_progress_pilot/full_141/val/episode_010.npz \
  --frames 20 --output "$MONITOR_ROOT/replay_check"
```

成功：`probability_parity:true`、`observation_binding:true`，并自动与 probe 目录中的
`validation_predictions.npz` 对照，显示 `offline_reference_checked:true`。输出
`replay_check/report.json` 和 `replay.jsonl`。只检查历史特征，历史时间不冒充实时帧龄。

## 3. 再做 inference-only A/B 检查（手动 GPU 任务，无 ROS、无机器人动作）

这一步加载一次 VLA，用 Episode 010 的一帧分别进行关闭/开启监控的动作推理。
两次使用同一随机状态，要求动作逐元素相同。生成的动作仅在内存中比较，绝不发送到机器人，
也不启动网络服务。不会训练或修改权重。

```bash
"$PYTHON_BIN" -B -m progress_monitor.check_online_monitor \
  --mode inference-only --probe "$PROBE" --checkpoint "$CKPT" \
  --episode 10 --frames 1 \
  --deploy-config configs/deploy/deploy_lingbot_package.yaml \
  --allow-input-mismatch --output "$MONITOR_ROOT/inference_check"
```

成功：`cases` 中 `actions_identical:true`、`predictions:1`、`actions_never_executed:true`。
输出 `inference_check/report.json` 和 `inference.jsonl`。
如果动作比较失败、probe 不匹配或监控停用，先检查错误，再考虑真机测试。
检查目录必须是新目录；重复检查时修改 `--output`，已有结果不会覆盖。

## 4. 手动启动带监控的 package 服务端

**完成上面两步后再执行。** 当前 `configs/server/launcher.yaml` 指向其他任务，
因此以下命令明确覆盖 package checkpoint、训练配置和归一化文件，不修改原 YAML。
`use_compile` 必须保持 false；这也是当前配置值。

```bash
PACKAGE_ARGS=(
  --checkpoint "$CKPT"
  --training_config_path "$CKPT/lingbotvla_cli.yaml"
  --robot_norm_path /home/kuavo/jiayi/kuavo_learning_studio/kuavo_model/external_models/lingbot-vla/assets/norm_stats/kuavo_package.json
  --qwen25_path /media/data/jiayi/hf_model/Qwen2.5-VL-3B-Instruct
)

"$PYTHON_BIN" kuavo_server/launch.py lingbotvla "${PACKAGE_ARGS[@]}" \
  --progress_monitor \
  --progress_monitor_checkpoint "$PROBE" \
  --progress_monitor_log "$MONITOR_ROOT/process.jsonl" \
  --progress_monitor_allow_input_mismatch
```

如果只想查看实际启动命令，在上述命令末尾加 `--dry-run`。
启动器会用现有 `lingbotvla` conda 环境运行服务端，需终端能找到 `conda`。
加载错误、错误任务的 checkpoint、相机顺序不匹配、日志写失败、队列满等只停用监控，
正常 VLA 推理继续。查看服务端 warning；不会回退到另一个未经核实的 probe。

## 5. 用户手动启动真机客户端

**下面会进入你原来的真机运行流程，只能由你手动执行。** 在已配置好 ROS/SDK 的
机器人终端中运行；使用该终端原本能正常运行 `eval.py` 的 Python 环境。

```bash
cd /home/kuavo/jiayi/kuavo_learning_studio
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" KUAVO_PROGRESS_MONITOR_TIMESTAMPS=1 python3 kuavo_deploy/eval.py
```

配置文件路径输入：

```text
/home/kuavo/jiayi/kuavo_learning_studio/configs/deploy/deploy_lingbot_package.yaml
```

菜单选择 **2**，沿用原有 package 推理、动作执行和安全逻辑。
环境变量仅为选中的相机帧附加时间戳；服务端不会拿新帧给旧预测重新打标签。

在另一个终端看日志：

```bash
tail -f /media/data/jiayi/outputs/lingbot_progress_pilot/online_monitor/process.jsonl
```

每条 `prediction` 包含：`timestamp`、`observation_id`、原始 `camera_timestamps`
（在 `timing` 中）、`frame_age_ms`、`predicted_process`、`confidence` 和六类 `probabilities`。
帧龄取同一次 observation 最旧相机在客户端的 ROS 帧龄，加上到日志生成时的 wall-clock
耗时；跨机器需要时钟同步。未提供采集元数据或时钟不合理时，帧龄写 `null`，不伪造为零。
`server_observation_age_ms` 仅表示服务端接收后的耗时，不等于相机帧龄。

预测频率等于新 action chunk 的生成频率，**不是相机帧率**：缓存动作执行期间不重复预测。
特征池化和小向量复制在推理线程进行，分类/磁盘日志在有界后台队列执行；会有少量监控开销。

## 关闭监控

停止原服务端后，在第 4 步的终端中用不带监控参数的命令重新启动：

```bash
"$PYTHON_BIN" kuavo_server/launch.py lingbotvla "${PACKAGE_ARGS[@]}"
```

客户端恢复原来的命令即可，不加时间戳环境变量：

```bash
python3 kuavo_deploy/eval.py
```

不需要删除 probe、改 VLA 权重或修改 SafetySupervisor。JSONL 日志追加写入，不覆盖旧记录。
