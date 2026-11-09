# Package Sorting 自动技能 Demo

> 当前完整操作流程见 [VLM + VLA Pipeline](../../docs/PACKAGE_SORTING_VLM_VLA_PIPELINE.md)，已核对至2026-11-04，包括右臂入口与常驻服务。当前默认 VLA GPU为1，Recovery checkpoint为global_step_39620；参数以配置和完整文档为准。

Normal 与 Recovery 共用 `configs/deploy/deploy_lingbot_package.yaml` 的相机、机器人环境和控制流程。技能配置位于 `configs/server/lingbot_package_skills.yaml`，分别指定 checkpoint、对应训练 YAML 和各自的 norm stats。切换不会调用 `env.reset()`，不修改权重、动作或 SafetySupervisor。

## 启动

当前默认复用独立 Qwen 常驻服务。先在单独终端、仓库根目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 /home/kuavo/miniforge3/envs/qwen35/bin/python -m vlm_test.qwen_service
```

看到 `Qwen ready` 后保持该终端开启。回车保持模型加载，输入 `q` / `quit` / `exit` 或 Ctrl+C 关闭服务、释放显存。关闭 Demo 不会关闭该服务，也不会关闭它的终端。服务仅监听本机 127.0.0.1:5560，不连接 ROS 或机器人。每次请求独立输入图像与 Prompt，不保存对话历史，不排队积压帧；原有 Monitor 继续检查帧时间戳和输出时效。Qwen 关闭或推理失败不默认判为 normal。

配置 `configs/deploy/lingbot_skill_demo.yaml` 中 `qwen_service_url` 为空字符串时恢复每次运行单独加载；当前地址为 `http://127.0.0.1:5560`。独立使用 Passive Monitor 时可加 `--qwen-service-url http://127.0.0.1:5560` 复用服务。

再在运行 VLA server 的环境中单独加载两个 checkpoint：

```bash
python -m vlm_test.skill_service --gpu 1
```

GPU 参数选择 VLA 所在的物理卡；Qwen 所在卡仍由其独立终端的 `CUDA_VISIBLE_DEVICES` 决定，当前示例 Qwen 用 GPU 0。等待两个 `server_ready` 和 `Skills ready`，保持这个终端，再启动 `eval.py`。回车保持两个 checkpoint；`q` 或 Ctrl+C 关闭。关闭前先结束 Demo。此脚本只启动原有模型服务和查询 metadata，不连接 ROS，不发送机器人动作。

加载日志位于 `outputs/skill_servers/<UTC时间>/normal_server.log` 和 `recovery_server.log`。`outputs/skill_servers/current.json` 保存本地连接信息和随机令牌，请勿分享；服务退出时移除。Demo 检查 checkpoint/config/norm 匹配，重置缓存后开始；退出时清缓存但不关闭常驻服务，下次运行不重载权重。一个常驻服务只允许一个 Demo 客户端占用。

默认 `skill_service_manifest: outputs/skill_servers/current.json`。设为空字符串恢复原来的自动启动/关闭 VLA 服务模式。原有服务端口 5557/5558 不变；先结束旧 Demo，避免端口冲突。服务关闭或失效时 Demo 不会偷偷重新加载权重。

在原来运行 real_single_test 的环境、完成原有 ROS 配置后，从仓库根目录运行：

```bash
python kuavo_deploy/eval.py
```

输入配置路径：

```text
/home/kuavo/jiayi/kuavo_learning_studio/configs/deploy/deploy_lingbot_package.yaml
```

菜单 `10` 只检查配置与文件，不启动 ROS、模型或机器人；菜单 `9` 启动自动 Demo，无人工确认步骤。启动前结束其他机器人推理客户端，避免多个客户端控制同一机器人。旧模型服务若仍占用显存，需要自行结束。

也可直接检查或执行：

```bash
python kuavo_deploy/src/scripts/script_skill_demo.py --config configs/deploy/deploy_lingbot_package.yaml --preflight
python kuavo_deploy/src/scripts/script_skill_demo.py --config configs/deploy/deploy_lingbot_package.yaml --execute
```

当前默认连接上述常驻 VLA 与 Qwen 服务，只自动启动每次会话的 Passive Monitor。禁用常驻模式时，执行入口才自动启动两个现有 LingBot 模型服务，使用 `lingbotvla` conda 环境。`vla_gpu` 控制自动加载模式的 VLA；常驻 VLA 使用 `--gpu`，常驻 Qwen 的 GPU 由启动服务时的 `CUDA_VISIBLE_DEVICES` 指定，`vlm_gpu` 仅在逐次加载模式控制 Qwen GPU。

## 自动切换规则

1. 每秒单帧判断。连续 **3 个不同且新鲜的输入帧** 为 `recovery` 才触发。1 Hz 时从第一次命中到第三次约 2 秒，另加推理、正在执行的一步和停止稳定时间。重复读取日志不会增加票数。
2. `normal`、无效输出或被拒绝的过期结果打断 Recovery 计数；相邻有效帧间隔超过 2.5 秒重新计数。触发时最后结果也必须新鲜。
3. 在同步动作边界停止提交 Normal 动作，清空 Normal 服务缓存的 action chunk。若触发在推理期间到达，丢弃该次尚未提交的动作。根据关节反馈等待稳定后，清空旧票数、只收集停稳后拍摄的新帧，记录 `VERIFY_SWITCH`。连续 3 次新鲜 `recovery` 才清空 Recovery 缓存并调用 Recovery；连续 3 次新鲜 `normal` 则取消恢复、清空 Normal 缓存后继续 Normal，记录 `RECOVERY_CANCELLED_NORMAL_RESUMED`。单次跳变不退出，也不切换；不稳定/缺失结果最多等 `verify_timeout`（默认 60 秒），期间不提交任何动作，超时才退出。取消恢复不占 Recovery 次数。
4. 每次 Recovery 独立等待左手的闭合再释放：本次已提交的左手命令 ≥0.5 且反馈 ≥0.25 持续至少 0.3 秒，记录 `grasp_observed`；之后命令和反馈均 ≤0.15 持续至少 0.5 秒，记录 `released`。这两个位置阈值是可配置的初始动作阶段判据，尚未实机标定，不能证明物体接触或抓取成功。初始张手不算释放；短暂闭合、缺失或间隔过长的采样不能累积稳定时间。
5. `released` 后清空之前的 VLM 票数，仅接收释放后拍摄的 3 次新鲜 `normal`（且 Recovery 已执行至少 2 秒），才停止 Recovery 并清空缓存。闭合前、搬运中的 `normal` 不会提前结束 Recovery。再次闭合会撤销释放状态。180 秒仍未完成则按原超时保护退出，不强行判成功。阶段变化记录在 `decisions.jsonl` 的 `recovery_gripper_stage`。
6. 停稳后重新收集 **3 次新的 `normal`**，同时检查左手张开，才清空 Normal 缓存并恢复 Normal。手指位置是释放的代理条件，不能证明物体已完全脱离手部。

初始环境只 reset 一次；初始化后重新检查场景，若已满足 Recovery 条件，会先进入 Recovery。运行中不要求人工确认。原 Passive Monitor 仍然只观察；模型动作由 Demo 调用原有 `env.step()` 发出；当前启用的右臂预定位经原有 `env.safe_control_arm()` 发送受检关节命令，入口细节见完整文档。

## 时效与退出

所有头部、双腕图像及关节、手部反馈必须有有效同域时间戳，默认最新数据年龄不超过 0.5 秒。VLM 结果最大年龄 1.5 秒；连续超过 3 秒没有有效判断则暂停提交新动作并清空当前技能缓存，记录 `WAITING_FOR_VLM`。最多等待 15 秒，连续 3 次新鲜有效的 normal 或 recovery 后记录 `VLM_AVAILABLE`，重新判断技能并推理；暂停前已经算好的动作被丢弃。超时仍无稳定结果才退出，不将不可用标记为 normal。已经提交给底层的目标可能仍在执行，这不是硬件急停。

默认停稳窗口 0.5 秒、超时 8 秒；Recovery 超时 180 秒，停止后的复核超时 60 秒；最多 3 次 Recovery，总运行超时 600 秒。Normal 动作数沿用部署配置的 `max_episode_steps`，Recovery 不占 Normal 步数。达到步数上限只表示本次 Demo 结束，不代表任务成功。上述参数位于 `configs/deploy/lingbot_skill_demo.yaml`。

模型/RPC 错误、反馈过期、监控持续失联且等待恢复超时、停稳失败、复核失败或原 SafetySupervisor 拒绝动作均退出，不自动放宽限制。菜单原有 `p` 暂停、`s` 停止及 Ctrl+C 可用。停止意味着不再提交新动作，不替代机器人硬件急停，已经提交的目标可能仍会执行。仅清理本次 Demo 启动的子进程。

## 日志与验证

每次运行记录在 `outputs/skill_demo/<UTC时间>/`：

- `settings.json`：技能与阈值配置。
- `decisions.jsonl`：逐次 VLM 投票、状态切换、动作提交、故障和结束记录。
- `vlm_monitor.log`：本次 Monitor 日志。常驻 VLA 的 `normal_server.log`、`recovery_server.log` 位于 `outputs/skill_servers/<UTC时间>/`；禁用常驻模式时才在本次 Demo 目录生成。
- `monitor/<会话>/events.jsonl` 及 `images/`：每次判断的原始输出、解析、时间戳、延迟与输入图像，沿用 Passive Monitor 格式。

`eval.py` 的合并执行日志仍在 `log/kuavo_deploy/kuavo_deploy.log`。

先用菜单 10 检查文件与配置；运行后对照输入图像检查连续投票及 `STOPPING_NORMAL → RECOVERING → VERIFY_RECOVERY → NORMAL_RESUMED` 事件，确认切换间没有旧技能动作提交，并核对失败/停止后没有新动作提交。轻微前缘超出仍按 v4 定义为 normal。

开发检查与操作者实机实验应分别看待：纯逻辑单元测试、preflight不证明任务成功；历史实机会话有实际切换与多种退出记录，具体结果以对应会话日志/图像为准。

传图与输入存档使用 PNG 无压缩模式，以减少 CPU 压缩耗时，像素和分辨率不变，图片文件会变大。时效仍为输入 0.5 秒、结果 1.5 秒；未通过时效检查的结果始终拒绝。修改后的 Monitor/客户端在下一次 Demo 生效，独立 Qwen 服务无需重载。
