# V2 Lightweight Visual Critic：仿真 Stage-1 实现说明

## 目标与边界

本次新增了一个单帧、异步的 Visual Critic，作为仿真测试的 Stage-1。它只回答“何时暂停”，输出 `PROGRESSING / STALLED / FAILURE / SUCCESS / UNKNOWN`；原有 Qwen3.5 Stage-2 继续负责失败类型判断与 `RESUME / RESET_AND_RETRY / STOP` 恢复决策。

没有修改任何已有文件。原有 Qwen 视频 trigger、仿真环境、VLA policy、Stage-2 verifier 和恢复流程均保持不变。

## 新增文件

- `kuavo_deploy/src/eval/failure_trigger/base_trigger.py`：通用 `FailureTrigger` 接口，以及 `CriticInput`、`CriticResult`。
- `kuavo_deploy/src/eval/failure_trigger/visual_critic_trigger.py`：严格 JSON 解析、Florence 模型隔离层、异步 latest-frame 调度、过期丢弃、触发判定。
- `kuavo_deploy/src/eval/sim_auto_test_visual_critic.py`：与现有 agentic episode runner 和 Stage-2 的兼容适配器。
- `kuavo_deploy/src/scripts/script_auto_test_visual_critic.py`：新的命令行入口。
- `kuavo_deploy/tests/test_visual_critic_trigger.py`：不加载模型的逻辑测试。

## Stage-1 数据流

```text
head camera 最新帧 + 当前 subtask
              │
              ▼
    FlorenceVisualCriticModel
              │ structured JSON
              ▼
      VisualCriticTrigger
  single in-flight + newest pending
              │
              ▼
        TriggerDecision
 FAILURE(M/H) ───────────────► 立即 PAUSE
 STALLED(M/H) × N ───────────► PAUSE
 stale / UNKNOWN / SUCCESS ──► CONTINUE
```

模型后端被隔离在 `FlorenceVisualCriticModel` 中。以后将 `model_name_or_path` 指向微调 checkpoint，或者注入实现相同 `infer(frame, subtask)` 接口的模型，不需要改调度器和仿真循环。

## 异步与时序保证

- 控制循环只提交帧和非阻塞轮询结果，不等待推理。
- 同时只运行一次推理；运行期间到达的帧只保留最新一帧，中间帧不排队。
- 每个结果携带 `source_step`、源时间、推理耗时和消费时 age。
- 超过 `max_result_age_s` 的结果标记为 stale；即使它是高置信度 `FAILURE` 也不会暂停。
- `STALLED` 默认连续 3 次中高置信度才触发；`FAILURE` 默认中高置信度立即触发。
- 解析失败或模型异常产生 `UNKNOWN/LOW`，不会通过自然语言关键词猜测失败。
- V2 路径不存在 `PREGRASP_NO_COMPLETED_ATTEMPT` 覆盖规则。

## Stage-2 配合方式

适配器把 V2 结果转换成现有 Stage-2 接受的边界对象，并在真正触发时把近期滚动帧保存成短视频。Stage-2 收到：

- critic state（放在现有 `current_phase` 字段）；
- confidence；
- reason / visible evidence；
- source step；
- 触发前短视频。

随后完全复用现有 `VLMFailureVerifier.verify()` 及恢复流程。Visual Critic 不产生 failure type 或 recovery action。

## 运行示例

```bash
python -m kuavo_deploy.src.scripts.script_auto_test_visual_critic \
  --config PATH_TO_EVAL_CONFIG \
  --critic-model microsoft/Florence-2-base \
  --critic-device cuda:0 \
  --critic-frequency-hz 5 \
  --critic-max-result-age-s 1.5 \
  --critic-stall-confirm-count 3 \
  --stage2-vlm-model-path /media/data/jiayi/hf_model/Qwen3.5-9B
```

先用 `--dry-run` 检查配置。Florence 权重需要预先存在于本机 Hugging Face cache，或把 `--critic-model` 指向本地目录。实际设备不支持 BF16 时使用 `--critic-dtype float16` 或 `float32`。

## 验证与调参

逻辑测试：

```bash
python -m unittest kuavo_deploy.tests.test_visual_critic_trigger
```

建议先分别测试 2、5、10 Hz，并根据 `[V2_CRITIC]` 日志中的 `inference` 和 `age` 调整频率与 `max_result_age_s`。若推理时延长期大于 stale 阈值，结果会被正确丢弃，但 Stage-1 也不会产生有效触发。

注意：通用 Florence-2-base 对任意 JSON 指令的遵循能力需要在目标机器与任务图像上实测。本实现首先保证软件架构和 Stage-2 闭环；若零样本结构化输出质量不足，应使用任务微调 checkpoint 或替换模型 wrapper，而不是添加文本关键词 guard。
