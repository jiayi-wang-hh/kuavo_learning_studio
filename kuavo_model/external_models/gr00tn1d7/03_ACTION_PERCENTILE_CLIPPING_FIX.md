# Action Percentile Clipping：问题记录与解决方案

## 问题摘要

Kuavo apple-pick 模型在 open-loop evaluation 中出现部分 action 维度长期保持常数、动作幅值被压缩的问题。典型现象包括：

- `right_gripper` 的 Ground Truth 从 `0` 切换到 `1`，预测最大只能到约 `0.52`；
- 部分机械臂关节的 Ground Truth 超出一个较窄区间后，预测停留在固定平台；
- 调整 checkpoint 的 `statistics.json` 后，夹爪输出上限随 q99 一起变化。

问题类型：**action percentile normalization clipping（动作百分位归一化截幅）**。

## 根因

GR00T N1.7 模型配置原先默认启用：

```text
use_percentiles = true
clip_outliers = true
```

启用后，state/action 使用 `q01` 和 `q99` 作为归一化边界；落在边界外的值被裁剪到 `[-1, 1]`。这适合排除真正的异常值，但对低频且合法的任务动作并不安全。

apple-pick checkpoint 的 `new_embodiment.action` 中存在如下统计：

```text
right_gripper: min=0, max=1, q01≈0, q99≈0.520647
```

因此真实的 `right_gripper=1` 在训练时被裁剪到归一化上界；推理反归一化时，上界又只能恢复成约 `0.520647`。open-loop 图中的预测平台与 q99 数值吻合。

机械臂也存在同类问题。例如某些维度的合法动作范围明显大于 q01/q99：

```text
left_arm[0]:  min≈-0.961, max≈0.448, q01≈-0.355, q99≈-0.091
right_arm[4]: min≈-0.658, max≈1.389, q01≈-0.071, q99≈0.421
```

这会把多个不同的合法动作映射到同一个边界值，模型无法从训练数据中恢复被裁掉的差异。

## 本分支的修复

本分支将 `launch_finetune.py` 的 fine-tuning 默认行为改为使用 `min/max` 归一化：

```text
FinetuneConfig.use_percentiles = false
```

`launch_finetune.py` 会显式把该配置传给：

```text
config.model.use_percentiles
```

这样可以保证新生成的 processor/checkpoint 在训练和推理时使用同一套 min/max 规则，避免合法但低频的动作被 q01/q99 截幅。

百分位归一化功能仍然保留。如果数据已经验证适合 q01/q99，可以显式开启：

```bash
--use-percentiles
```

启用前必须逐维检查 q01/q99，确认其覆盖任务的有效动作空间。

## 训练方式

现有 Kuavo fine-tuning 命令无需新增参数。省略 `--use-percentiles` 即使用修复后的 min/max 默认值：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path <APPLE_PICK_DATASET> \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path ./kuavo_config.py \
  --output-dir <NEW_OUTPUT_DIR> \
  <OTHER_OPTIONS>
```

不要复用旧实验的输出目录，以免混入使用 percentile normalization 的 checkpoint。

## 旧 checkpoint 的处理

旧 checkpoint 在错误的 q01/q99 边界下训练。只编辑其 `statistics.json` 可以修正部分反归一化幅值，例如把右夹爪的输出从约 `0.52` 恢复到 `1.0`，但不能恢复训练阶段已经被裁掉的动作差异。

因此：

- 修改旧 checkpoint statistics 仅适合作为问题诊断；
- 不建议把这种临时修改后的 checkpoint 直接用于真机；
- 完整修复需要使用本分支从头 fine-tune；
- 不应从旧的 post-training checkpoint 恢复训练，应从原始 base model 开始。

## 验证步骤

### 1. 检查新 checkpoint 配置

确认 checkpoint 的 `processor_config.json` 包含：

```json
"use_percentiles": false
```

同时保留正确的 `statistics.json`，因为 min/max、mean/std 和维度信息仍用于 processor。

### 2. 固定 episode 运行 open-loop evaluation

```bash
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path <APPLE_PICK_DATASET> \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path <NEW_CHECKPOINT> \
  --traj-ids 0 \
  --steps 140 \
  --action-horizon 16
```

重点检查：

- 夹爪预测能覆盖完整的 `0 -> 1` 范围；
- 机械臂预测不再停留在旧 q01/q99 对应的平台；
- 每个 action 维度的预测曲线能够随 observation 更新；
- 训练集和独立验证集均进行评估。

### 3. 对比多个 checkpoint

固定相同的 dataset、trajectory IDs、steps 和 action horizon，对不同训练步数的 checkpoint 比较 per-dimension 曲线、MSE 和 MAE。最终仍需通过仿真或低速真机 closed-loop evaluation 验证任务成功率和安全性。

## 回归测试

测试覆盖以下行为：

- fine-tuning 默认使用 `use_percentiles=False`；
- 如有需要，仍可以显式设置 `use_percentiles=True`。

运行：

```bash
uv run pytest tests/gr00t/model/test_model_forward.py -k FinetuneNormalizationConfig
```
