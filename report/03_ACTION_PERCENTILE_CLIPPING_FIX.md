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

## 训练配置已改为 false，但 checkpoint 仍显示 true

排查 Kuavo 数据时还发现了一个容易误导诊断的配置同步问题：即使
`FinetuneConfig.use_percentiles` 已设为 `false`，旧版训练流程保存出的 checkpoint 根目录
`config.json` 仍可能显示：

```json
"use_percentiles": true
```

原因是 `launch_finetune.py` 修改的是训练 pipeline 的 `config.model`，而
`AutoModel.from_pretrained()` 创建的 Hugging Face 模型会保留 base checkpoint 中的
`model.config.use_percentiles=true`。Trainer 保存模型时序列化的是后者。训练 processor
与模型配置因此可能出现互相矛盾的记录。

本次修复在模型创建后显式同步：

```python
model.config.use_percentiles = self.model_config.use_percentiles
```

同时在 processor 创建后检查二者是否一致；不一致时立即终止训练，避免生成归一化语义
不明确的 checkpoint。训练入口也会打印最终解析值。Tyro 的布尔参数
`--use-percentiles` 表示启用该选项；使用默认的 min/max 时必须省略该参数，不能写成
`--use-percentiles false`。

### 三处配置都必须一致

新训练启动后应检查：

1. 控制台打印 `config.model.use_percentiles=False`；
2. `<output_dir>/processor/processor_config.json` 中为 `false`；
3. 新 checkpoint 根目录 `config.json` 中也为 `false`。

训练代码会调用 `trainer.train(resume_from_checkpoint=True)`。因此切换归一化方式时必须使用
全新的输出目录，并从原始 base model 开始；复用旧输出目录会自动恢复已有 checkpoint，
而从旧 post-training checkpoint 开始也会继承已经在 percentile clipping 下学习的权重。

### 本次数据诊断证据

在 `task2_pick_apple_messy_lerobot_jiayi` 数据中，open-loop 预测平台与 action percentile
边界逐维吻合。例如：

```text
Action 6:  prediction ~= 0.25, q01 = 0.2485567521
Action 15: prediction ~= 0.52, q99 = 0.5206471342
```

这说明该结果实际使用或继承了 percentile normalization。另需注意，state 与 action 的
夹爪统计不同：`state.left_gripper.max=0.46`，而 `action.left_gripper.max=1.0`；诊断时不能
把 state 统计误当作 action 统计。

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

## 纠错记录：`use_percentiles=false` 之后仍出现 percentile 平台

本节按实际排查顺序保留错误判断、补充证据和纠正结果，便于以后遇到相似现象时回顾。

### 第一次判断：误认为图来自 LingBot-VLA

仅根据 open-loop 曲线的图例和排版，最初把结果误认为 LingBot-VLA，并尝试用
`subtract_state` 和 delta action 解释 Action 6 的直线。用户确认实际模型为 GR00T 后，检查
`gr00t/eval/open_loop_eval.py`，确认两套评估脚本的绘图样式相似，之前的判断不适用。

纠正原则：不能根据图的外观判断模型来源；必须先核对实际执行脚本、checkpoint 路径和
action modality 顺序。

### 第二次判断：认为 Action 6 是 min/max 中点塌缩

得知用户声称结果使用 `use_percentiles=false` 后，一度推测模型在 normalized action 空间
输出约 `0`，经 min/max 反归一化后落在 `(min + max) / 2`，因而形成水平线。

复制真实数据后，该解释被否定。数据集 `action[6]` 的统计为：

```text
min = -0.2414417416
max =  0.5934119225
min/max midpoint = 0.1759850904
q01 = 0.2485567521
```

图中 Action 6 的预测平台约为 `0.25`，明显不等于 min/max 中点，却几乎精确等于 `q01`。

### 第三步：逐维比对平台与 q01/q99

继续检查后发现多个预测平台同时落在 percentile 边界：

```text
Action 0:  prediction ~= -0.10, q99 = -0.0905339289
Action 2:  prediction ~= -0.067, q99 = -0.0669989070
Action 3:  prediction ~= -1.90, q01 = -1.9056813223
Action 6:  prediction ~=  0.25, q01 =  0.2485567521
Action 15: prediction ~=  0.52, q99 =  0.5206471342
```

多个维度同时吻合排除了偶然的均值或中点塌缩，说明该结果实际使用或继承了 percentile
normalization。

### 第四步：纠正 state/action statistics 混淆

排查中曾收到以下 `left_gripper` 统计：

```text
min=0, max=0.46, mean=0.0631593, std=0.1424558
```

复制数据后确认，这组数值属于 `state.left_gripper`，不是 `action.left_gripper`。真实 action
统计为：

```text
action.left_gripper: min=0, max=1, mean=0.1970038, std=0.3970192
```

纠正原则：检查 `statistics.json` 时必须同时确认外层 modality 和 joint key，不能只根据
`left_gripper` 名称判断统计对象。

### 第五步：发现模型配置和 processor 配置可能不同步

`FinetuneConfig.use_percentiles` 和 `launch_finetune.py` 中的 pipeline 配置可以已经是
`false`，但 `AutoModel.from_pretrained()` 创建的 Hugging Face 模型仍会继承 base checkpoint
的 `model.config.use_percentiles=true`。Trainer 保存根目录 `config.json` 时序列化的是模型
自身的 config，因此它可能与训练 processor 的配置不一致。

另一方面，真正执行 state/action 归一化的是 `processor/processor_config.json`。本次用户
检查后确认该文件也为 `true`，说明不能只把根目录 `config.json=true` 当作无害的保存元数据；
必须继续检查命令行参数、自动 resume 和实际运行的代码路径。

### 最终解决方案与防复发措施

本次代码修复包含：

1. 训练启动时打印 `ft_config.use_percentiles` 与最终 `config.model.use_percentiles`；
2. 启用 percentile 时明确警告：Tyro 的 `--use-percentiles` 本身就表示 `true`，使用 min/max
   时应省略该参数，不能写 `--use-percentiles false`；
3. 模型创建后执行
   `model.config.use_percentiles = self.model_config.use_percentiles`，保证 checkpoint 根目录
   `config.json` 与训练配置一致；
4. processor 创建后校验 `processor.use_percentiles == model_config.use_percentiles`，不一致时
   立即终止训练。

重新训练时还必须使用全新的输出目录，并从原始 base model 开始。当前训练代码调用
`trainer.train(resume_from_checkpoint=True)`；复用已有输出目录会自动恢复旧 checkpoint，
从旧 post-training checkpoint 开始也会继承 percentile clipping 下已经学习的权重。

新训练启动后应同时确认：

```text
控制台: config.model.use_percentiles=False
processor/processor_config.json: use_percentiles=false
checkpoint/config.json: use_percentiles=false
```

### 第六步：保护检查暴露 `from_pretrained()` 丢弃 override

加入 processor/model 一致性检查后，新训练在启动阶段报错：

```text
Normalization configuration mismatch:
processor.use_percentiles=True, model_config.use_percentiles=False
```

这说明前面的“配置可能不同步”还不是完整根因。继续检查
`Gr00tN1d7Processor.from_pretrained()` 后发现，`setup.py` 虽然明确传入：

```python
AutoProcessor.from_pretrained(..., use_percentiles=False)
```

但 `from_pretrained()` 只会应用 `override_keys` 白名单中的参数，而该列表遗漏了
`use_percentiles`。因此传入的 `False` 被静默丢弃，processor 继续采用 base checkpoint
保存的 `True`。这也解释了为什么 `FinetuneConfig` 和 pipeline 都显示 false，最终生成的
`processor/processor_config.json` 却仍为 true。

最终补充修复是在 processor 的 override 白名单中加入：

```python
"use_percentiles",
```

并增加回归测试，确认 `Gr00tN1d7Processor.from_pretrained(..., use_percentiles=...)` 能覆盖
checkpoint 内的旧值。之前加入的一致性检查应继续保留：它不是错误来源，而是成功阻止了
训练在错误 normalization 配置下继续运行。
