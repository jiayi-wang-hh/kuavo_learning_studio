# Linear Probe 全量训练流程

在同一个终端按顺序执行。全量使用 **141 个有标签 episode、168171 个有效帧**：
**99 个训练 / 21 个验证 / 21 个测试**，保持原划分。
原来的 `pilot_v1` 只有 16 个 episode，不能用于本次全量训练；先执行第 2 步提取全量特征。

## 1. 设置路径

```bash
cd /home/kuavo/jiayi/kuavo_learning_studio

PYTHON_BIN=/home/kuavo/miniforge3/envs/lingbotvla/bin/python
OUT=/media/data/jiayi/outputs/lingbot_progress_pilot
LABELS=/media/data/jiayi/dataset/progress_training_v1_5
```

## 2. 提取全部 141 个 episode 的特征（GPU）

```bash
"$PYTHON_BIN" -B -m progress_monitor.select_pilot \
  --labels "$LABELS" --scope full \
  --output "$OUT/full_selection.json"

"$PYTHON_BIN" -u -B progress_monitor/extract_lingbot_features.py \
  --checkpoint /media/data/jiayi/outputs/kuavo_lingbotvla/task3_pacakge_lora/global_step_66480/hf_ckpt \
  --dataset /media/data/training_data/task3_package_pick_place_pack_flip_lerobot/lerobot \
  --labels "$LABELS" \
  --qwen /media/data/jiayi/hf_model/Qwen2.5-VL-3B-Instruct \
  --output "$OUT" --run-name full_141 \
  --pooling all vision language \
  --batch-size 8 --cpu-threads 4
```

不要加 `--selection`、`--episodes` 或 `--smoke`，这样才会提取全部 141 个 episode。
特征保存在 `$OUT/full_141/{train,val,test}/`，总提取帧数应为 **168171**。
如果显存不足，把 `--batch-size 8` 改为 `4` 或 `1`。

## 3. 开始全量训练（CPU）

一次训练三种表示 × 两种损失，共 6 个 Linear Probe。
使用全部 99 个训练 episode（116070 帧），由 21 个验证 episode（27035 帧）选择最佳模型。

```bash
"$PYTHON_BIN" -u -B progress_monitor/train_linear_probe.py \
  --features "$OUT/full_141" \
  --selection "$OUT/full_selection.json" \
  --labels "$LABELS" --scope full \
  --pooling all vision language --weighting both \
  --epochs 50 --patience 8 --seed 42 --device cpu \
  --output "$OUT/full_probes"
```

自动执行训练集归一化、AdamW、验证集早停。最佳模型保存在
`$OUT/full_probes/<pooling>_<weighting>/best.pt`，训练报告为 `$OUT/full_probes/report.md`。

## 4. 比较验证集结果

```bash
"$PYTHON_BIN" -B -m progress_monitor.evaluate_linear_probe \
  --features "$OUT/full_141" \
  --selection "$OUT/full_selection.json" \
  --labels "$LABELS" --scope full --split val \
  --checkpoints "$OUT"/full_probes/*/best.pt \
  --plot --output "$OUT/full_validation"

cat "$OUT/full_validation/comparison.md"
```

结果包含准确率、Macro-F1、各类别指标、P2a/P3 混淆和混淆矩阵 CSV/PNG。
21 个测试 episode（25066 帧）保持独立，选定模型后再做最终测试，不参与训练或选型。

已有文件不会被覆盖；新实验可修改第 1 步的 `OUT`，已完成的步骤不必重复执行。
