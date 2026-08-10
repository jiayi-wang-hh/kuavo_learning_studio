# GR00T N1.7 Package Pick-and-Place 微调配置（AC32）

记录日期：2026-08-06  
代码分支：`gr00tn1d7-visual-lora`  
实验名称：`grootn17-package-pick-and-place-ac32`

## 训练命令

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 uv run python gr00t/experiment/launch_finetune.py --base-model-path nvidia/GR00T-N1.7-3B --dataset-path /root/bayes-tmp/kuavo_dataset/package_pick_and_place_demo_lerobot_jiayi/lerobot --embodiment-tag NEW_EMBODIMENT --modality-config-path ./kuavo_config.py --num-gpus 1 --output-dir /root/bayes-tmp/jiayi/kuavo_learning_studio/outputs/grootn17-package-pick-and-place-ac32 --save-total-limit 10 --save-steps 5000 --max-steps 50000 --global-batch-size 12 --gradient-accumulation-steps 1 --dataloader-num-workers 4 --load-bf16 --gradient-checkpointing --use-diffusion-lora --no-tune-diffusion-model --diffusion-lora-rank 8 --diffusion-lora-alpha 16 --diffusion-lora-dropout 0.05 --use-visual-lora --no-tune-visual --visual-lora-rank 8 --visual-lora-alpha 16 --visual-lora-dropout 0.05 --use-wandb
```

## Action chunk 配置

Action chunk 不由上述命令行参数设置，而由 `kuavo_config.py` 中 action modality 的 `delta_indices` 决定。本实验使用 32 步预测长度：

```python
"action": ModalityConfig(
    delta_indices=list(range(32)),
    # modality_keys 和 action_configs 保持机器人当前配置
),
```

模型内部的 `action_horizon` 上限为 40；32 步有效动作会补齐到 40，并使用 action mask 排除补齐部分的损失。

## 关键参数

| 类别 | 参数 | 值 |
| --- | --- | --- |
| 基础模型 | `base_model_path` | `nvidia/GR00T-N1.7-3B` |
| 数据集 | `dataset_path` | `/root/bayes-tmp/kuavo_dataset/package_pick_and_place_demo_lerobot_jiayi/lerobot` |
| Embodiment | `embodiment_tag` | `NEW_EMBODIMENT` |
| Action chunk | `delta_indices` | `range(32)` |
| 训练步数 | `max_steps` | 50,000 |
| 保存间隔 | `save_steps` | 5,000 |
| checkpoint 保留数 | `save_total_limit` | 10 |
| GPU 数量 | `num_gpus` | 1 |
| GPU | `CUDA_VISIBLE_DEVICES` | 1 |
| 单步 batch | `global_batch_size` | 12（单卡时即 per-device batch） |
| 梯度累积 | `gradient_accumulation_steps` | 1 |
| 有效 batch |  | 12 |
| 数据加载进程 | `dataloader_num_workers` | 4 |
| 学习率 | `learning_rate` | `1e-4`（代码默认值） |
| LR scheduler |  | cosine（代码默认值） |
| Warmup | `warmup_ratio` | 0.05，即 2,500 steps（代码默认值） |
| Weight decay | `weight_decay` | `1e-5`（代码默认值） |
| 计算精度 |  | BF16 |
| Backbone 加载 | `load_bf16` | 开启 |
| Gradient checkpointing |  | 开启 |
| 实验记录 | W&B | 开启 |

## 模块训练状态

| 模块 | 训练状态 |
| --- | --- |
| Qwen3-VL language model | 冻结 |
| 视觉编码器主体 | 冻结 |
| 视觉 attention | Visual LoRA，rank 8、alpha 16、dropout 0.05 |
| 多模态 projector | 全量训练（默认 `tune_projector=true`） |
| Diffusion/DiT 主体 | 冻结 |
| Diffusion attention | Diffusion LoRA，rank 8、alpha 16、dropout 0.05 |

## Checkpoint 计划

预期保存以下 checkpoint，并保留全部 10 个：

```text
checkpoint-5000
checkpoint-10000
checkpoint-15000
checkpoint-20000
checkpoint-25000
checkpoint-30000
checkpoint-35000
checkpoint-40000
checkpoint-45000
checkpoint-50000
```

训练启动后应在日志中确认 Visual LoRA 和 Diffusion LoRA 均成功注入，并确认保存的 processor/modality 配置使用 32-step action horizon。推理时可以预测 32 步，但只执行前若干步后重新观测和规划；执行 horizon 不必等于预测 horizon。
