# GR00T LoRA Training Log

## Start Training With Log

Run this command from:

```bash
kuavo_model/external_models/gr00tn1d7
```

```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /root/bayes-tmp/kuavo_icra_dataset/icra_task1_lerobot_jiayi/lerobot \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path ./kuavo_config.py \
  --num-gpus 1 \
  --output-dir /path/to/output \
  --use-lora \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --no-tune-projector \
  --no-tune-diffusion-model \
  2>&1 | tee logs/gr00t_lora_train_$(date +%Y%m%d_%H%M%S).log
```

## View Log In Real Time

```bash
tail -f logs/gr00t_lora_train_*.log
```

## Run In Background

```bash
mkdir -p logs

nohup bash -c 'CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /root/bayes-tmp/kuavo_icra_dataset/icra_task1_lerobot_jiayi/lerobot \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path ./kuavo_config.py \
  --num-gpus 1 \
  --output-dir /path/to/output \
  --use-lora \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --no-tune-projector \
  --no-tune-diffusion-model' \
  > logs/gr00t_lora_train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## Check Latest Log

```bash
ls -lt logs/gr00t_lora_train_*.log | head
tail -n 100 logs/gr00t_lora_train_*.log
```

