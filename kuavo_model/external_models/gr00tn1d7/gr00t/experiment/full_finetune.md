CUDA_VISIBLE_DEVICES=1 uv run python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /media/data/training_data/task2_pick_apple_messy_lerobot/lerobot \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path ./kuavo_config.py \
    --num-gpus 1 \
    --output-dir /media/data/jiayi/outputs/grootn17-apple-pick/full_finetune \
    --save-total-limit 5 \
    --save-steps 6000 \
    --max-steps 60000 \
    --global-batch-size 128 \
    --dataloader-num-workers 4


    cd kuavo_model/external_models/gr00tn1d7/scripts/lerobot_conversion
uv run python convert_v3_to_v2.py --repo-id lerobot   --root /media/data/training_data/task2_pick_apple_messy_lerobot/lerobot