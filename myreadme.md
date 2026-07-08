uv run --project /home/yuan/yuan_meng_kuavo/jiayi/kuavo_learning_studio/kuavo_model/external_models/gr00tn1d7 \
python kuavo_server/serve.py \
  --adapter isaac_gr00t_n17 \
  --checkpoint /home/yuan/yuan_meng_kuavo/jiayi/kuavo_learning_studio/outputs/gr00t_lora_action_only/checkpoint-50000\
  --which_arm both \
  --embodiment_tag NEW_EMBODIMENT