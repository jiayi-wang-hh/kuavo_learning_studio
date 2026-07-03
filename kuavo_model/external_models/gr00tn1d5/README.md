# Isaac GR00T N1.5 native integration

This directory hosts the Learning Studio integration for NVIDIA's native
`Isaac-GR00T` N1.5 runtime. The upstream source is intentionally fetched at
setup time instead of being copied into this repository.

The integration is pinned to the signed `n1.5-release` commit:

- repository: `https://github.com/NVIDIA/Isaac-GR00T`
- tag: `n1.5-release`
- commit: `4af2b62`
- base checkpoint: `nvidia/GR00T-N1.5-3B`

## Bootstrap

From PowerShell:

```powershell
.\kuavo_model\external_models\gr00tn1d5\bootstrap.ps1 -Install
```

The source is checked out under `source/`. That directory is ignored by the
Learning Studio repository and remains an independent upstream checkout.

## Planned Learning Studio surface

1. Native N1.5 fine-tuning through upstream `scripts/gr00t_finetune.py`.
2. A Kuavo data config and modality transform.
3. `isaac_gr00t_n15` inference adapter for the unified model server.
4. LoRA controls preserved from N1.5 (`lora_rank`, `lora_alpha`,
   `lora_dropout`, and `lora_full_model`).

The existing `configs/train/lerobot/gr00t.yaml` remains the LeRobot wrapper;
this directory is the separate native NVIDIA path.
