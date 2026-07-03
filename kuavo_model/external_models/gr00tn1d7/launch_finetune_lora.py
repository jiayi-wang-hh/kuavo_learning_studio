"""Hook-injected PEFT LoRA fine-tuning for GR00T N1.7 in Learning Studio.

Adapted from https://github.com/jinnymo/gr00t-n17-lora. The base checkpoint is
loaded first, then the pipeline model is wrapped by PEFT. Every Trainer
checkpoint additionally gets an ``adapter_only/`` directory.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tyro
from peft import LoraConfig, get_peft_model

GR00T_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(GR00T_ROOT))

from gr00t.configs.base_config import get_default_config  # noqa: E402
from gr00t.configs.finetune_config import FinetuneConfig  # noqa: E402
from gr00t.experiment.experiment import run  # noqa: E402
from gr00t.experiment.trainer import Gr00tTrainer  # noqa: E402
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline  # noqa: E402


@dataclass
class LearningStudioLoraConfig(FinetuneConfig):
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_action_head_only: bool = False
    lora_include_mlp: bool = False
    lora_modules_to_save_action_head: bool = False
    logging_steps: int = 10
    """Write Trainer metrics (including loss and learning rate) every N steps."""

    log_file: str = "train.log"
    """Log filename created inside the resolved training output directory."""


_RUNTIME: dict[str, Any] = {}
_ORIGINAL_CREATE_MODEL = Gr00tN1d7Pipeline._create_model
_ORIGINAL_SAVE_MODEL = Gr00tTrainer.save_model


class TeeStream:
    """Mirror a console stream to a persistent UTF-8 training log."""

    def __init__(self, console, log):
        self.console = console
        self.log = log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def isatty(self):
        return self.console.isatty()

    def fileno(self):
        return self.console.fileno()


def discover_lora_targets(
    model: torch.nn.Module,
    *,
    action_head_only: bool = False,
    include_mlp: bool = False,
) -> list[str]:
    patterns = ["q_proj", "k_proj", "v_proj", "o_proj", "to_q", "to_k", "to_v", "to_out.0"]
    if include_mlp:
        patterns += [
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "ff.net.0.proj",
            "ff.net.2",
        ]
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and (not action_head_only or "action_head" in name)
        and any(pattern in name for pattern in patterns)
    ]


def inject_peft_lora(
    model: torch.nn.Module,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    action_head_only: bool = False,
    include_mlp: bool = False,
    modules_to_save_action_head: bool = False,
) -> torch.nn.Module:
    targets = discover_lora_targets(
        model, action_head_only=action_head_only, include_mlp=include_mlp
    )
    if not targets:
        raise ValueError(
            "No LoRA target modules matched "
            f"(action_head_only={action_head_only}, include_mlp={include_mlp})"
        )
    modules_to_save = None
    if modules_to_save_action_head:
        modules_to_save = [
            "state_encoder",
            "action_encoder",
            "action_decoder",
            "position_embedding",
            "vlln",
            "vl_self_attention",
        ]
    print(f"[learningstudio-lora] injecting {len(targets)} layers (r={rank}, alpha={alpha})")
    wrapped = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=targets,
            modules_to_save=modules_to_save,
            lora_dropout=dropout,
            bias="none",
            task_type=None,
        ),
    )
    wrapped.print_trainable_parameters()
    return wrapped


def _create_model_with_lora(self):
    model = _ORIGINAL_CREATE_MODEL(self)
    if _RUNTIME["rank"] <= 0:
        return model
    return inject_peft_lora(model, **_RUNTIME)


def _find_peft_model(trainer):
    for candidate in (getattr(trainer, "model_wrapped", None), trainer.model):
        if candidate is not None:
            candidate = getattr(candidate, "module", candidate)
            if hasattr(candidate, "peft_config"):
                return candidate
    return None


def _save_model_with_adapter(self, output_dir=None, _internal_call=False):
    _ORIGINAL_SAVE_MODEL(self, output_dir, _internal_call)
    model = _find_peft_model(self)
    if model is not None:
        adapter_dir = Path(output_dir or self.args.output_dir) / "adapter_only"
        model.save_pretrained(adapter_dir)
        print(f"[learningstudio-lora] saved {adapter_dir}")


def install_lora_hooks() -> None:
    Gr00tN1d7Pipeline._create_model = _create_model_with_lora
    Gr00tTrainer.save_model = _save_model_with_adapter


def _load_modality_config(value: str) -> None:
    import importlib.util

    path = Path(value).resolve()
    if not path.is_file() or path.suffix != ".py":
        raise FileNotFoundError(f"Invalid modality config: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import modality config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def build_training_config(ft: LearningStudioLoraConfig):
    from gr00t.data.embodiment_tags import EmbodimentTag

    embodiment = EmbodimentTag.resolve(ft.embodiment_tag)
    if ft.modality_config_path:
        _load_modality_config(ft.modality_config_path)
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [ft.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment.value,
                    }
                ],
            }
        }
    )
    config.load_config_path = None
    # Do not combine this whole-model PEFT wrapper with the older in-model path.
    config.model.use_lora = False
    for name in (
        "tune_llm",
        "tune_visual",
        "tune_projector",
        "tune_diffusion_model",
        "state_dropout_prob",
        "random_rotation_angle",
        "color_jitter_params",
    ):
        setattr(config.model, name, getattr(ft, name))
    config.model.extra_augmentation_config = (
        json.loads(ft.extra_augmentation_config) if ft.extra_augmentation_config else None
    )
    config.model.load_bf16 = True
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.start_from_checkpoint = ft.base_model_path
    for name in (
        "experiment_name",
        "global_batch_size",
        "dataloader_num_workers",
        "learning_rate",
        "gradient_accumulation_steps",
        "output_dir",
        "save_steps",
        "save_total_limit",
        "num_gpus",
        "use_wandb",
        "max_steps",
        "weight_decay",
        "warmup_ratio",
        "wandb_project",
        "save_only_model",
        "skip_weight_loading",
    ):
        setattr(config.training, name, getattr(ft, name))
    config.training.gradient_checkpointing = True
    config.training.logging_steps = ft.logging_steps
    for name in ("shard_size", "episode_sampling_rate", "num_shards_per_epoch"):
        setattr(config.data, name, getattr(ft, name))
    return config


def main() -> None:
    os.environ.setdefault("LOGURU_LEVEL", "INFO")
    ft = tyro.cli(LearningStudioLoraConfig, description=__doc__)
    _RUNTIME.update(
        rank=ft.lora_rank,
        alpha=ft.lora_alpha,
        dropout=ft.lora_dropout,
        action_head_only=ft.lora_action_head_only,
        include_mlp=ft.lora_include_mlp,
        modules_to_save_action_head=ft.lora_modules_to_save_action_head,
    )
    install_lora_hooks()
    config = build_training_config(ft)
    output_dir = Path(config.training.output_dir)
    if config.training.experiment_name:
        output_dir /= config.training.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / ft.log_file
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        sys.stdout = TeeStream(original_stdout, log)
        sys.stderr = TeeStream(original_stderr, log)
        try:
            print(f"[learningstudio-lora] training log: {log_path.resolve()}")
            if config.training.use_wandb:
                print(
                    "[learningstudio-lora] W&B enabled: "
                    f"project={config.training.wandb_project}, "
                    f"logging_steps={config.training.logging_steps}"
                )
            run(config)
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    main()
