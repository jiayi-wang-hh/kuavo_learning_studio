import argparse
import os
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lingbotvla.checkpoint import ckpt_to_state_dict
from lingbotvla.models import save_model_weights
from lingbotvla.utils import helper


logger = helper.create_logger(__name__)


def merge_lora_state_dict(state_dict):
    lora_prefixes = sorted(
        key[: -len(".lora_A.weight")]
        for key in state_dict.keys()
        if key.endswith(".lora_A.weight")
    )
    if not lora_prefixes:
        raise ValueError("`--merge-lora` was set, but no LoRA tensors were found in the checkpoint.")

    merged_state_dict = {}
    lora_prefix_set = set(lora_prefixes)

    def is_lora_internal_key(key):
        for suffix in (".base_layer.weight", ".base_layer.bias", ".lora_A.weight", ".lora_B.weight", ".lora_scaling"):
            if key.endswith(suffix) and key[: -len(suffix)] in lora_prefix_set:
                return True
        return False

    for key, value in state_dict.items():
        if not is_lora_internal_key(key):
            merged_state_dict[key] = value

    for prefix in lora_prefixes:
        base_weight_key = f"{prefix}.base_layer.weight"
        lora_a_key = f"{prefix}.lora_A.weight"
        lora_b_key = f"{prefix}.lora_B.weight"
        scaling_key = f"{prefix}.lora_scaling"

        base_weight = state_dict[base_weight_key]
        lora_a = state_dict[lora_a_key]
        lora_b = state_dict[lora_b_key]
        scaling = state_dict.get(scaling_key, torch.tensor(1.0))
        scaling = float(scaling.detach().float().cpu().item())
        delta = torch.matmul(lora_b.float(), lora_a.float()) * scaling
        merged_state_dict[f"{prefix}.weight"] = (base_weight.float() + delta).to(base_weight.dtype)

        base_bias_key = f"{prefix}.base_layer.bias"
        if base_bias_key in state_dict:
            merged_state_dict[f"{prefix}.bias"] = state_dict[base_bias_key]

    return merged_state_dict, len(lora_prefixes)


def copy_cli_yaml(load_dir: str, save_path: str, model_assets_dir: str = None):
    candidates = []
    load_path = Path(load_dir).resolve()
    candidates.extend(parent / "lingbotvla_cli.yaml" for parent in [load_path, *load_path.parents])
    if model_assets_dir is not None:
        candidates.append(Path(model_assets_dir) / "lingbotvla_cli.yaml")

    for candidate in candidates:
        if candidate.exists():
            shutil.copy2(candidate, Path(save_path) / "lingbotvla_cli.yaml")
            logger.info(f"Copied lingbotvla_cli.yaml from {candidate}")
            return
    logger.warning("No lingbotvla_cli.yaml found to copy into the HF checkpoint directory.")


def copy_model_assets(model_assets_dir: str | None, save_path: str):
    if model_assets_dir is None:
        return

    source_dir = Path(model_assets_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"model_assets_dir does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"model_assets_dir is not a directory: {source_dir}")

    target_dir = Path(save_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.iterdir():
        target = target_dir / source.name
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    logger.info(f"Copied model assets from {source_dir} to {target_dir}")


def merge_to_hf_pt(
    load_dir: str,
    save_path: str,
    model_assets_dir: str = None,
    ckpt_manager: str = "bytecheckpoint",
    merge_lora: bool = False,
):
    # save model in huggingface's format
    state_dict = ckpt_to_state_dict(
        save_checkpoint_path=load_dir,
        output_dir=save_path,
        ckpt_manager=ckpt_manager,
    )
    if merge_lora:
        state_dict, merged_count = merge_lora_state_dict(state_dict)
        logger.info(f"Merged LoRA weights into {merged_count} Linear layers.")

    copy_model_assets(model_assets_dir, save_path)
    save_model_weights(save_path, state_dict)
    copy_cli_yaml(load_dir, save_path, model_assets_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--model_assets_dir", type=str, default=None)
    parser.add_argument("--ckpt-manager", type=str, choices=["bytecheckpoint", "dcp", "native"], default="bytecheckpoint")
    parser.add_argument("--merge-lora", action="store_true")
    args = parser.parse_args()
    load_dir = args.load_dir
    save_dir = os.path.join(load_dir, "hf_ckpt") if args.save_dir is None else args.save_dir
    model_assets_dir = args.model_assets_dir
    logger.info(f"Merge Args: {args}")
    merge_to_hf_pt(load_dir, save_dir, model_assets_dir, args.ckpt_manager, args.merge_lora)
    logger.info(f"Merge to hf pt success! Save to: {save_dir}")
