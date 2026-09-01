#!/usr/bin/env python3
"""Generate scene reasoning with the VLM weights stored inside a LingBot-VLA checkpoint.

This deliberately does not instantiate LingBot's action expert. LingBot checkpoints
normally omit Qwen's lm_head; in that case only that output projection is restored
from the matching base Qwen checkpoint, while every VLM backbone tensor comes from
the LingBot checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
import yaml
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration


PROMPT = """You are a robot manipulation reasoning assistant.

Observe the image and answer the following questions:
1. What objects are visible?
2. What is the target object?
3. What is the robot currently doing?
4. What stage of the task has been completed?
5. Is there any failure or abnormal situation?
6. What should the robot do next?
Answer concisely.

Use exactly these section labels:
Objects:
Target:
Current action:
Task progress:
Failure:
Next step:"""

VLM_PREFIX = "model.qwenvl_with_expert.qwenvl."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True,
                        help="LingBot-VLA HF checkpoint directory")
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        help="Qwen2.5-VL processor/config path. If omitted, infer it from lingbotvla_cli.yaml.",
    )
    parser.add_argument(
        "--lm_head_path",
        default=None,
        help="Base Qwen checkpoint containing lm_head.weight; defaults to --tokenizer_path. "
             "Used only when the LingBot checkpoint omitted lm_head.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn_implementation", choices=("eager", "sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 selects deterministic greedy decoding")
    parser.add_argument(
        "--output_log",
        type=Path,
        default=None,
        help="Optional UTF-8 log file for model provenance and generated reasoning",
    )
    return parser.parse_args()


def infer_tokenizer_path(model_path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    training_yaml = model_path / "lingbotvla_cli.yaml"
    if training_yaml.is_file():
        with training_yaml.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        tokenizer_path = config.get("model", {}).get("tokenizer_path")
        if tokenizer_path:
            return str(tokenizer_path)
    raise ValueError(
        "Could not infer tokenizer_path. Pass --tokenizer_path pointing to the matching "
        "Qwen2.5-VL-3B-Instruct processor/config directory. This supplies architecture and "
        "preprocessing files only; VLM parameters are loaded from --model_path."
    )


def checkpoint_files(model_path: Path) -> list[Path]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as handle:
            names = sorted(set(json.load(handle)["weight_map"].values()))
        files = [model_path / name for name in names]
    else:
        files = sorted(model_path.glob("*.safetensors"))
    missing = [str(path) for path in files if not path.is_file()]
    if not files or missing:
        raise FileNotFoundError(f"Invalid LingBot safetensors checkpoint; missing={missing or 'all shards'}")
    return files


def iter_vlm_tensors(files: Iterable[Path]):
    for shard in files:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for checkpoint_key in handle.keys():
                if checkpoint_key.startswith(VLM_PREFIX):
                    yield checkpoint_key[len(VLM_PREFIX):], handle.get_tensor(checkpoint_key)


def load_base_lm_head(path_string: str) -> torch.Tensor:
    path = Path(path_string)
    if not path.is_dir():
        raise FileNotFoundError(
            "LingBot omitted lm_head.weight, so --lm_head_path (or --tokenizer_path) must be "
            "a local base-Qwen checkpoint directory, not merely a Hugging Face repo ID."
        )
    for shard in checkpoint_files(path):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if "lm_head.weight" in handle.keys():
                return handle.get_tensor("lm_head.weight")
    raise KeyError(f"No lm_head.weight found in base Qwen checkpoint: {path}")


def load_lingbot_vlm(model_path: Path, tokenizer_path: str, args: argparse.Namespace):
    dtype = getattr(torch, args.dtype)
    config = AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True)
    config._attn_implementation = args.attn_implementation
    model = Qwen2_5_VLForConditionalGeneration.from_config(config).to(dtype=dtype)

    # Load only the checkpoint subtree used by LingBot's Qwen VLM.  The action expert,
    # state/action projections, and optional depth modules are never constructed.
    vlm_state = dict(iter_vlm_tensors(checkpoint_files(model_path)))
    if not vlm_state:
        raise RuntimeError(f"No tensors with checkpoint prefix {VLM_PREFIX!r} were found")

    checkpoint_has_lm_head = "lm_head.weight" in vlm_state
    missing, unexpected = model.load_state_dict(vlm_state, strict=False)
    allowed_missing = {"lm_head.weight"} if not checkpoint_has_lm_head else set()
    bad_missing = sorted(set(missing) - allowed_missing)
    if bad_missing or unexpected:
        raise RuntimeError(
            "LingBot VLM checkpoint does not match the tokenizer/config architecture. "
            f"Missing (first 20): {bad_missing[:20]}; unexpected (first 20): {unexpected[:20]}"
        )
    if not checkpoint_has_lm_head:
        lm_head_path = args.lm_head_path or tokenizer_path
        lm_head = load_base_lm_head(lm_head_path)
        if tuple(lm_head.shape) != tuple(model.lm_head.weight.shape):
            raise RuntimeError(
                f"Base lm_head shape {tuple(lm_head.shape)} does not match VLM "
                f"shape {tuple(model.lm_head.weight.shape)}"
            )
        model.lm_head.weight.data.copy_(lm_head.to(dtype=dtype))

    model.to(args.device).eval()
    provenance = [
        f"Loaded {len(vlm_state)} LingBot VLM tensors from: {model_path}",
        f"Checkpoint prefix: {VLM_PREFIX}",
        "lm_head source: " + ("LingBot checkpoint" if checkpoint_has_lm_head else
        f"base Qwen output projection only: {args.lm_head_path or tokenizer_path}"),
    ]
    print("\n".join(provenance))
    return model, provenance


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if not args.model_path.is_dir():
        raise NotADirectoryError(args.model_path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script on the GPU server or pass --device cpu")

    tokenizer_path = infer_tokenizer_path(args.model_path, args.tokenizer_path)
    processor = AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)
    model, provenance = load_lingbot_vlm(args.model_path, tokenizer_path, args)

    image = Image.open(args.image).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPT},
    ]}]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_text], images=[image], padding=True, return_tensors="pt")
    inputs = {name: tensor.to(args.device) for name, tensor in inputs.items()}

    generate_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.temperature > 0:
        generate_kwargs.update(do_sample=True, temperature=args.temperature)
    else:
        generate_kwargs["do_sample"] = False
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    reasoning = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    print("\n=== LingBot VLM Reasoning ===\n")
    print(reasoning)

    if args.output_log is not None:
        args.output_log.parent.mkdir(parents=True, exist_ok=True)
        log_text = "\n".join([
            "=== LingBot VLM Model Provenance ===",
            *provenance,
            f"Image: {args.image.resolve()}",
            "",
            "=== LingBot VLM Reasoning ===",
            "",
            reasoning,
            "",
        ])
        args.output_log.write_text(log_text, encoding="utf-8")
        print(f"\nSaved log to: {args.output_log.resolve()}")


if __name__ == "__main__":
    main()
