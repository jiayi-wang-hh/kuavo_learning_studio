#!/usr/bin/env python3
"""Compare Qwen2.5-VL and RoboBrain2.0 scene reasoning on identical images."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import re


PROMPT = """You are a robot manipulation reasoning assistant.

The robot's task instruction: pick the red apple from the cluttered scene.

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


MODE_INFO = {
    "qwen": {
        "name": "Qwen2.5-VL-3B-Instruct baseline",
        "default_path": "/root/bayes-tmp/jiayi/hf_models/Qwen2.5-VL-3B-Instruct",
        "default_log": (
            "/root/bayes-tmp/jiayi/kuavo_learning_studio/outputs/vlm_test/logs/"
            "qwen25_vl_3b_baseline.log"
        ),
    },
    "robobrain": {
        "name": "RoboBrain2.0-7B",
        "default_path": "/root/bayes-tmp/jiayi/hf_models/RoboBrain2.0-7B",
        "default_log": (
            "/root/bayes-tmp/jiayi/kuavo_learning_studio/outputs/vlm_test/logs/"
            "robobrain2_7b.log"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(MODE_INFO), required=True)
    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="Override the default local checkpoint path selected by --mode",
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=Path("/root/bayes-tmp/kuavo_dataset/apple_scene_frames"),
        help="Directory containing PNG images recursively",
    )
    parser.add_argument(
        "--output_log",
        type=Path,
        default=None,
        help="Override the mode-specific output log path",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 uses deterministic greedy decoding",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally test only the first N images",
    )
    return parser.parse_args()

def frame_time(path: Path) -> int:
    match = re.search(r"frame_(\d+)s", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse frame time from: {path}")
    return int(match.group(1))

def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    mode_info = MODE_INFO[args.mode]
    model_path = args.model_path or Path(mode_info["default_path"])
    output_log = args.output_log or Path(mode_info["default_log"])
    return model_path, output_log


def load_model_and_processor(
    model_path: Path,
    args: argparse.Namespace,
):
    dtype = getattr(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        use_fast=False,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(args.device).eval()

    model_name = MODE_INFO[args.mode]["name"]
    provenance = [
        f"Mode: {args.mode}",
        f"Model type: {model_name}",
        f"Full checkpoint: {model_path.resolve()}",
        f"Processor source: {model_path.resolve()}",
        f"dtype: {args.dtype}",
        f"attention: {args.attn_implementation}",
        "LingBot weights loaded: no",
    ]
    print("\n".join(provenance))
    return model, processor, provenance


def generate_reasoning(
    model,
    processor,
    image_path: Path,
    args: argparse.Namespace,
) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if args.mode == "robobrain":
        chat_text += "<think></think><answer>"
    inputs = processor(
        text=[chat_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {name: tensor.to(args.device) for name, tensor in inputs.items()}

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generate_kwargs["temperature"] = args.temperature

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    new_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    reasoning = processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    if args.mode == "robobrain":
            for tag in (
                "<think>",
                "</think>",
                "<answer>",
                "</answer>",
            ):
                reasoning = reasoning.replace(tag, "")

            reasoning = reasoning.strip()

    return reasoning


def main() -> None:
    args = parse_args()
    model_path, output_log = resolve_paths(args)

    if not model_path.is_dir():
        raise NotADirectoryError(model_path)
    if not args.image_dir.is_dir():
        raise NotADirectoryError(args.image_dir)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run on the GPU server or pass --device cpu")

    image_paths = sorted(
        args.image_dir.rglob("*.png"),
        key=lambda path: (path.parent.as_posix(), frame_time(path)),
    )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise RuntimeError(f"No PNG images found under: {args.image_dir}")

    model, processor, provenance = load_model_and_processor(model_path, args)
    output_log.parent.mkdir(parents=True, exist_ok=True)
    output_log.write_text("", encoding="utf-8")
    model_name = MODE_INFO[args.mode]["name"]

    print(f"\nFound {len(image_paths)} images.")
    for index, image_path in enumerate(image_paths, start=1):
        print(f"\n[{index}/{len(image_paths)}] Testing: {image_path}")
        reasoning = generate_reasoning(model, processor, image_path, args)
        print(f"\n=== {model_name} Reasoning ===\n")
        print(reasoning)

        log_text = "\n".join(
            [
                "=" * 80,
                f"Image: {image_path.resolve()}",
                "",
                f"=== {model_name} Provenance ===",
                *provenance,
                "",
                f"=== {model_name} Reasoning ===",
                "",
                reasoning,
                "",
            ]
        )
        with output_log.open("a", encoding="utf-8") as handle:
            handle.write(log_text)
            handle.write("\n")

    print(f"\nAll results saved to: {output_log.resolve()}")


if __name__ == "__main__":
    main()
