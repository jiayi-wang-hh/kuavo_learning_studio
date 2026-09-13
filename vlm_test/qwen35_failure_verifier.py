#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
)

from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--prompt",
        required=True,
    )

    parser.add_argument(
        "--device-map",
        default="auto",
    )

    parser.add_argument(
        "--dtype",
        choices=[
            "bfloat16",
            "float16",
            "float32",
        ],
        default="bfloat16",
    )

    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=192,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    return parser.parse_args()


def extract_json(text: str) -> dict:

    text = text.strip()

    if text.startswith("```"):
        text = text.replace(
            "```json",
            "",
        ).replace(
            "```",
            "",
        ).strip()

    try:
        value = json.loads(text)

        if isinstance(value, dict):
            return value

    except json.JSONDecodeError:
        pass

    # Find embedded JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:

        try:
            value = json.loads(
                text[start:end + 1]
            )

            if isinstance(value, dict):
                return value

        except json.JSONDecodeError:
            pass

    return {}


def main():

    args = parse_args()

    video = args.video.expanduser().resolve()
    model_path = (
        args.model_path
        .expanduser()
        .resolve()
    )

    if not video.is_file():
        raise FileNotFoundError(video)

    if not model_path.is_dir():
        raise NotADirectoryError(model_path)

    dtype = getattr(
        torch,
        args.dtype,
    )

    # IMPORTANT:
    # All diagnostic messages go to STDERR.
    # STDOUT must contain JSON only because the parent process parses it.

    print(
        f"[qwen35 verifier] loading model from {model_path}",
        file=sys.stderr,
    )

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        use_fast=False,
    )

    model = (
        AutoModelForMultimodalLM
        .from_pretrained(
            str(model_path),

            trust_remote_code=True,

            dtype=dtype,

            device_map=args.device_map,

            attn_implementation=(
                args.attn_implementation
            ),
        )
        .eval()
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video}",
                    "fps": 4.0,
                },
                {
                    "type": "text",
                    "text": args.prompt,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    image_inputs, video_inputs = (
        process_vision_info(
            messages
        )
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Find actual model input device
    try:
        device = (
            model
            .get_input_embeddings()
            .weight
            .device
        )

    except Exception:
        device = next(
            model.parameters()
        ).device

    inputs = inputs.to(device)

    generate_kwargs = {
        "max_new_tokens":
            args.max_new_tokens,

        "do_sample":
            args.temperature > 0,
    }

    if args.temperature > 0:

        generate_kwargs[
            "temperature"
        ] = args.temperature

    start = time.perf_counter()

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            **generate_kwargs,
        )

    generation_seconds = (
        time.perf_counter() - start
    )

    input_length = (
        inputs.input_ids.shape[1]
    )

    generated_ids = (
        generated_ids[
            :,
            input_length:
        ]
    )

    raw = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    parsed = extract_json(raw)

    if not parsed:

        result = {
            "failure_type": "UNKNOWN",
            "recovery_action": "STOP",
            "confidence": "LOW",
            "evidence": (
                "Qwen3.5 returned malformed output."
            ),
        }

    else:
        result = parsed

    # Diagnostics go to stderr
    print(
        f"[qwen35 verifier] inference={generation_seconds:.3f}s",
        file=sys.stderr,
    )

    # ONLY JSON goes to stdout
    print(
        json.dumps(
            result,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
