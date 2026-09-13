#!/usr/bin/env python3
"""Offline Florence-2 probe for basic robot-scene visual facts.

This is deliberately independent from the V2 trigger. It compares a native
Florence task prompt with the free-form JSON contract used by the prototype.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


FACT_KEYS = (
    "gripper_visible",
    "target_visible",
    "grasp_state",
    "object_position",
    "vertical_relation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Florence-2 basic visual facts")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default="microsoft/Florence-2-base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```")
        text = text.removesuffix("```").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def fact_prompt(target: str) -> str:
    return f"""Inspect this robot image. The target object is: {target}.
Report only directly visible facts. Return exactly one JSON object:
{{
  "gripper_visible": true,
  "target_visible": true,
  "grasp_state": "HELD|NOT_HELD|AMBIGUOUS",
  "object_position": "LEFT|CENTER|RIGHT|UNKNOWN",
  "vertical_relation": "ON_SURFACE|IN_AIR|IN_CONTAINER|UNKNOWN"
}}
HELD means the target is visibly supported by and moving with the closed gripper;
from one still image use HELD only when contact/support is visually clear."""


def generate(model, processor, image: Image.Image, prompt: str, device, max_tokens: int):
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - started) * 1000
    if not getattr(model.config, "is_encoder_decoder", False):
        generated = generated[:, inputs["input_ids"].shape[1] :]
    text = processor.batch_decode(generated, skip_special_tokens=False)[0]
    return text, latency_ms


def validate_prediction(value: dict) -> bool:
    return (
        set(FACT_KEYS).issubset(value)
        and isinstance(value["gripper_visible"], bool)
        and isinstance(value["target_visible"], bool)
        and value["grasp_state"] in {"HELD", "NOT_HELD", "AMBIGUOUS"}
        and value["object_position"] in {"LEFT", "CENTER", "RIGHT", "UNKNOWN"}
        and value["vertical_relation"]
        in {"ON_SURFACE", "IN_AIR", "IN_CONTAINER", "UNKNOWN"}
    )


def safe_generate(model, processor, image, prompt, device, max_tokens):
    """Record unsupported prompts/generation failures as experimental outcomes."""
    try:
        raw, latency_ms = generate(
            model, processor, image, prompt, device, max_tokens
        )
        return raw, latency_ms, None
    except Exception as exc:
        return "", None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    args = parse_args()
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must be a non-empty JSON list")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype
    ).to(device).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for row in rows:
            image_path = (args.manifest.parent / row["image"]).resolve()
            image = Image.open(image_path).convert("RGB")
            native_prompt = "<MORE_DETAILED_CAPTION>"
            native_raw, native_ms, native_error = safe_generate(
                model, processor, image, native_prompt, device, args.max_new_tokens
            )
            try:
                if native_error:
                    raise RuntimeError(native_error)
                native_processed = processor.post_process_generation(
                    native_raw, task=native_prompt, image_size=image.size
                )
            except Exception as exc:
                native_processed = {"postprocess_error": type(exc).__name__}
            json_raw, json_ms, json_error = safe_generate(
                model, processor, image, fact_prompt(row["target"]), device,
                args.max_new_tokens,
            )
            prediction = extract_json(json_raw)
            record = {
                "id": row["id"],
                "image": str(image_path),
                "target": row["target"],
                "ground_truth": row.get("ground_truth"),
                "native_detailed_caption": native_processed,
                "native_raw": native_raw,
                "native_error": native_error,
                "native_latency_ms": native_ms,
                "free_json_raw": json_raw,
                "free_json_error": json_error,
                "free_json_prediction": prediction,
                "free_json_schema_valid": validate_prediction(prediction),
                "free_json_latency_ms": json_ms,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
