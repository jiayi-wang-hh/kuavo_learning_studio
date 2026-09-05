#!/usr/bin/env python3
"""Benchmark Qwen, RoboBrain, and Cosmos models on scene reasoning images.

For the controlled Qwen comparison, run qwen25_vl_3b, qwen3_vl_4b, and
qwen35_4b with the same CLI generation and image-processing arguments.
"""
from __future__ import annotations

import argparse, csv, json, platform, re, statistics, sys, time
from pathlib import Path
from typing import Any
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor

PROMPT = """You are a visual scene-understanding assistant for robot manipulation.

Task: Pick the red apple from the cluttered scene.

Analyze only what is directly visible in this single image.
Do not infer motion or task history that cannot be established from one image.
For current_action, task_progress, or failure, use "UNCERTAIN" when visual
evidence is insufficient.

Return exactly one valid JSON object with these keys:
{
  "objects": ["short object name"],
  "target": "red apple | NOT_VISIBLE | UNCERTAIN",
  "current_action": "one short observable description | UNCERTAIN",
  "task_progress": "NOT_STARTED | APPROACHING | CONTACTING | GRASPED | LIFTED | COMPLETED | UNCERTAIN",
  "failure": "NO_VISIBLE_FAILURE | POSSIBLE_FAILURE | FAILURE | UNCERTAIN",
  "next_step": "one short executable robot action | UNCERTAIN"
}

Return only the JSON object. Do not output Markdown, code fences,
step-by-step reasoning, frame-by-frame narration, or additional text.
Keep every string concise and keep the full response below 160 tokens."""

MODE_INFO = {
    "qwen25_vl_3b": ("Qwen2.5-VL-3B-Instruct", "/media/data/jiayi/hf_model/Qwen2.5-VL-3B-Instruct", "qwen25_vl_3b.log"),
    "qwen3_vl_4b": ("Qwen3-VL-4B-Instruct", "/media/data/jiayi/hf_model/Qwen3-VL-4B-Instruct", "qwen3_vl_4b.log"),
    "qwen35_4b": ("Qwen3.5-4B", "/media/data/jiayi/hf_model/Qwen3.5-4B", "qwen35_4b.log"),
    "qwen3_8_27b": ("Qwen3.8-27B", "/media/data/jiayi/hf_model/Qwen3.8-27B", "qwen3_8_27b.log"),
    "robobrain": ("RoboBrain2.0-7B", "/media/data/jiayi/hf_model/RoboBrain2.0-7B", "robobrain2_7b.log"),
    "robobrain_2.5": ("RoboBrain2.5-8B-NV", "/media/data/jiayi/hf_model/RoboBrain2.5-8B-NV", "robobrain2_5_8b_nv.log"),
    "cosmos_reason2": ("Cosmos-Reason2-2B", "~/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B", "cosmos_reason2_2b.log"),    "cosmos_reason2_32b": (
        "Cosmos-Reason2-32B",
        "/media/data/jiayi/hf_model/Cosmos-Reason2-32B",
        "cosmos_reason2_32b.log",
    ),

}

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=tuple(MODE_INFO), required=True)
    p.add_argument("--model_path", type=Path)
    p.add_argument("--image_dir", type=Path, default=Path("/media/data/jiayi/dataset/apple_scene_frames"))
    p.add_argument("--output_log", type=Path)
    p.add_argument("--metrics_dir", type=Path, default=Path("/media/data/jiayi/outputs/vlm_test/metrics"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--attn_implementation", choices=("eager", "sdpa", "flash_attention_2"), default="sdpa")
    p.add_argument("--max_new_tokens", type=int, default=192)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name")
    p.add_argument("--device_map", default="auto")
    p.add_argument(
        "--qwen38_thinking",
        action="store_true",
        help="Enable Qwen3.8 thinking. Disabled by default for comparable structured output.",
    )
    p.add_argument(
        "--qwen35_thinking",
        action="store_true",
        help="Enable Qwen3.5 thinking. Disabled by default for the fair Qwen comparison.",
    )
    p.add_argument("--robobrain25_repo", type=Path, default=Path("/media/data/jiayi/RoboBrain2.5"))
    return p.parse_args()

def frame_time(path):
    match = re.search(r"frame_(\d+)s", path.stem)
    if not match:
        raise ValueError(f"Cannot parse frame time: {path}")
    return int(match.group(1))

def resolve_hf_cache_snapshot(path: Path) -> Path:
    """Turn a Hugging Face Hub models--org--name directory into a snapshot."""
    path = path.expanduser()
    if (path / "config.json").is_file():
        return path
    main_ref = path / "refs" / "main"
    if main_ref.is_file():
        snapshot = path / "snapshots" / main_ref.read_text(encoding="utf-8").strip()
        if (snapshot / "config.json").is_file():
            return snapshot
    candidates = [p for p in (path / "snapshots").glob("*") if (p / "config.json").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No complete Hugging Face snapshot under: {path}")
    return max(candidates, key=lambda p: p.stat().st_mtime)

def sync_cuda():
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            with torch.cuda.device(device_index):
                torch.cuda.synchronize()

def reset_peaks():
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            with torch.cuda.device(device_index):
                torch.cuda.reset_peak_memory_stats()

def memory_mb():
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "peak_allocated": 0.0, "peak_reserved": 0.0}
    scale = 1024 ** 2
    totals = {
        "allocated": 0,
        "reserved": 0,
        "peak_allocated": 0,
        "peak_reserved": 0,
    }
    # Use an explicit CUDA context instead of passing an integer device
    # argument. Some PyTorch/CUDA builds reject integer arguments here.
    for device_index in range(torch.cuda.device_count()):
        with torch.cuda.device(device_index):
            totals["allocated"] += torch.cuda.memory_allocated()
            totals["reserved"] += torch.cuda.memory_reserved()
            totals["peak_allocated"] += torch.cuda.max_memory_allocated()
            totals["peak_reserved"] += torch.cuda.max_memory_reserved()
    return {key: value / scale for key, value in totals.items()}

def percentile(values, fraction):
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]

REQUIRED_OUTPUT_KEYS = (
    "objects", "target", "current_action", "task_progress", "failure", "next_step"
)

def parse_json_output(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a single JSON object, tolerating an accidental Markdown fence."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(value, dict):
        return None, "top-level output is not a JSON object"
    return value, None

def output_status(text: str) -> dict[str, Any]:
    parsed, error = parse_json_output(text)
    missing = list(REQUIRED_OUTPUT_KEYS) if parsed is None else [
        key for key in REQUIRED_OUTPUT_KEYS if key not in parsed
    ]
    nonempty = parsed is not None and all(
        parsed.get(key) not in (None, "", []) for key in REQUIRED_OUTPUT_KEYS
    )
    return {
        "output_valid_json": parsed is not None,
        "output_complete": parsed is not None and not missing and nonempty,
        "missing_fields": "|".join(missing),
        "parse_error": error or "",
        "parsed": parsed or {},
    }

def repetition_score(text):
    words = re.findall(r"\w+", text.lower())
    return 1.0 if not words else 1.0 - len(set(words)) / len(words)

def max_token_run(text):
    words = re.findall(r"\w+", text.lower())
    best = current = int(bool(words))
    for a, b in zip(words, words[1:]):
        current = current + 1 if a == b else 1
        best = max(best, current)
    return best

def input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def load_model(model_path: Path, args):
    """Load the selected model and its processor."""
    name = MODE_INFO[args.mode][0]
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        use_fast=False,
    )

    if args.mode == "qwen25_vl_3b":
        from transformers import Qwen2_5_VLForConditionalGeneration
        cls = Qwen2_5_VLForConditionalGeneration
    elif args.mode == "qwen3_vl_4b":
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen3-VL requires a recent Transformers version containing "
                "Qwen3VLForConditionalGeneration. Upgrade Transformers first."
            ) from exc
        cls = Qwen3VLForConditionalGeneration
    elif args.mode == "qwen35_4b":
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError as exc:
            raise ImportError(
                "Qwen3.5 requires the latest Transformers version containing "
                "AutoModelForMultimodalLM."
            ) from exc
        cls = AutoModelForMultimodalLM
    elif args.mode in ("qwen3_8_27b", "robobrain"):
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError as exc:
            raise ImportError(
                f"{name} requires a recent Transformers version containing "
                "AutoModelForMultimodalLM."
            ) from exc
        cls = AutoModelForMultimodalLM
    elif args.mode == "robobrain_2.5":
        from transformers import AutoModelForImageTextToText
        cls = AutoModelForImageTextToText
    elif args.mode in ("cosmos_reason2", "cosmos_reason2_32b"):
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                f"{name} requires a Transformers version with Qwen3-VL support."
            ) from exc
        cls = Qwen3VLForConditionalGeneration
    else:
        raise ValueError(args.mode)

    dtype = getattr(torch, args.dtype)
    kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": args.attn_implementation,
        "device_map": args.device_map,
    }
    if args.mode in (
        "qwen3_vl_4b",
        "qwen35_4b",
        "qwen3_8_27b",
        "cosmos_reason2",
        "cosmos_reason2_32b",
    ):
        kwargs["dtype"] = dtype
    else:
        kwargs["torch_dtype"] = dtype

    print(f"Loading {name} from {model_path}")
    model = cls.from_pretrained(str(model_path), **kwargs)
    model.eval()
    provenance = [
        f"Mode: {args.mode}",
        f"Model: {name}",
        f"Checkpoint: {model_path}",
        f"Processor: {model_path}",
        f"dtype: {args.dtype}",
        f"attention: {args.attn_implementation}",
        f"device_map: {args.device_map}",
        f"Backend: {cls.__name__}",
    ]
    if args.mode == "qwen35_4b":
        provenance.append(f"Thinking: {args.qwen35_thinking}")
    elif args.mode == "qwen3_8_27b":
        provenance.append(f"Thinking: {args.qwen38_thinking}")
    return model, processor, provenance


def generate(model, processor, image_path, args):
    total_start = time.perf_counter()
    if args.mode == "robobrain_2.5":
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [{"type": "image", "image": f"file://{image_path.resolve()}"}, {"type": "text", "text": PROMPT}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
        # Preserve compatibility with the previous custom UnifiedInference
        # wrapper while also supporting the Transformers auto model.
        inner = model.model if model.__class__.__name__ == "UnifiedInference" else model
    elif args.mode in ("cosmos_reason2", "cosmos_reason2_32b"):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "image", "image": str(image_path.resolve())}, {"type": "text", "text": PROMPT}]},
        ]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inner = model
    else:
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
        template_kwargs = {}
        if args.mode == "qwen3_8_27b":
            template_kwargs["enable_thinking"] = args.qwen38_thinking
        elif args.mode == "qwen35_4b":
            template_kwargs["enable_thinking"] = args.qwen35_thinking
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        if args.mode == "robobrain":
            text += "<think></think><answer>"
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inner = model
    inputs = inputs.to(input_device(inner))
    sync_cuda()
    preprocess_s = time.perf_counter() - total_start
    kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": args.temperature > 0}
    if args.temperature > 0:
        kwargs["temperature"] = args.temperature
    start = time.perf_counter()
    with torch.inference_mode():
        output = inner.generate(**inputs, **kwargs)
    sync_cuda()
    generation_s = time.perf_counter() - start
    input_tokens = inputs.input_ids.shape[1]
    new_ids = output[:, input_tokens:]
    reasoning = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    if args.mode == "robobrain":
        for tag in ("<think>", "</think>", "<answer>", "</answer>"):
            reasoning = reasoning.replace(tag, "")
    total_s = time.perf_counter() - total_start
    nout = int(new_ids.shape[1])
    return reasoning.strip(), {
        "preprocess_seconds": preprocess_s,
        "generation_seconds": generation_s,
        "end_to_end_seconds": total_s,
        "input_tokens": int(input_tokens),
        "output_tokens": nout,
        "tokens_per_second": nout / generation_s,
        "hit_max_new_tokens": nout >= args.max_new_tokens,
    }

def main():
    args = parse_args()
    name, default_path, default_log = MODE_INFO[args.mode]
    model_path = args.model_path or Path(default_path)
    if args.mode in ("cosmos_reason2", "cosmos_reason2_32b"):
        model_path = resolve_hf_cache_snapshot(model_path)
    output_log = args.output_log or Path("/media/data/jiayi/outputs/vlm_test/logs") / default_log
    if not model_path.is_dir() or not args.image_dir.is_dir():
        raise NotADirectoryError(f"Check model/image paths: {model_path}, {args.image_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    paths = sorted(args.image_dir.rglob("*.png"), key=lambda p: (p.parent.as_posix(), frame_time(p)))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise RuntimeError("No PNG images found")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.empty_cache(); reset_peaks(); sync_cuda()
    start = time.perf_counter()
    model, processor, provenance = load_model(model_path, args)
    sync_cuda()
    load_seconds = time.perf_counter() - start
    model_memory = memory_mb()
    print("\n".join(provenance))
    print(f"Load: {load_seconds:.3f}s; model memory: {model_memory['allocated']:.1f} MB")
    for i in range(args.warmup):
        print(f"Warm-up {i+1}/{args.warmup}")
        generate(model, processor, paths[0], args)

    output_log.parent.mkdir(parents=True, exist_ok=True)
    output_log.write_text("", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for i, path in enumerate(paths, 1):
        reset_peaks(); before = memory_mb()
        reasoning, timing = generate(model, processor, path, args)
        after = memory_mb()
        status = output_status(reasoning)
        parsed = status.pop("parsed")
        row = {
            "mode": args.mode,
            "model": name,
            "image": str(path.resolve()),
            "episode": path.parent.name,
            "frame_seconds": frame_time(path),
            "pred_objects": json.dumps(parsed.get("objects"), ensure_ascii=False),
            "pred_target": parsed.get("target", ""),
            "pred_current_action": parsed.get("current_action", ""),
            "pred_task_progress": parsed.get("task_progress", ""),
            "pred_failure": parsed.get("failure", ""),
            "pred_next_step": parsed.get("next_step", ""),
            **timing,
            "model_memory_mb": before["allocated"],
            "peak_allocated_mb": after["peak_allocated"],
            "peak_reserved_mb": after["peak_reserved"],
            "incremental_peak_mb": max(0.0, after["peak_allocated"] - before["allocated"]),
            **status,
            "repetition_score": repetition_score(reasoning),
            "max_token_run": max_token_run(reasoning),
        }
        rows.append(row)
        print(f"[{i}/{len(paths)}] {path}: {timing['end_to_end_seconds']:.3f}s, {timing['tokens_per_second']:.2f} token/s, peak {after['peak_allocated']:.1f} MB")
        block = "\n".join(["="*80, f"Image: {path.resolve()}", *provenance, "=== Performance ===", json.dumps(row, ensure_ascii=False), f"=== {name} Reasoning ===", reasoning, ""])
        with output_log.open("a", encoding="utf-8") as f:
            f.write(block + "\n")

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    base = args.metrics_dir / f"{args.mode}_{suffix}"
    with Path(str(base) + "_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    e2e = [r["end_to_end_seconds"] for r in rows]; gen = [r["generation_seconds"] for r in rows]
    summary = {"mode": args.mode, "model": name, "model_path": str(model_path.resolve()), "image_dir": str(args.image_dir.resolve()), "num_images": len(rows), "seed": args.seed, "dtype": args.dtype, "attention": args.attn_implementation, "max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "warmup": args.warmup, "qwen35_thinking": args.qwen35_thinking if args.mode == "qwen35_4b" else None, "qwen38_thinking": args.qwen38_thinking if args.mode == "qwen3_8_27b" else None, "model_load_seconds": load_seconds, "model_allocated_mb": model_memory["allocated"], "model_reserved_mb": model_memory["reserved"], "e2e_mean_s": statistics.mean(e2e), "e2e_median_s": statistics.median(e2e), "e2e_p95_s": percentile(e2e, .95), "generation_mean_s": statistics.mean(gen), "generation_median_s": statistics.median(gen), "generation_p95_s": percentile(gen, .95), "tokens_per_second_mean": statistics.mean(r["tokens_per_second"] for r in rows), "peak_allocated_mb": max(r["peak_allocated_mb"] for r in rows), "peak_reserved_mb": max(r["peak_reserved_mb"] for r in rows), "valid_json_rate": sum(r["output_valid_json"] for r in rows)/len(rows), "completion_rate": sum(r["output_complete"] for r in rows)/len(rows), "truncation_rate": sum(r["hit_max_new_tokens"] for r in rows)/len(rows), "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "prompt": PROMPT}
    summary_path = Path(str(base) + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Reasoning: {output_log}\nMetrics: {base}_per_image.csv\nSummary: {summary_path}")

if __name__ == "__main__":
    main()
