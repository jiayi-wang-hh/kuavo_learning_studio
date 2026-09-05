#!/usr/bin/env python3
"""Benchmark Qwen2.5-VL, RoboBrain2.0, and RoboBrain2.5 scene reasoning."""
from __future__ import annotations

import argparse, csv, json, platform, re, statistics, sys, time
from pathlib import Path
from typing import Any
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor

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
    "qwen": ("Qwen2.5-VL-3B-Instruct baseline", "/media/data/jiayi/hf_model/Qwen2.5-VL-3B-Instruct", "qwen25_vl_3b_baseline.log"),
    "robobrain": ("RoboBrain2.0-7B", "/media/data/jiayi/hf_model/RoboBrain2.0-7B", "robobrain2_7b.log"),
    "robobrain_2.5": ("RoboBrain2.5-8B-NV", "/media/data/jiayi/hf_model/RoboBrain2.5-8B-NV", "robobrain2_5_8b_nv.log"),
    "cosmos_reason2": ("Cosmos-Reason2-2B", "~/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B", "cosmos_reason2_2b.log"),
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
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name")
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
        torch.cuda.synchronize()

def reset_peaks():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def memory_mb():
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "peak_allocated": 0.0, "peak_reserved": 0.0}
    scale = 1024 ** 2
    return {
        "allocated": torch.cuda.memory_allocated() / scale,
        "reserved": torch.cuda.memory_reserved() / scale,
        "peak_allocated": torch.cuda.max_memory_allocated() / scale,
        "peak_reserved": torch.cuda.max_memory_reserved() / scale,
    }

def percentile(values, fraction):
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]

def complete(text):
    text = text.lower()
    return all(x in text for x in ("objects:", "target:", "current action:", "task progress:", "failure:", "next step:"))

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
    return model.get_input_embeddings().weight.device

def load_model(path, args):
    name = MODE_INFO[args.mode][0]
    if args.mode == "robobrain_2.5":
        repo = args.robobrain25_repo.resolve()
        if not repo.is_dir():
            raise NotADirectoryError(repo)
        sys.path.insert(0, str(repo))
        from inference import UnifiedInference
        wrapper = UnifiedInference(str(path))
        provenance = [f"Mode: {args.mode}", f"Model: {name}", f"Checkpoint: {path.resolve()}", f"Official repo: {repo}", "Backend: UnifiedInference internal model/processor"]
        return wrapper, wrapper.processor, provenance

    processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True, use_fast=False)
    if args.mode == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration
        cls = Qwen2_5_VLForConditionalGeneration
    elif args.mode == "robobrain":
        from transformers import AutoModelForMultimodalLM
        cls = AutoModelForMultimodalLM
    else:
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Cosmos-Reason2 requires transformers>=4.57.0 with Qwen3-VL support"
            ) from exc
        cls = Qwen3VLForConditionalGeneration
    load_kwargs = {"attn_implementation": args.attn_implementation, "trust_remote_code": True, "low_cpu_mem_usage": True}
    if args.mode == "cosmos_reason2":
        load_kwargs["dtype"] = getattr(torch, args.dtype)
    else:
        load_kwargs["torch_dtype"] = getattr(torch, args.dtype)
    model = cls.from_pretrained(str(path), **load_kwargs)
    model.to(args.device).eval()
    provenance = [f"Mode: {args.mode}", f"Model: {name}", f"Checkpoint: {path.resolve()}", f"Processor: {path.resolve()}", f"dtype: {args.dtype}", f"attention: {args.attn_implementation}", f"Backend: {cls.__name__}"]
    return model, processor, provenance

def generate(model, processor, image_path, args):
    total_start = time.perf_counter()
    if args.mode == "robobrain_2.5":
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [{"type": "image", "image": f"file://{image_path.resolve()}"}, {"type": "text", "text": PROMPT}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
        inner = model.model
    elif args.mode == "cosmos_reason2":
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "image", "image": str(image_path.resolve())}, {"type": "text", "text": PROMPT}]},
        ]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inner = model
    else:
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    return reasoning.strip(), {"preprocess_seconds": preprocess_s, "generation_seconds": generation_s, "end_to_end_seconds": total_s, "input_tokens": int(input_tokens), "output_tokens": nout, "tokens_per_second": nout / generation_s}

def main():
    args = parse_args()
    name, default_path, default_log = MODE_INFO[args.mode]
    model_path = args.model_path or Path(default_path)
    if args.mode == "cosmos_reason2":
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
        row = {"mode": args.mode, "model": name, "image": str(path.resolve()), "episode": path.parent.name, "frame_seconds": frame_time(path), **timing, "model_memory_mb": before["allocated"], "peak_allocated_mb": after["peak_allocated"], "peak_reserved_mb": after["peak_reserved"], "incremental_peak_mb": max(0.0, after["peak_allocated"] - before["allocated"]), "output_complete": complete(reasoning), "repetition_score": repetition_score(reasoning), "max_token_run": max_token_run(reasoning)}
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
    summary = {"mode": args.mode, "model": name, "model_path": str(model_path.resolve()), "image_dir": str(args.image_dir.resolve()), "num_images": len(rows), "seed": args.seed, "dtype": args.dtype, "attention": args.attn_implementation, "max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "warmup": args.warmup, "model_load_seconds": load_seconds, "model_allocated_mb": model_memory["allocated"], "model_reserved_mb": model_memory["reserved"], "e2e_mean_s": statistics.mean(e2e), "e2e_median_s": statistics.median(e2e), "e2e_p95_s": percentile(e2e, .95), "generation_mean_s": statistics.mean(gen), "generation_median_s": statistics.median(gen), "generation_p95_s": percentile(gen, .95), "tokens_per_second_mean": statistics.mean(r["tokens_per_second"] for r in rows), "peak_allocated_mb": max(r["peak_allocated_mb"] for r in rows), "peak_reserved_mb": max(r["peak_reserved_mb"] for r in rows), "completion_rate": sum(r["output_complete"] for r in rows)/len(rows), "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "prompt": PROMPT}
    summary_path = Path(str(base) + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Reasoning: {output_log}\nMetrics: {base}_per_image.csv\nSummary: {summary_path}")

if __name__ == "__main__":
    main()
