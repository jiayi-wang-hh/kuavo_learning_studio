#!/usr/bin/env python3
"""Evaluate VLMs on a bimanual toy-to-basket task without running any model at setup time.

Expected filenames (case-insensitive; mp4/mov/mkv/avi):
    rollout01_head.mp4
    rollout01_left.mp4
    rollout01_right.mp4
    ...

The script supports head-only, wrist-only, all individual views, synchronized
three-view input, or all of them. Ground truth is read from a JSON file so that
the nine rollouts can be annotated without editing Python code.

Example:
    python vlm_test/test_vlm_video_joy.py \
      --mode qwen \
      --video-dir /media/data/jiayi/dataset/toy_basket_rollouts \
      --annotations vlm_test/toy_basket_annotations.json \
      --view-mode head+three \
      --fps 2 \
      --max-new-tokens 256
"""

from __future__ import annotations

import os

os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"

import argparse
import csv
import json
import platform
import re
import statistics
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import transformers
from transformers import AutoProcessor


MODE_INFO = {
    "qwen": (
        "Qwen2.5-VL-3B-Instruct",
        "/media/data/jiayi/hf_model/Qwen2.5-VL-3B-Instruct",
        "qwen25_vl_3b",
    ),
    "qwen25_vl_7b": (
        "Qwen2.5-VL-7B-Instruct",
        "/media/data/jiayi/hf_model/Qwen2.5-VL-7B-Instruct",
        "qwen25_vl_7b",
    ),
    "qwen3_vl_4b": (
        "Qwen3-VL-4B-Instruct",
        "/media/data/jiayi/hf_model/Qwen3-VL-4B-Instruct",
        "qwen3_vl_4b",
    ),
    "qwen35_4b": (
        "Qwen3.5-4B",
        "/media/data/jiayi/hf_model/Qwen3.5-4B",
        "qwen35_4b",
    ),
    "qwen35_9b": (
        "Qwen3.5-9B",
        "/media/data/jiayi/hf_model/Qwen3.5-9B",
        "qwen35_9b",
    ),
    "qwen3_8_27b": (
        "Qwen3.8-27B",
        "/media/data/jiayi/hf_model/Qwen3.8-27B",
        "qwen3_8_27b",
    ),
    "robobrain": (
        "RoboBrain2.0-7B",
        "/media/data/jiayi/hf_model/RoboBrain2.0-7B",
        "robobrain2_7b",
    ),
    "robobrain_2.5": (
        "RoboBrain2.5-8B-NV",
        "/media/data/jiayi/hf_model/RoboBrain2.5-8B-NV",
        "robobrain2_5_8b_nv",
    ),
    "cosmos_reason2_32b": (
        "Cosmos-Reason2-32B",
        "/media/data/jiayi/hf_model/Cosmos-Reason2-32B",
        "cosmos_reason2_32b",
    ),
}

FAILURE_TYPES = (
    "NONE",
    "LEFT_GRASP_MISS",
    "RIGHT_GRASP_MISS",
    "BOTH_GRASP_MISS",
    "LEFT_OBJECT_DROP",
    "RIGHT_OBJECT_DROP",
    "BOTH_OBJECT_DROP",
    "OBJECT_OUTSIDE_BASKET",
    "OBJECT_ON_BASKET_EDGE",
    "BIMANUAL_DESYNCHRONIZATION",
    "COLLISION",
    "NO_PROGRESS",
    "MULTIPLE_FAILURES",
    "OTHER",
)

DEFAULT_TASK = (
    "Use both arms to pick up both toys from the table and release both toys "
    "fully inside the available baskets. Either toy may be placed in either basket."
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=tuple(MODE_INFO), required=True)
    p.add_argument("--video-dir", type=Path, default= "/media/data/jiayi/dataset/joy_videos")
    p.add_argument("--annotations", type=Path, default=None, 
                   help="GT JSON")
    p.add_argument(
        "--view-mode",
        choices=("head", "wrists", "individual", "three", "head+three", "all"),
        default="head+three",
        help=(
            "head: head only; wrists: left/right separately; individual: all three "
            "separately; three: synchronized head+left+right; head+three: recommended; "
            "all: individual plus three-view."
        ),
    )
    p.add_argument("--task-instruction", default=DEFAULT_TASK)
    p.add_argument("--model-path", type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/media/data/jiayi/outputs/vlm_bimanual_toy_basket"),
    )
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    p.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    p.add_argument("--device-map", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--run-name")
    p.add_argument("--qwen38-thinking", action="store_true")
    p.add_argument("--qwen35-thinking", action="store_true")
    return p.parse_args()


@dataclass
class Sample:
    sample_id: str
    rollout: str
    view: str
    videos: list[Path]
    gt: dict[str, Any]


VIDEO_PATTERN = re.compile(
    r"^(?:rollout|episode)[_-]?0*(\d+)[_-](head|left|right|left_wrist|right_wrist)"
    r"\.(mp4|mov|mkv|avi)$",
    re.IGNORECASE,
)


def normalize_view(view: str) -> str:
    return {"left_wrist": "left", "right_wrist": "right"}.get(view.lower(), view.lower())


def normalize_rollout(number: str | int) -> str:
    return f"rollout{int(number):02d}"


def discover_videos(video_dir: Path) -> dict[str, dict[str, Path]]:
    root = video_dir.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    grouped: dict[str, dict[str, Path]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        match = VIDEO_PATTERN.match(path.name)
        if not match:
            continue
        rollout = normalize_rollout(match.group(1))
        view = normalize_view(match.group(2))
        if view in grouped.setdefault(rollout, {}):
            raise RuntimeError(f"Duplicate {view} video for {rollout}: {path}")
        grouped[rollout][view] = path.resolve()

    if not grouped:
        raise RuntimeError(
            "No videos found. Expected names such as rollout01_head.mp4, "
            "rollout01_left.mp4, and rollout01_right.mp4."
        )
    return grouped


def load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    annotation_path = path.expanduser().resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    items = raw.get("rollouts", raw)
    if not isinstance(items, dict):
        raise ValueError("Annotations must contain a 'rollouts' object.")

    normalized = {}
    for key, value in items.items():
        match = re.search(r"(\d+)", str(key))
        if not match or not isinstance(value, dict):
            continue
        normalized[normalize_rollout(match.group(1))] = value
    return normalized


def validate_annotation(rollout: str, gt: dict[str, Any]) -> None:
    outcome = str(gt.get("outcome", "")).upper()
    failure = str(gt.get("failure_detected", "")).upper()
    if outcome not in {"SUCCESS", "FAILURE"}:
        raise ValueError(f"{rollout}: outcome must be SUCCESS or FAILURE")
    if failure not in {"YES", "NO"}:
        raise ValueError(f"{rollout}: failure_detected must be YES or NO")
    expected_failure = "YES" if outcome == "FAILURE" else "NO"
    if failure != expected_failure:
        raise ValueError(f"{rollout}: outcome and failure_detected are inconsistent")


def build_samples(grouped, annotations, view_mode: str) -> list[Sample]:
    samples = []
    for rollout in sorted(grouped):
        # In inference-only mode annotations is empty. Keep every discovered
        # rollout and leave its ground-truth fields blank instead of skipping it.
        gt = annotations.get(rollout, {})
        if gt:
            validate_annotation(rollout, gt)
        views = grouped[rollout]

        individual_views: tuple[str, ...] = ()
        if view_mode == "head":
            individual_views = ("head",)
        elif view_mode == "wrists":
            individual_views = ("left", "right")
        elif view_mode in {"individual", "all"}:
            individual_views = ("head", "left", "right")
        elif view_mode == "head+three":
            individual_views = ("head",)

        for view in individual_views:
            if view in views:
                samples.append(Sample(f"{rollout}_{view}", rollout, view, [views[view]], gt))
            else:
                print(f"[WARN] {rollout}: missing {view} view.")

        if view_mode in {"three", "head+three", "all"}:
            missing = [view for view in ("head", "left", "right") if view not in views]
            if missing:
                print(f"[WARN] {rollout}: three-view sample skipped; missing {missing}.")
            else:
                samples.append(
                    Sample(
                        f"{rollout}_head+left+right",
                        rollout,
                        "head+left+right",
                        [views["head"], views["left"], views["right"]],
                        gt,
                    )
                )
    return samples


def build_prompt(task_instruction: str, views: list[str]) -> str:
    view_statement = (
        f"You are given one {views[0].upper()}-camera video."
        if len(views) == 1
        else "You are given synchronized HEAD, LEFT-WRIST, and RIGHT-WRIST videos of the same rollout. Combine evidence across views and do not treat them as separate trials."
    )
    failure_types = ", ".join(FAILURE_TYPES)
    return f"""You are a visual failure monitor for bimanual robot manipulation.

{view_statement}

Task:
{task_instruction}

Success criterion:
- Both toys must be released fully and stably inside a basket at the end.
- Either toy may be placed in either basket.
- Touching, lifting, transporting, or moving a toy alone is not task success.
- Do not infer successful placement from the robot's motion or placement
  attempt alone. Verify the final visible state of each toy separately.
- A toy on the table, held by the gripper, outside a basket, or resting on a
  basket edge is not successfully placed.
- If only one toy is inside a basket at the end, the overall outcome is FAILURE and task_progress is PARTIAL.
- If the final state of either toy cannot be verified visually, return
  UNCERTAIN rather than assuming SUCCESS.

Temporal decision rules:
- A normal approach, open gripper, ongoing grasp attempt, transport motion,
  or unfinished placement is IN_PROGRESS, not a failure by itself.
- Report a failure only after observable evidence appears: a completed grasp
  miss, slip or drop, placement outside a basket, collision, persistent loss
  of progress, or the rollout ending without both toys successfully placed.
- Do not predict failure merely because the input comes from a failure benchmark.
- Use only directly visible evidence. If the relevant interaction or final toy
  state is occluded or ambiguous, use UNCERTAIN.
- For a full video, outcome describes the final task result.
- first_failure_time_s is the earliest time at which a visible failure becomes
  observable, not the end of the video.
- Do not reveal chain-of-thought or narrate frames.

failure_type must be exactly one of:
{failure_types}

Return exactly one valid JSON object and no other text:
{{
  "outcome": "SUCCESS | FAILURE | UNCERTAIN",
  "task_progress": "NOT_STARTED | IN_PROGRESS | PARTIAL | COMPLETE | FAILED | UNCERTAIN",
  "left_toy_final_state": "ON_TABLE | HELD | IN_BASKET | ON_BASKET_EDGE | NOT_VISIBLE | UNCERTAIN",
  "right_toy_final_state": "ON_TABLE | HELD | IN_BASKET | ON_BASKET_EDGE | NOT_VISIBLE | UNCERTAIN",
  "failure_detected": "YES | NO | UNCERTAIN",
  "first_failure_time_s": null,
  "failure_type": "one allowed label",
  "failed_side": "LEFT | RIGHT | BOTH | NONE | UNCERTAIN",
  "evidence": "one short sentence describing only visible evidence",
  "recovery_needed": "YES | NO | UNCERTAIN",
  "recovery_action": "one short executable subtask or null",
  "recovery_exit_condition": "one visually verifiable condition or null"
}}

Consistency rules:
- SUCCESS => task_progress=COMPLETE, failure_detected=NO, failure_type=NONE, failed_side=NONE, recovery_needed=NO.
- FAILURE => failure_detected=YES and recovery_needed=YES unless recovery is unsafe.
- UNCERTAIN => do not invent a failure time or failure type.
- Use null for first_failure_time_s when no failure time is visible.
- Keep the complete output below 220 tokens.
"""


def model_class(mode: str):
    if mode in {"qwen", "qwen25_vl_7b"}:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    if mode == "qwen3_vl_4b":
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    if mode == "robobrain_2.5":
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText
    if mode == "cosmos_reason2_32b":
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    try:
        from transformers import AutoModelForMultimodalLM
        return AutoModelForMultimodalLM
    except ImportError as exc:
        raise ImportError(
            f"{mode} requires a recent Transformers version with AutoModelForMultimodalLM."
        ) from exc


def input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def load_model(path: Path, args):
    model_name = MODE_INFO[args.mode][0]
    processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True, use_fast=False)
    cls = model_class(args.mode)
    dtype = getattr(torch, args.dtype)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": args.attn_implementation,
        "device_map": args.device_map,
    }
    if args.mode in {"cosmos_reason2_32b", "qwen3_8_27b", "qwen3_vl_4b", "qwen35_4b", "qwen35_9b"}:
        kwargs["dtype"] = dtype
    else:
        kwargs["torch_dtype"] = dtype
    print(f"Loading {model_name} from {path}")
    model = cls.from_pretrained(str(path), **kwargs).eval()
    provenance = [
        f"Mode: {args.mode}", f"Model: {model_name}", f"Checkpoint: {path}",
        f"dtype: {args.dtype}", f"attention: {args.attn_implementation}",
        f"device_map: {args.device_map}", f"Backend: {cls.__name__}",
    ]
    return model, processor, provenance


def native_video_mode(mode: str) -> bool:
    return mode in {
        "qwen3_8_27b", "qwen3_vl_4b", "qwen35_4b", "qwen35_9b", "cosmos_reason2_32b"
    }


def build_messages(sample: Sample, prompt: str, fps: float, native: bool):
    labels = sample.view.split("+")
    content = []
    for index, (label, path) in enumerate(zip(labels, sample.videos), 1):
        content.append({"type": "text", "text": f"VIDEO {index}: {label.upper()} CAMERA"})
        content.append(
            {
                "type": "video",
                "video": str(path) if native else f"file://{path}",
                "fps": fps,
            }
        )
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def thinking_kwargs(args) -> dict[str, Any]:
    if args.mode == "qwen3_8_27b":
        return {"enable_thinking": args.qwen38_thinking, "preserve_thinking": False}
    if args.mode in {"qwen35_4b", "qwen35_9b"}:
        return {"enable_thinking": args.qwen35_thinking}
    return {}


def process_qwen_utils(processor, messages, template_kwargs=None):
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise ImportError("Install qwen-vl-utils for video inference.") from exc
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **(template_kwargs or {})
    )
    result = process_vision_info(messages)
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("Unexpected result from process_vision_info().")
    return processor(
        text=[text], images=result[0], videos=result[1], padding=True, return_tensors="pt"
    )


def prepare_inputs(model, processor, sample: Sample, args, prompt: str):
    native = native_video_mode(args.mode)
    messages = build_messages(sample, prompt, args.fps, native)
    template_kwargs = thinking_kwargs(args)

    if native:
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"fps": args.fps},
                **template_kwargs,
            )
            if any(
                key in inputs
                for key in ("pixel_values", "pixel_values_videos", "video_grid_thw", "image_grid_thw")
            ):
                return inputs.to(input_device(model))
        except Exception as exc:
            print(f"[{args.mode}] native video processing failed; using qwen_vl_utils: {exc}")

    fallback = deepcopy(messages)
    for message in fallback:
        for item in message.get("content", []):
            if item.get("type") == "video":
                value = str(item["video"])
                if not value.startswith(("file://", "http://", "https://")):
                    item["video"] = f"file://{value}"
    inputs = process_qwen_utils(processor, fallback, template_kwargs)
    return inputs.to(input_device(model))


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_memory_mb():
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "peak_allocated": 0.0, "peak_reserved": 0.0}
    scale = 1024 ** 2
    return {
        "allocated": torch.cuda.memory_allocated() / scale,
        "peak_allocated": torch.cuda.max_memory_allocated() / scale,
        "peak_reserved": torch.cuda.max_memory_reserved() / scale,
    }


def generate(model, processor, sample: Sample, args, prompt: str):
    total_start = time.perf_counter()
    inputs = prepare_inputs(model, processor, sample, args, prompt)
    sync_cuda()
    preprocess_seconds = time.perf_counter() - total_start
    kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": args.temperature > 0}
    if args.temperature > 0:
        kwargs["temperature"] = args.temperature
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **kwargs)
    sync_cuda()
    generation_seconds = time.perf_counter() - start
    input_tokens = int(inputs.input_ids.shape[1])
    generated_ids = output[:, input_tokens:]
    text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    output_tokens = int(generated_ids.shape[1])
    return text, {
        "preprocess_seconds": preprocess_seconds,
        "generation_seconds": generation_seconds,
        "end_to_end_seconds": time.perf_counter() - total_start,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": output_tokens / generation_seconds if generation_seconds else 0.0,
    }


def extract_json(text: str) -> tuple[dict[str, Any], bool]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    try:
        value = json.loads(cleaned)
        return (value, isinstance(value, dict))
    except json.JSONDecodeError:
        starts = [match.start() for match in re.finditer(r"\{", cleaned)]
        for start in reversed(starts):
            try:
                value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
                if isinstance(value, dict):
                    return value, False
            except json.JSONDecodeError:
                pass
    return {}, False


def norm(value: Any) -> str:
    return str(value or "").strip().upper()


def parse_time(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def accuracy(rows, field):
    values = [bool(row[field]) for row in rows if row[field] != ""]
    return sum(values) / len(values) if values else None


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_name, default_path, short_name = MODE_INFO[args.mode]
    model_path = (args.model_path or Path(default_path)).expanduser().resolve()
    if not model_path.is_dir():
        raise NotADirectoryError(f"Model path does not exist: {model_path}")

    annotations = (
        load_annotations(args.annotations)
        if args.annotations is not None
        else {}
    )
    grouped = discover_videos(args.video_dir)
    samples = build_samples(grouped, annotations, args.view_mode)
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise RuntimeError("No samples were created. Check filenames, annotations, and view mode.")

    print("Discovered samples:")
    for sample in samples:
        print(f"  {sample.sample_id}: {', '.join(path.name for path in sample.videos)}")

    output_root = args.output_dir.expanduser().resolve()
    logs_dir, metrics_dir = output_root / "logs", output_root / "metrics"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    base = f"{short_name}_{args.view_mode}_{suffix}"
    log_path = logs_dir / f"{base}.log"
    csv_path = metrics_dir / f"{base}_per_sample.csv"
    summary_path = metrics_dir / f"{base}_summary.json"

    torch.cuda.empty_cache()
    reset_peak_memory()
    sync_cuda()
    load_start = time.perf_counter()
    model, processor, provenance = load_model(model_path, args)
    sync_cuda()
    model_load_seconds = time.perf_counter() - load_start
    load_memory = gpu_memory_mb()

    if args.warmup:
        warm = samples[0]
        warm_prompt = build_prompt(args.task_instruction, warm.view.split("+"))
        for index in range(args.warmup):
            print(f"Warm-up {index + 1}/{args.warmup}: {warm.sample_id}")
            generate(model, processor, warm, args, warm_prompt)

    log_path.write_text("", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    required_keys = {
        "outcome", "task_progress", "left_toy_final_state", "right_toy_final_state",
        "failure_detected", "first_failure_time_s", "failure_type", "failed_side",
        "evidence", "recovery_needed", "recovery_action", "recovery_exit_condition",
    }

    for index, sample in enumerate(samples, 1):
        prompt = build_prompt(args.task_instruction, sample.view.split("+"))
        reset_peak_memory()
        before = gpu_memory_mb()
        output, timing = generate(model, processor, sample, args, prompt)
        after = gpu_memory_mb()
        parsed, json_only = extract_json(output)

        gt_outcome = norm(sample.gt.get("outcome"))
        gt_failure = norm(sample.gt.get("failure_detected"))
        gt_type = norm(sample.gt.get("failure_type"))
        gt_time = parse_time(sample.gt.get("first_failure_time_s"))

        pred_outcome = norm(parsed.get("outcome"))
        pred_failure = norm(parsed.get("failure_detected"))
        pred_type = norm(parsed.get("failure_type"))
        pred_time = parse_time(parsed.get("first_failure_time_s"))

        outcome_correct = (
            pred_outcome == gt_outcome
            if gt_outcome
            else ""
        )

        failure_detection_correct = (
            pred_failure == gt_failure
            if gt_failure
            else ""
        )

        failure_type_correct = (
            pred_type == gt_type
            if gt_type
            else ""
        )

        time_error = (
            abs(pred_time - gt_time) if pred_time is not None and gt_time is not None else None
        )
        row = {
            "mode": args.mode,
            "model": model_name,
            "sample_id": sample.sample_id,
            "rollout": sample.rollout,
            "view": sample.view,
            "videos": " | ".join(str(path) for path in sample.videos),
            "gt_outcome": gt_outcome,
            "pred_outcome": pred_outcome,
            "outcome_correct": outcome_correct,
            "gt_failure_detected": gt_failure,
            "pred_failure_detected": pred_failure,
            "failure_detection_correct": failure_detection_correct,
            "gt_failure_type": gt_type,
            "pred_failure_type": pred_type,
            "failure_type_correct": failure_type_correct,
            "gt_first_failure_time_s": gt_time if gt_time is not None else "",
            "pred_first_failure_time_s": pred_time if pred_time is not None else "",
            "failure_time_absolute_error_s": time_error if time_error is not None else "",
            "pred_task_progress": norm(parsed.get("task_progress")),
            "pred_left_toy_final_state": norm(parsed.get("left_toy_final_state")),
            "pred_right_toy_final_state": norm(parsed.get("right_toy_final_state")),
            "pred_failed_side": norm(parsed.get("failed_side")),
            "evidence": str(parsed.get("evidence") or ""),
            "recovery_needed": norm(parsed.get("recovery_needed")),
            "recovery_action": parsed.get("recovery_action"),
            "recovery_exit_condition": parsed.get("recovery_exit_condition"),
            "json_only": json_only,
            "output_complete": required_keys.issubset(parsed),
            "fps": args.fps,
            **timing,
            "gpu_allocated_before_mb": before["allocated"],
            "peak_allocated_mb": after["peak_allocated"],
            "peak_reserved_mb": after["peak_reserved"],
        }
        rows.append(row)
        print(
            f"[{index}/{len(samples)}] {sample.sample_id}: GT={gt_outcome}/{gt_failure}, "
            f"pred={pred_outcome}/{pred_failure}, time={timing['end_to_end_seconds']:.2f}s"
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                "=" * 100 + "\n"
                + f"Sample: {sample.sample_id}\nVideos: {row['videos']}\n"
                + f"Ground truth: {json.dumps(sample.gt, ensure_ascii=False)}\n"
                + "\n".join(provenance) + f"\nFPS: {args.fps}\nTask: {args.task_instruction}\n\n"
                + f"=== MODEL OUTPUT ===\n{output}\n\n"
                + f"=== PARSED METRICS ===\n{json.dumps(row, indent=2, ensure_ascii=False)}\n\n"
            )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    failure_rows = [row for row in rows if row["gt_failure_detected"] == "YES"]
    success_rows = [row for row in rows if row["gt_failure_detected"] == "NO"]
    time_errors = [
        float(row["failure_time_absolute_error_s"])
        for row in failure_rows
        if row["failure_time_absolute_error_s"] != ""
    ]
    by_view = {}
    for view in sorted({row["view"] for row in rows}):
        subset = [row for row in rows if row["view"] == view]
        by_view[view] = {
            "num_samples": len(subset),
            "outcome_accuracy": accuracy(subset, "outcome_correct"),
            "failure_detection_accuracy": accuracy(subset, "failure_detection_correct"),
        }

    summary = {
        "mode": args.mode,
        "model": model_name,
        "model_path": str(model_path),
        "video_dir": str(args.video_dir.expanduser().resolve()),
        "annotations": (
            str(args.annotations.expanduser().resolve())
            if args.annotations is not None
            else None
        ),
        "task_instruction": args.task_instruction,
        "view_mode": args.view_mode,
        "fps": args.fps,
        "num_rollouts": len({row["rollout"] for row in rows}),
        "num_samples": len(rows),
        "outcome_accuracy": accuracy(rows, "outcome_correct"),
        "failure_detection_accuracy": accuracy(rows, "failure_detection_correct"),
        "failure_recall": (
            sum(row["pred_failure_detected"] == "YES" for row in failure_rows) / len(failure_rows)
            if failure_rows else None
        ),
        "success_specificity": (
            sum(row["pred_failure_detected"] == "NO" for row in success_rows) / len(success_rows)
            if success_rows else None
        ),
        "failure_type_accuracy": accuracy(failure_rows, "failure_type_correct"),
        "failure_time_mae_s": statistics.mean(time_errors) if time_errors else None,
        "json_only_rate": sum(row["json_only"] for row in rows) / len(rows),
        "completion_rate": sum(row["output_complete"] for row in rows) / len(rows),
        "mean_end_to_end_seconds": statistics.mean(row["end_to_end_seconds"] for row in rows),
        "median_end_to_end_seconds": statistics.median(row["end_to_end_seconds"] for row in rows),
        "p95_end_to_end_seconds": percentile([row["end_to_end_seconds"] for row in rows], 0.95),
        "mean_generation_seconds": statistics.mean(row["generation_seconds"] for row in rows),
        "mean_tokens_per_second": statistics.mean(row["tokens_per_second"] for row in rows),
        "by_view": by_view,
        "model_load_seconds": model_load_seconds,
        "model_allocated_after_load_mb": load_memory["allocated"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available() else [],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE\nLog: {log_path}\nCSV: {csv_path}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
