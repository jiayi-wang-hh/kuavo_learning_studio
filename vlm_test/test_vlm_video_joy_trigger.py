#!/usr/bin/env python3
"""Causal sliding-window VLM evaluation for a stage-1 interruption trigger.

Unlike a full-video benchmark, every decision sees only a short video window
ending at the current simulated observation time. The detector only decides
whether the current VLA execution should continue, be interrupted, or needs
another observation. It does not diagnose failure type or plan recovery.

Expected filenames (case-insensitive):
    rollout01_head.mp4
    rollout01_left.mp4
    rollout01_right.mp4

Example:
    python vlm_test/test_vlm_video_joy_trigger_only.py \
      --mode qwen25_vl_7b \
      --video-dir /media/data/jiayi/dataset/joy_videos \
      --annotations vlm_test/toy_annotations.json \
      --view-mode head+three \
      --window-seconds 3 \
      --stride-seconds 1 \
      --fps 2

Requirements:
    ffmpeg and ffprobe must be available on PATH.
"""

from __future__ import annotations

import os
import importlib.util
import warnings

# Prefer decord for Qwen video decoding so we do not rely on torchvision.io,
# whose video APIs are deprecated in torchvision >= 0.22.
#
# If decord is not installed, keep torchvision as a compatibility fallback,
# but silence only torchvision's video deprecation warning.
if importlib.util.find_spec("decord") is not None:
    os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"
else:
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchvision")
    warnings.filterwarnings(
        "ignore",
        message=r"The video decoding and encoding capabilities of torchvision are deprecated.*",
        category=UserWarning,
        module=r"torchvision\.io\._video_deprecation_warning",
    )

import argparse
import csv
import json
import platform
import re
import shutil
import statistics
import subprocess
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

DEFAULT_TASK = (
    "Use both arms to pick up both toys from the table and release both toys "
    "fully inside the available baskets. Either toy may be placed in either basket."
)

VIDEO_PATTERN = re.compile(
    r"^(?:rollout|episode)[_-]?0*(\d+)[_-]"
    r"(head|left|right|left_wrist|right_wrist)\.(mp4|mov|mkv|avi)$",
    re.IGNORECASE,
)


@dataclass
class Stream:
    stream_id: str
    rollout: str
    view: str
    videos: list[Path]
    gt: dict[str, Any]


@dataclass
class Window:
    start_s: float
    end_s: float
    videos: list[Path]


@dataclass
class MonitorMemory:
    """Small, structured memory carried between causal windows."""

    left_toy_state: str = "UNKNOWN"
    right_toy_state: str = "UNKNOWN"
    left_candidate_state: str = "UNKNOWN"
    right_candidate_state: str = "UNKNOWN"
    left_candidate_count: int = 0
    right_candidate_count: int = 0
    left_ever_held: bool = False
    right_ever_held: bool = False
    task_complete_streak: int = 0
    task_complete_confirmed: bool = False
    task_complete_time_s: Optional[float] = None
    last_confirmed_time_s: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "left_toy_state": self.left_toy_state,
            "right_toy_state": self.right_toy_state,
            "left_candidate_state": self.left_candidate_state,
            "right_candidate_state": self.right_candidate_state,
            "left_candidate_count": self.left_candidate_count,
            "right_candidate_count": self.right_candidate_count,
            "left_ever_held": self.left_ever_held,
            "right_ever_held": self.right_ever_held,
            "task_complete_streak": self.task_complete_streak,
            "task_complete_confirmed": self.task_complete_confirmed,
            "task_complete_time_s": self.task_complete_time_s,
            "last_confirmed_time_s": self.last_confirmed_time_s,
        }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=tuple(MODE_INFO), required=True)
    p.add_argument(
        "--video-dir",
        type=Path,
        default=Path("/media/data/jiayi/dataset/joy_videos"),
    )
    p.add_argument("--annotations", 
                   type=Path, 
                   default="vlm_test/toy_annotations.json", 
                   help="Optional GT JSON"
    )
    p.add_argument(
        "--view-mode",
        choices=("head", "wrists", "individual", "three", "head+three", "all"),
        default="head",
    )
    p.add_argument("--task-instruction", default=DEFAULT_TASK)
    p.add_argument("--model-path", type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/media/data/jiayi/outputs/vlm_bimanual_toy_trigger_only"),
    )
    p.add_argument("--window-seconds", type=float, default=3.0)
    p.add_argument("--stride-seconds", type=float, default=1.0)
    p.add_argument("--start-time", type=float, default=0.0)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--detection-max-new-tokens", type=int, default=128)
    p.add_argument(
        "--state-confirmations",
        type=int,
        default=2,
        help="Consecutive identical observations required to confirm a toy state.",
    )
    p.add_argument(
        "--interrupt-confirmations",
        type=int,
        default=2,
        help="Consecutive guarded INTERRUPT predictions required to stop.",
    )
    p.add_argument(
        "--task-complete-confirmations",
        type=int,
        default=2,
        help="Consecutive credible both-IN_BASKET observations required to lock completion.",
    )
    p.add_argument(
        "--timely-threshold-seconds",
        type=float,
        default=2.0,
        help="Maximum post-GT delay counted as a timely trigger.",
    )
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
    p.add_argument("--limit-rollouts", type=int)
    p.add_argument("--max-windows-per-rollout", type=int)
    p.add_argument("--run-name")
    p.add_argument("--qwen38-thinking", action="store_true")
    p.add_argument("--qwen35-thinking", action="store_true")
    p.add_argument(
        "--stop-on-interrupt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop later windows after the first predicted INTERRUPT.",
    )
    p.add_argument(
        "--stop-on-task-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop monitoring after task completion is temporally confirmed.",
    )
    p.add_argument(
        "--overwrite-window-cache",
        action="store_true",
        help="Recreate cached ffmpeg clips.",
    )
    args = p.parse_args()
    if args.window_seconds <= 0 or args.stride_seconds <= 0 or args.fps <= 0:
        p.error("window-seconds, stride-seconds, and fps must be positive")
    if (
        args.state_confirmations < 1
        or args.interrupt_confirmations < 1
        or args.task_complete_confirmations < 1
    ):
        p.error("all confirmation counts must be >= 1")
    if args.timely_threshold_seconds < 0:
        p.error("timely-threshold-seconds must be non-negative")
    return args


def normalize_view(view: str) -> str:
    return {"left_wrist": "left", "right_wrist": "right"}.get(
        view.lower(), view.lower()
    )


def normalize_rollout(number: str | int) -> str:
    return f"rollout{int(number):02d}"


def discover_videos(video_dir: Path) -> dict[str, dict[str, Path]]:
    root = video_dir.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    grouped: dict[str, dict[str, Path]] = {}
    # Do not cap the filesystem walk here.  A cap silently drops rollouts or
    # camera views depending on lexical path order, making results depend on
    # directory layout rather than the requested dataset.
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
        raise RuntimeError("No rolloutXX_head/left/right videos were found.")
    return grouped


def load_annotations(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    items = raw.get("rollouts", raw)
    if not isinstance(items, dict):
        raise ValueError("Annotations must contain a 'rollouts' object.")
    result = {}
    for key, value in items.items():
        match = re.search(r"(\d+)", str(key))
        if match and isinstance(value, dict):
            result[normalize_rollout(match.group(1))] = value
    return result


def validate_annotation(rollout: str, gt: dict[str, Any]):
    outcome = norm(gt.get("outcome"))
    failure = norm(gt.get("failure_detected"))
    if outcome not in {"SUCCESS", "FAILURE"}:
        raise ValueError(f"{rollout}: outcome must be SUCCESS or FAILURE")
    if failure not in {"YES", "NO"}:
        raise ValueError(f"{rollout}: failure_detected must be YES or NO")
    if failure != ("YES" if outcome == "FAILURE" else "NO"):
        raise ValueError(f"{rollout}: outcome and failure_detected disagree")
    if failure == "YES" and parse_time(gt.get("first_failure_time_s")) is None:
        raise ValueError(f"{rollout}: failure requires first_failure_time_s")


def build_streams(grouped, annotations, view_mode: str) -> list[Stream]:
    streams: list[Stream] = []
    for rollout in sorted(grouped):
        gt = annotations.get(rollout, {})
        if gt:
            validate_annotation(rollout, gt)
        views = grouped[rollout]
        individual: tuple[str, ...] = ()
        if view_mode == "head":
            individual = ("head",)
        elif view_mode == "wrists":
            individual = ("left", "right")
        elif view_mode in {"individual", "all"}:
            individual = ("head", "left", "right")
        elif view_mode == "head+three":
            individual = ("head",)
        for view in individual:
            if view in views:
                streams.append(Stream(f"{rollout}_{view}", rollout, view, [views[view]], gt))
            else:
                print(f"[WARN] {rollout}: missing {view} view")
        if view_mode in {"three", "head+three", "all"}:
            missing = [v for v in ("head", "left", "right") if v not in views]
            if missing:
                print(f"[WARN] {rollout}: skip three-view stream; missing {missing}")
            else:
                streams.append(
                    Stream(
                        f"{rollout}_head+left+right",
                        rollout,
                        "head+left+right",
                        [views["head"], views["left"], views["right"]],
                        gt,
                    )
                )
    return streams


def require_ffmpeg():
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"{executable} was not found on PATH")


def video_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def decision_times(duration: float, start: float, window: float, stride: float):
    first = min(duration, max(start + window, window))
    values = []
    current = first
    while current < duration - 1e-6:
        values.append(round(current, 6))
        current += stride
    if duration > start and (not values or abs(values[-1] - duration) > 1e-6):
        values.append(round(duration, 6))
    return values


def make_clip(
    source: Path,
    destination: Path,
    start_s: float,
    end_s: float,
    overwrite: bool,
):
    if destination.is_file() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(source),
        "-t",
        f"{end_s - start_s:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(destination),
    ]
    subprocess.run(command, check=True)


def build_window(stream: Stream, end_s: float, args, cache_root: Path) -> Window:
    start_s = max(args.start_time, end_s - args.window_seconds)
    clipped = []
    labels = stream.view.split("+")
    for label, source in zip(labels, stream.videos):
        name = f"{int(round(start_s * 1000)):08d}_{int(round(end_s * 1000)):08d}.mp4"
        destination = cache_root / stream.rollout / label / name
        make_clip(source, destination, start_s, end_s, args.overwrite_window_cache)
        clipped.append(destination.resolve())
    return Window(start_s, end_s, clipped)


def view_statement(views: list[str]) -> str:
    if len(views) == 1:
        return f"You are given one {views[0].upper()}-camera video window."
    return (
        "You are given synchronized HEAD, LEFT-WRIST, and RIGHT-WRIST video "
        "windows from the same rollout. Combine their evidence."
    )


def build_detection_prompt(
    task: str,
    views: list[str],
    start_s: float,
    end_s: float,
    previous_state: MonitorMemory,
):
    memory_summary = (
        f"left={previous_state.left_toy_state}, "
        f"right={previous_state.right_toy_state}, "
        f"left_ever_held={previous_state.left_ever_held}, "
        f"right_ever_held={previous_state.right_ever_held}"
    )
    return f"""You are an online visual event detector for bimanual manipulation.

{view_statement(views)} The window covers {start_s:.1f} to {end_s:.1f} seconds.
You may use only observations up to {end_s:.1f} seconds. Never infer future outcomes.

Task:
{task}

Earlier confirmed memory (may be incomplete): {memory_summary}

First inspect the temporal motion in this video window and select exactly one
visible_event:
- NONE: no completed failure event is directly visible.
- GRASP_MISS: a gripper completes a grasp/lift attempt without acquiring its toy.
- OBJECT_DROP: a toy visibly leaves a gripper and falls outside a basket.
- RELEASE_OUTSIDE_BASKET: a gripper visibly releases a toy outside a basket or
  leaves it resting on a basket edge.
- COLLISION: visible unsafe contact that requires stopping.
- AMBIGUOUS: the relevant action outcome is occluded or the event is unclear.

Set monitor_state mechanically from visible_event:
- NONE -> NORMAL
- AMBIGUOUS -> UNCERTAIN
- GRASP_MISS, OBJECT_DROP, RELEASE_OUTSIDE_BASKET, or COLLISION -> INTERRUPT

Important:
- Toy position alone is not a failure event. A toy on the table during approach,
  grasp, or early lift is normal.
- An action still in progress is not GRASP_MISS.
- Task completion is NONE/NORMAL, never INTERRUPT.
- Do not infer an event from the expected task outcome or from memory.
- Evidence must name the motion or action outcome visible in this window. For
  NONE, briefly describe the ongoing normal action. For AMBIGUOUS, name what is
  occluded.
- Return only the JSON object; no narration or reasoning outside it.

toy state: UNKNOWN | ON_TABLE | HELD | IN_BASKET | ON_BASKET_EDGE | NOT_VISIBLE
action stage: APPROACH | GRASP | LIFT | TRANSPORT | PLACE | UNKNOWN

{{
  "monitor_state": "NORMAL | INTERRUPT | UNCERTAIN",
  "visible_event": "NONE | GRASP_MISS | OBJECT_DROP | RELEASE_OUTSIDE_BASKET | COLLISION | AMBIGUOUS",
  "action_stage": "one allowed action stage",
  "left_toy_state": "one allowed toy state",
  "right_toy_state": "one allowed toy state",
  "evidence": "one short directly visible observation"
}}

The evidence must describe only directly visible evidence.
"""


CONFIRMABLE_TOY_STATES = {
    "ON_TABLE",
    "HELD",
    "IN_BASKET",
    "ON_BASKET_EDGE",
}

VISIBLE_FAILURE_EVENTS = {
    "GRASP_MISS",
    "OBJECT_DROP",
    "RELEASE_OUTSIDE_BASKET",
    "COLLISION",
}

EXPECTED_MONITOR_STATE_BY_EVENT = {
    "NONE": "NORMAL",
    "AMBIGUOUS": "UNCERTAIN",
    **{event: "INTERRUPT" for event in VISIBLE_FAILURE_EVENTS},
}


def has_invalid_toy_transition(
    detection: dict[str, Any], memory: MonitorMemory
) -> bool:
    """Return whether an IN_BASKET observation skips all held history."""
    return any(
        memory_state in {"UNKNOWN", "ON_TABLE"}
        and not ever_held
        and norm(detection.get(f"{side}_toy_state")) == "IN_BASKET"
        for side, memory_state, ever_held in (
            ("left", memory.left_toy_state, memory.left_ever_held),
            ("right", memory.right_toy_state, memory.right_ever_held),
        )
    )


def update_monitor_memory(
    memory: MonitorMemory,
    detection: dict[str, Any],
    end_s: float,
    confirmations: int,
) -> MonitorMemory:
    """Confirm toy states only after repeated identical observations.

    NOT_VISIBLE, UNKNOWN and malformed values do not erase an earlier confirmed
    state.  UNCERTAIN windows also cannot modify memory.
    """
    if norm(detection.get("monitor_state")) == "UNCERTAIN":
        return memory

    changed = False
    for side in ("left", "right"):
        observed = norm(detection.get(f"{side}_toy_state"))
        if observed not in CONFIRMABLE_TOY_STATES:
            continue

        candidate_attr = f"{side}_candidate_state"
        count_attr = f"{side}_candidate_count"
        confirmed_attr = f"{side}_toy_state"
        ever_held_attr = f"{side}_ever_held"

        if observed == getattr(memory, candidate_attr):
            setattr(memory, count_attr, getattr(memory, count_attr) + 1)
        else:
            setattr(memory, candidate_attr, observed)
            setattr(memory, count_attr, 1)

        if getattr(memory, count_attr) >= confirmations:
            if getattr(memory, confirmed_attr) != observed:
                setattr(memory, confirmed_attr, observed)
                changed = True
            if observed == "HELD":
                setattr(memory, ever_held_attr, True)

    if changed:
        memory.last_confirmed_time_s = end_s
    return memory


def apply_trigger_guards(
    detection: dict[str, Any],
    memory: MonitorMemory,
) -> tuple[str, str]:
    """Derive the trigger from a directly visible event, not object position."""
    event = norm(detection.get("visible_event"))
    left = norm(detection.get("left_toy_state"))
    right = norm(detection.get("right_toy_state"))

    credible_task_complete = (
        left == "IN_BASKET"
        and right == "IN_BASKET"
        and not has_invalid_toy_transition(detection, memory)
    )
    if credible_task_complete:
        return "NORMAL", "TASK_COMPLETE_NOT_FAILURE"

    if event == "AMBIGUOUS":
        return "UNCERTAIN", "EVENT_AMBIGUOUS"
    if event in VISIBLE_FAILURE_EVENTS:
        return "INTERRUPT", f"VISIBLE_{event}"
    if event != "NONE":
        return "UNCERTAIN", "INVALID_OR_MISSING_EVENT"
    return "NORMAL", "NO_VISIBLE_FAILURE_EVENT"


def update_task_completion(
    memory: MonitorMemory,
    detection: dict[str, Any],
    guard_reason: str,
    end_s: float,
    confirmations: int,
) -> bool:
    """Latch credible task completion as an absorbing terminal state."""
    if memory.task_complete_confirmed:
        return True

    both_in_basket = (
        norm(detection.get("left_toy_state")) == "IN_BASKET"
        and norm(detection.get("right_toy_state")) == "IN_BASKET"
    )
    credible_completion = (
        both_in_basket and guard_reason == "TASK_COMPLETE_NOT_FAILURE"
    )
    memory.task_complete_streak = (
        memory.task_complete_streak + 1 if credible_completion else 0
    )

    if memory.task_complete_streak >= confirmations:
        memory.task_complete_confirmed = True
        memory.task_complete_time_s = end_s

    return memory.task_complete_confirmed


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
        raise ImportError(f"{mode} requires a newer Transformers version") from exc


def load_model(path: Path, args):
    name = MODE_INFO[args.mode][0]
    processor = AutoProcessor.from_pretrained(
        str(path), trust_remote_code=True, use_fast=False
    )
    cls = model_class(args.mode)
    dtype = getattr(torch, args.dtype)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": args.attn_implementation,
        "device_map": args.device_map,
    }
    if args.mode in {
        "cosmos_reason2_32b",
        "qwen3_8_27b",
        "qwen3_vl_4b",
        "qwen35_4b",
        "qwen35_9b",
    }:
        kwargs["dtype"] = dtype
    else:
        kwargs["torch_dtype"] = dtype
    print(f"Loading {name} from {path}")
    model = cls.from_pretrained(str(path), **kwargs).eval()
    provenance = [
        f"Mode: {args.mode}",
        f"Model: {name}",
        f"Checkpoint: {path}",
        f"dtype: {args.dtype}",
        f"attention: {args.attn_implementation}",
        f"device_map: {args.device_map}",
        f"Backend: {cls.__name__}",
    ]
    return model, processor, provenance


def native_video_mode(mode: str) -> bool:
    # Qwen3.5's native Transformers video path imports TorchCodec.  The current
    # runtime uses PyTorch 2.9.1 and an ABI-incompatible TorchCodec build, while
    # the qwen_vl_utils + decord path works with the same processor/model.
    # Route Qwen3.5 directly to that path instead of raising and printing the
    # same TorchCodec loader error for every window before falling back.
    return mode in {
        "qwen3_8_27b",
        "qwen3_vl_4b",
        "cosmos_reason2_32b",
    }


def thinking_kwargs(args):
    if args.mode == "qwen3_8_27b":
        return {"enable_thinking": args.qwen38_thinking, "preserve_thinking": False}
    if args.mode in {"qwen35_4b", "qwen35_9b"}:
        return {"enable_thinking": args.qwen35_thinking}
    return {}


def input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def build_messages(view: str, videos: list[Path], prompt: str, fps: float, native: bool):
    content = []
    for index, (label, path) in enumerate(zip(view.split("+"), videos), 1):
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


def process_qwen_utils(processor, messages, template_kwargs):
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise ImportError("Install qwen-vl-utils for video inference") from exc
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    result = process_vision_info(messages)
    return processor(
        text=[text],
        images=result[0],
        videos=result[1],
        padding=True,
        return_tensors="pt",
    )


def prepare_inputs(model, processor, view, videos, prompt, args):
    native = native_video_mode(args.mode)
    messages = build_messages(view, videos, prompt, args.fps, native)
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
                for key in (
                    "pixel_values",
                    "pixel_values_videos",
                    "video_grid_thw",
                    "image_grid_thw",
                )
            ):
                return inputs.to(input_device(model))
        except Exception as exc:
            print(f"[{args.mode}] native processing failed; fallback: {exc}")
    fallback = deepcopy(messages)
    for message in fallback:
        for item in message.get("content", []):
            if item.get("type") == "video":
                value = str(item["video"])
                if not value.startswith(("file://", "http://", "https://")):
                    item["video"] = f"file://{value}"
    return process_qwen_utils(processor, fallback, template_kwargs).to(input_device(model))


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_memory_mb():
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "peak_allocated": 0.0, "peak_reserved": 0.0}
    scale = 1024**2
    return {
        "allocated": torch.cuda.memory_allocated() / scale,
        "peak_allocated": torch.cuda.max_memory_allocated() / scale,
        "peak_reserved": torch.cuda.max_memory_reserved() / scale,
    }


def generate(model, processor, view, videos, prompt, args, max_new_tokens):
    total_start = time.perf_counter()
    inputs = prepare_inputs(model, processor, view, videos, prompt, args)
    sync_cuda()
    preprocess_seconds = time.perf_counter() - total_start
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        kwargs["temperature"] = args.temperature
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **kwargs)
    sync_cuda()
    generation_seconds = time.perf_counter() - start
    input_tokens = int(inputs.input_ids.shape[1])
    generated = output[:, input_tokens:]
    text = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return text, {
        "preprocess_seconds": preprocess_seconds,
        "generation_seconds": generation_seconds,
        "end_to_end_seconds": time.perf_counter() - total_start,
        "input_tokens": input_tokens,
        "output_tokens": int(generated.shape[1]),
        "tokens_per_second": (
            int(generated.shape[1]) / generation_seconds if generation_seconds else 0.0
        ),
    }


def extract_json(text: str):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    try:
        value = json.loads(cleaned)
        return (value if isinstance(value, dict) else {}, isinstance(value, dict))
    except json.JSONDecodeError:
        for start in reversed([m.start() for m in re.finditer(r"\{", cleaned)]):
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


def mean_or_none(values):
    values = [float(value) for value in values if value is not None]
    return statistics.mean(values) if values else None


def main():
    args = parse_args()
    require_ffmpeg()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_name, default_path, short_name = MODE_INFO[args.mode]
    model_path = (args.model_path or Path(default_path)).expanduser().resolve()
    if not model_path.is_dir():
        raise NotADirectoryError(f"Model path does not exist: {model_path}")

    annotations = load_annotations(args.annotations)
    grouped = discover_videos(args.video_dir)
    if args.limit_rollouts:
        keep = set(sorted(grouped)[: args.limit_rollouts])
        grouped = {key: value for key, value in grouped.items() if key in keep}
    streams = build_streams(grouped, annotations, args.view_mode)
    if not streams:
        raise RuntimeError("No streams were created. Check filenames and view mode.")

    output_root = args.output_dir.expanduser().resolve()
    logs_dir = output_root / "logs"
    metrics_dir = output_root / "metrics"
    cache_root = output_root / "window_cache"
    for directory in (logs_dir, metrics_dir, cache_root):
        directory.mkdir(parents=True, exist_ok=True)
    suffix = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    base = f"{short_name}_{args.view_mode}_trigger_only_{suffix}"
    log_path = logs_dir / f"{base}.log"
    window_csv = metrics_dir / f"{base}_windows.csv"
    rollout_csv = metrics_dir / f"{base}_rollouts.csv"
    summary_path = metrics_dir / f"{base}_summary.json"
    log_path.write_text("", encoding="utf-8")

    torch.cuda.empty_cache()
    reset_peak_memory()
    sync_cuda()
    load_start = time.perf_counter()
    model, processor, provenance = load_model(model_path, args)
    sync_cuda()
    model_load_seconds = time.perf_counter() - load_start
    load_memory = gpu_memory_mb()

    window_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    warmed = False

    for stream_index, stream in enumerate(streams, 1):
        duration = min(video_duration(path) for path in stream.videos)
        ends = decision_times(
            duration, args.start_time, args.window_seconds, args.stride_seconds
        )
        if args.max_windows_per_rollout:
            ends = ends[: args.max_windows_per_rollout]
        gt_failure = norm(stream.gt.get("failure_detected"))
        gt_time = parse_time(stream.gt.get("first_failure_time_s"))
        first_interrupt_time = None
        first_inference_s = None
        false_alarm = False
        interrupt_streak = 0
        monitor_memory = MonitorMemory()
        print(
            f"\n[{stream_index}/{len(streams)}] {stream.stream_id}: "
            f"duration={duration:.2f}s, windows={len(ends)}"
        )

        for window_index, end_s in enumerate(ends, 1):
            clip_start = time.perf_counter()
            window = build_window(stream, end_s, args, cache_root)
            clip_seconds = time.perf_counter() - clip_start
            memory_before = MonitorMemory(**monitor_memory.as_dict())
            prompt = build_detection_prompt(
                args.task_instruction,
                stream.view.split("+"),
                window.start_s,
                window.end_s,
                memory_before,
            )
            if not warmed and args.warmup:
                for warm_index in range(args.warmup):
                    print(f"Warm-up {warm_index + 1}/{args.warmup}")
                    generate(
                        model,
                        processor,
                        stream.view,
                        window.videos,
                        prompt,
                        args,
                        args.detection_max_new_tokens,
                    )
                warmed = True

            reset_peak_memory()
            raw, timing = generate(
                model,
                processor,
                stream.view,
                window.videos,
                prompt,
                args,
                args.detection_max_new_tokens,
            )
            memory = gpu_memory_mb()
            parsed, json_only = extract_json(raw)
            raw_pred_state = norm(parsed.get("monitor_state"))
            guarded_state, guard_reason = apply_trigger_guards(
                parsed, memory_before
            )
            task_complete_confirmed = update_task_completion(
                monitor_memory,
                parsed,
                guard_reason,
                end_s,
                args.task_complete_confirmations,
            )
            if task_complete_confirmed:
                guarded_state = "NORMAL"
                guard_reason = "TASK_COMPLETE_LOCKED"
                interrupt_streak = 0
            elif guarded_state == "INTERRUPT":
                interrupt_streak += 1
            else:
                interrupt_streak = 0

            if guarded_state == "INTERRUPT" and interrupt_streak < args.interrupt_confirmations:
                pred_state = "UNCERTAIN"
                confirmation_status = "PENDING_CONFIRMATION"
            else:
                pred_state = guarded_state
                confirmation_status = (
                    "CONFIRMED"
                    if pred_state == "INTERRUPT"
                    else "NOT_APPLICABLE"
                )
            expected_state = ""
            if gt_failure == "NO":
                expected_state = "NORMAL"
            elif gt_failure == "YES" and gt_time is not None:
                expected_state = "INTERRUPT" if end_s >= gt_time else "NORMAL"
            window_correct = pred_state == expected_state if expected_state else ""
            is_interrupt = pred_state == "INTERRUPT"

            if is_interrupt and first_interrupt_time is None:
                first_interrupt_time = end_s
                first_inference_s = timing["end_to_end_seconds"]
                false_alarm = bool(gt_time is None or end_s < gt_time)

            memory_detection = dict(parsed)
            memory_detection["monitor_state"] = pred_state
            toy_state_update_applied = not has_invalid_toy_transition(
                parsed, memory_before
            )
            if toy_state_update_applied:
                update_monitor_memory(
                    monitor_memory,
                    memory_detection,
                    end_s,
                    args.state_confirmations,
                )

            required = {
                "monitor_state",
                "visible_event",
                "action_stage",
                "left_toy_state",
                "right_toy_state",
                "evidence",
            }
            allowed_monitor_states = {"NORMAL", "INTERRUPT", "UNCERTAIN"}
            allowed_action_stages = {
                "APPROACH", "GRASP", "LIFT", "TRANSPORT", "PLACE", "UNKNOWN"
            }
            allowed_toy_states = CONFIRMABLE_TOY_STATES | {
                "UNKNOWN", "NOT_VISIBLE"
            }
            output_complete = required.issubset(parsed)
            visible_event = norm(parsed.get("visible_event"))
            expected_state_from_event = EXPECTED_MONITOR_STATE_BY_EVENT.get(
                visible_event
            )
            model_output_consistent = (
                expected_state_from_event is not None
                and raw_pred_state == expected_state_from_event
            )
            schema_valid = (
                output_complete
                and raw_pred_state in allowed_monitor_states
                and visible_event in EXPECTED_MONITOR_STATE_BY_EVENT
                and model_output_consistent
                and norm(parsed.get("action_stage")) in allowed_action_stages
                and norm(parsed.get("left_toy_state")) in allowed_toy_states
                and norm(parsed.get("right_toy_state")) in allowed_toy_states
            )
            row = {
                "mode": args.mode,
                "model": model_name,
                "stream_id": stream.stream_id,
                "rollout": stream.rollout,
                "view": stream.view,
                "window_index": window_index,
                "window_start_s": window.start_s,
                "window_end_s": window.end_s,
                "gt_first_failure_time_s": gt_time if gt_time is not None else "",
                "expected_monitor_state": expected_state,
                "raw_pred_monitor_state": raw_pred_state,
                "visible_event": visible_event,
                "expected_state_from_event": expected_state_from_event or "",
                "model_output_consistent": model_output_consistent,
                "guarded_monitor_state": guarded_state,
                "pred_monitor_state": pred_state,
                "action_stage": norm(parsed.get("action_stage")),
                "guard_reason": guard_reason,
                "toy_state_update_applied": toy_state_update_applied,
                "interrupt_streak": interrupt_streak,
                "interrupt_confirmation_status": confirmation_status,
                "task_complete_streak": monitor_memory.task_complete_streak,
                "task_complete_confirmed": monitor_memory.task_complete_confirmed,
                "task_complete_time_s": (
                    monitor_memory.task_complete_time_s
                    if monitor_memory.task_complete_time_s is not None
                    else ""
                ),
                "window_state_correct": window_correct,
                "previous_left_toy_state": memory_before.left_toy_state,
                "previous_right_toy_state": memory_before.right_toy_state,
                "pred_left_toy_state": norm(parsed.get("left_toy_state")),
                "pred_right_toy_state": norm(parsed.get("right_toy_state")),
                "confirmed_left_toy_state": monitor_memory.left_toy_state,
                "confirmed_right_toy_state": monitor_memory.right_toy_state,
                "left_ever_held": monitor_memory.left_ever_held,
                "right_ever_held": monitor_memory.right_ever_held,
                "evidence": str(parsed.get("evidence") or ""),
                "first_interrupt": is_interrupt and first_interrupt_time == end_s,
                "false_interrupt": is_interrupt and bool(gt_time is None or end_s < gt_time),
                "json_only": json_only,
                "output_complete": output_complete,
                "schema_valid": schema_valid,
                "clip_seconds": clip_seconds,
                **timing,
                "peak_allocated_mb": memory["peak_allocated"],
                "peak_reserved_mb": memory["peak_reserved"],
            }
            window_rows.append(row)
            print(
                f"  t={end_s:5.1f}s [{window.start_s:4.1f},{end_s:4.1f}] "
                f"GT={expected_state or '-':9s} raw={raw_pred_state or '-':9s} "
                f"final={pred_state or '-':9s} guard={guard_reason} "
                f"event={norm(parsed.get('visible_event')) or '-'} "
                f"state={monitor_memory.left_toy_state}/{monitor_memory.right_toy_state} "
                f"inference={timing['end_to_end_seconds']:.2f}s"
            )
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    "=" * 100
                    + f"\nStream: {stream.stream_id}\nWindow: {window.start_s:.3f}-{end_s:.3f}s"
                    + f"\nVideos: {' | '.join(str(path) for path in window.videos)}"
                    + f"\nGround truth: {json.dumps(stream.gt, ensure_ascii=False)}"
                    + f"\nPrevious confirmed state: {json.dumps(memory_before.as_dict(), ensure_ascii=False)}"
                    + f"\nUpdated confirmed state: {json.dumps(monitor_memory.as_dict(), ensure_ascii=False)}"
                    + "\n"
                    + "\n".join(provenance)
                    + f"\nFPS: {args.fps}\n\n=== DETECTOR OUTPUT ===\n{raw}"
                    + f"\n\n=== DETECTOR METRICS ===\n{json.dumps(row, indent=2, ensure_ascii=False)}"
                    + "\n\n"
                )
            if monitor_memory.task_complete_confirmed and args.stop_on_task_complete:
                break
            if is_interrupt and args.stop_on_interrupt:
                break

        triggered_after_gt = bool(
            gt_failure == "YES"
            and gt_time is not None
            and first_interrupt_time is not None
            and first_interrupt_time >= gt_time
        )
        detection_delay = (
            first_interrupt_time - gt_time
            if triggered_after_gt and gt_time is not None
            else None
        )
        reaction_delay = (
            detection_delay + first_inference_s
            if detection_delay is not None and first_inference_s is not None
            else None
        )
        if gt_failure == "YES":
            if first_interrupt_time is None:
                trigger_outcome = "MISSED"
            elif gt_time is not None and first_interrupt_time < gt_time:
                trigger_outcome = "EARLY_FALSE_INTERRUPT"
            elif (
                detection_delay is not None
                and detection_delay <= args.timely_threshold_seconds
            ):
                trigger_outcome = "TIMELY_TRIGGER"
            else:
                trigger_outcome = "LATE_TRIGGER"
        else:
            trigger_outcome = (
                "CORRECT_REJECTION"
                if first_interrupt_time is None
                else "FALSE_INTERRUPT"
            )
        rollout_rows.append(
            {
                "mode": args.mode,
                "model": model_name,
                "stream_id": stream.stream_id,
                "rollout": stream.rollout,
                "view": stream.view,
                "duration_s": duration,
                "gt_failure_detected": gt_failure,
                "gt_first_failure_time_s": gt_time if gt_time is not None else "",
                "first_pred_interrupt_time_s": first_interrupt_time if first_interrupt_time is not None else "",
                "triggered_after_gt": triggered_after_gt,
                "false_interrupt": false_alarm,
                "missed_interrupt": gt_failure == "YES" and not triggered_after_gt,
                "trigger_outcome": trigger_outcome,
                "task_complete_confirmed": monitor_memory.task_complete_confirmed,
                "task_complete_time_s": (
                    monitor_memory.task_complete_time_s
                    if monitor_memory.task_complete_time_s is not None
                    else ""
                ),
                "detection_delay_s": detection_delay if detection_delay is not None else "",
                "detector_inference_at_alarm_s": first_inference_s if first_inference_s is not None else "",
                "estimated_reaction_delay_s": reaction_delay if reaction_delay is not None else "",
            }
        )

    for path, rows in ((window_csv, window_rows), (rollout_csv, rollout_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    evaluable_failures = [r for r in rollout_rows if r["gt_failure_detected"] == "YES"]
    evaluable_successes = [r for r in rollout_rows if r["gt_failure_detected"] == "NO"]
    delays = [
        float(r["detection_delay_s"])
        for r in evaluable_failures
        if r["detection_delay_s"] != ""
    ]
    reactions = [
        float(r["estimated_reaction_delay_s"])
        for r in evaluable_failures
        if r["estimated_reaction_delay_s"] != ""
    ]
    outcome_counts = {
        label: sum(r["trigger_outcome"] == label for r in rollout_rows)
        for label in (
            "TIMELY_TRIGGER",
            "LATE_TRIGGER",
            "EARLY_FALSE_INTERRUPT",
            "MISSED",
            "FALSE_INTERRUPT",
            "CORRECT_REJECTION",
        )
    }
    valid_triggers = sum(bool(r["triggered_after_gt"]) for r in evaluable_failures)
    false_interrupts = sum(bool(r["false_interrupt"]) for r in rollout_rows)
    correct_rejections = sum(
        r["trigger_outcome"] == "CORRECT_REJECTION" for r in evaluable_successes
    )
    summary = {
        "mode": args.mode,
        "model": model_name,
        "model_path": str(model_path),
        "video_dir": str(args.video_dir.expanduser().resolve()),
        "annotations": str(args.annotations.expanduser().resolve()) if args.annotations else None,
        "task_instruction": args.task_instruction,
        "view_mode": args.view_mode,
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "fps": args.fps,
        "state_confirmations": args.state_confirmations,
        "interrupt_confirmations": args.interrupt_confirmations,
        "task_complete_confirmations": args.task_complete_confirmations,
        "timely_threshold_seconds": args.timely_threshold_seconds,
        "num_streams": len(rollout_rows),
        "num_windows_evaluated": len(window_rows),
        "online_trigger_recall": (
            valid_triggers / len(evaluable_failures)
            if evaluable_failures
            else None
        ),
        "timely_trigger_recall": (
            outcome_counts["TIMELY_TRIGGER"] / len(evaluable_failures)
            if evaluable_failures
            else None
        ),
        "late_trigger_rate": (
            outcome_counts["LATE_TRIGGER"] / len(evaluable_failures)
            if evaluable_failures
            else None
        ),
        "trigger_precision": (
            valid_triggers / (valid_triggers + false_interrupts)
            if valid_triggers + false_interrupts
            else None
        ),
        "success_false_interrupt_rate": (
            sum(bool(r["false_interrupt"]) for r in evaluable_successes)
            / len(evaluable_successes)
            if evaluable_successes
            else None
        ),
        "success_specificity": (
            correct_rejections / len(evaluable_successes)
            if evaluable_successes
            else None
        ),
        "success_task_complete_rate": (
            sum(bool(r["task_complete_confirmed"]) for r in evaluable_successes)
            / len(evaluable_successes)
            if evaluable_successes
            else None
        ),
        "failure_false_task_complete_rate": (
            sum(bool(r["task_complete_confirmed"]) for r in evaluable_failures)
            / len(evaluable_failures)
            if evaluable_failures
            else None
        ),
        "trigger_outcome_counts": outcome_counts,
        "mean_detection_delay_s": mean_or_none(delays),
        "mean_estimated_reaction_delay_s": mean_or_none(reactions),
        "mean_detector_end_to_end_s": mean_or_none(
            [r["end_to_end_seconds"] for r in window_rows]
        ),
        "json_only_rate": (
            sum(bool(r["json_only"]) for r in window_rows) / len(window_rows)
        ),
        "completion_rate": (
            sum(bool(r["output_complete"]) for r in window_rows) / len(window_rows)
        ),
        "schema_valid_rate": (
            sum(bool(r["schema_valid"]) for r in window_rows) / len(window_rows)
        ),
        "model_output_consistency_rate": (
            sum(bool(r["model_output_consistent"]) for r in window_rows)
            / len(window_rows)
        ),
        "model_load_seconds": model_load_seconds,
        "model_allocated_after_load_mb": load_memory["allocated"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpus": (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"\nDONE\nLog: {log_path}\nWindow CSV: {window_csv}"
        f"\nRollout CSV: {rollout_csv}\nSummary: {summary_path}"
    )


if __name__ == "__main__":
    main()
