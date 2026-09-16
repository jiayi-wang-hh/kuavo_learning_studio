#!/usr/bin/env python3
"""
Third-person robot failure-recovery video benchmark.

Current dataset convention:
    eval01_front.MOV  -> SUCCESS
    eval01_side.MOV   -> SUCCESS
    eval09_front.MOV  -> FAILURE
    eval09_side.MOV   -> FAILURE
    eval10_front.MOV  -> FAILURE
    eval10_side.MOV   -> FAILURE

Supports:
    1) individual-view evaluation (front/side separately)
    2) paired multi-view evaluation (front + side together)
    3) both

Models:
    qwen
    qwen3_8_27b
    robobrain
    robobrain_2.5
    cosmos_reason2_32b

Example:
    python vlm_test/test_vlm_video.py \
        --mode qwen \
        --video_dir /media/data/jiayi/dataset/failure_videos \
        --view_mode both \
        --task_instruction "pick the red apple from the cluttered scene" \
        --fps 4 \
        --warmup 0
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import statistics
import time
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import torch
import transformers
from transformers import AutoProcessor


# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------

MODE_INFO = {
    "qwen": (
        "Qwen2.5-VL-3B-Instruct",
        "/media/data/jiayi/hf_model/Qwen2.5-VL-3B-Instruct",
        "qwen25_vl_3b",
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


# ---------------------------------------------------------------------
# Ground truth for the four videos you currently have
# ---------------------------------------------------------------------

GROUND_TRUTH = {
    "eval01": {
        "outcome": "SUCCESS",
        "failure_detected": "NO",
    },
    "eval09": {
        "outcome": "FAILURE",
        "failure_detected": "YES",
    },
    "eval10": {
        "outcome": "FAILURE",
        "failure_detected": "YES",
    },  
}


# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------

def build_prompt(task_instruction: str, multi_view: bool) -> str:
    view_text = (
        "You are given synchronized FRONT and SIDE videos of the same robot execution."
        if multi_view
        else
        "You are given one third-person external-camera video of a robot execution."
    )

    return f"""You are a robot manipulation failure-recovery reasoning assistant.

{view_text}

Robot task instruction:
{task_instruction}

The videos are recorded in a real laboratory and may contain irrelevant visual distractors,
including human operators, monitors, furniture, cars, cables, equipment, and background motion.

IMPORTANT RULES:
- Focus on the robot, its arms/grippers, the manipulation workspace, the target object,
  relevant nearby objects, and robot-object interactions.
- Visible humans are NOT part of the manipulation task unless the human physically intervenes
  in the robot execution or changes the task state.
- Human motion or background motion alone is NOT evidence of robot failure.
- Do NOT assume the execution failed merely because this is a failure-recovery benchmark.
- Judge success or failure only from visible temporal evidence.
- Use temporal evidence: approach, contact, grasp, lift, slip, drop, miss, collision,
  lack of progress, or completion.
- If the visual evidence is insufficient, use UNCERTAIN instead of guessing.
- Do not reveal chain-of-thought or provide frame-by-frame analysis.
- Do not mention that this is a benchmark.
- Start the response immediately with "Task-relevant scene:".
- Output only the requested labeled fields, with no preamble or closing text.

A manipulation failure can include, for example:
- gripper misses the intended target,
- target is contacted but not secured,
- target slips or falls after grasping,
- wrong object is grasped,
- target is unintentionally displaced,
- robot collides with a task-relevant object,
- robot becomes stuck or stops making meaningful task progress,
- expected task state is not achieved after an attempted action.

Your purpose is to decide whether autonomous recovery is needed and, if so,
produce a concise recovery subtask suitable for sending to a downstream VLA policy.

Answer using EXACTLY these section labels:

Task-relevant scene:
Distractors:
Target:
Action sequence:
Task progress:
Outcome:
Failure detected:
Failure time:
Failure type:
Failure evidence:
Recovery needed:
Recovery entry trigger:
Recovery action:
Recovery exit condition:
Next action:

STRICT OUTPUT RULES:
- Outcome must be exactly one of: SUCCESS, FAILURE, UNCERTAIN.
- Failure detected must be exactly one of: YES, NO, UNCERTAIN.
- Recovery needed must be exactly one of: YES, NO, UNCERTAIN.
- If no failure is visible, write "None" for Failure time, Failure type,
  Failure evidence, Recovery entry trigger, Recovery action, and Recovery exit condition.
- Failure time should be an approximate timestamp or interval if inferable.
- Failure evidence must describe visible robot/object evidence, not speculation.
- Recovery entry trigger must be an observable condition that should switch the system
  into recovery mode.
- Recovery action must be a SHORT executable language-level subtask.
- Do not output low-level joint commands, coordinates, torques, or motor commands.
- Recovery exit condition must be an observable condition showing that recovery succeeded.
- Next action means what the robot should do immediately after the current video ends.
- Keep every field concise; use at most two sentences per field.
"""


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--mode", choices=tuple(MODE_INFO), required=True)

    p.add_argument(
        "--video_dir",
        type=Path,
        default=Path("/media/data/jiayi/dataset/failure_videos"),
    )

    p.add_argument(
        "--view_mode",
        choices=("individual", "paired", "both"),
        default="both",
        help=(
            "individual: evaluate front and side separately; "
            "paired: front+side together; both: run both."
        ),
    )

    p.add_argument(
        "--task_instruction",
        default="pick the red apple from the cluttered scene",
    )

    p.add_argument("--model_path", type=Path)

    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/media/data/jiayi/outputs/vlm_failure_recovery"),
    )

    p.add_argument("--fps", type=float, default=4.0)

    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=768,
    )

    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )

    p.add_argument(
        "--attn_implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )

    p.add_argument(
        "--device_map",
        default="auto",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--run_name")
    p.add_argument(
        "--qwen38_thinking",
        action="store_true",
        help=(
            "Enable Qwen3.8 thinking mode. It is disabled by default so the "
            "benchmark output remains concise and directly parseable."
        ),
    )

    return p.parse_args()


# ---------------------------------------------------------------------
# Sample representation
# ---------------------------------------------------------------------

@dataclass
class Sample:
    sample_id: str
    episode: str
    view: str
    videos: list[Path]
    gt_outcome: Optional[str]
    gt_failure: Optional[str]


# ---------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------

VIDEO_PATTERN = re.compile(
    r"^(eval\d+)_(front|side)\.(mov|mp4)$",
    re.IGNORECASE,
)


def normalize_episode_name(name: str) -> str:
    """
    eval1 -> eval01
    eval01 -> eval01
    eval9 -> eval09
    eval09 -> eval09
    """
    m = re.fullmatch(r"eval0*(\d+)", name.lower())
    if not m:
        return name.lower()
    return f"eval{int(m.group(1)):02d}"


def discover_videos(video_dir: Path) -> dict[str, dict[str, Path]]:
    video_dir = video_dir.expanduser().resolve()

    if not video_dir.is_dir():
        raise NotADirectoryError(video_dir)

    grouped: dict[str, dict[str, Path]] = {}

    for path in sorted(video_dir.iterdir()):
        if not path.is_file():
            continue

        m = VIDEO_PATTERN.match(path.name)

        if not m:
            continue

        episode = normalize_episode_name(m.group(1))
        view = m.group(2).lower()

        grouped.setdefault(episode, {})[view] = path.resolve()

    if not grouped:
        raise RuntimeError(
            f"No files matching evalXX_front/side.MOV or .mp4 under {video_dir}"
        )

    return grouped


def build_samples(
    grouped: dict[str, dict[str, Path]],
    view_mode: str,
) -> list[Sample]:

    samples: list[Sample] = []

    for episode in sorted(grouped):
        views = grouped[episode]

        gt = GROUND_TRUTH.get(episode, {})
        gt_outcome = gt.get("outcome")
        gt_failure = gt.get("failure_detected")

        if view_mode in ("individual", "both"):
            for view in ("front", "side"):
                if view in views:
                    samples.append(
                        Sample(
                            sample_id=f"{episode}_{view}",
                            episode=episode,
                            view=view,
                            videos=[views[view]],
                            gt_outcome=gt_outcome,
                            gt_failure=gt_failure,
                        )
                    )

        if view_mode in ("paired", "both"):
            if "front" in views and "side" in views:
                samples.append(
                    Sample(
                        sample_id=f"{episode}_front+side",
                        episode=episode,
                        view="front+side",
                        videos=[views["front"], views["side"]],
                        gt_outcome=gt_outcome,
                        gt_failure=gt_failure,
                    )
                )
            else:
                print(
                    f"[WARN] {episode}: paired mode skipped because "
                    f"front/side pair is incomplete."
                )

    return samples


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def load_model(path: Path, args):
    model_name = MODE_INFO[args.mode][0]

    processor = AutoProcessor.from_pretrained(
        str(path),
        trust_remote_code=True,
        use_fast=False,
    )

    if args.mode == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration

        cls = Qwen2_5_VLForConditionalGeneration

    elif args.mode == "qwen3_8_27b":
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError as exc:
            raise ImportError(
                "Qwen3.8-27B requires a recent Transformers version with "
                "AutoModelForMultimodalLM support. Upgrade Transformers in "
                "the Qwen3.8 environment."
            ) from exc

        cls = AutoModelForMultimodalLM

    elif args.mode == "robobrain":
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError as exc:
            raise ImportError(
                "This Transformers environment does not contain "
                "AutoModelForMultimodalLM. Run RoboBrain2.0 in the "
                "environment where your existing RoboBrain2.0 benchmark works."
            ) from exc

        cls = AutoModelForMultimodalLM

    elif args.mode == "robobrain_2.5":
        from transformers import AutoModelForImageTextToText

        cls = AutoModelForImageTextToText

    elif args.mode == "cosmos_reason2_32b":
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Cosmos-Reason2-32B requires a Transformers version "
                "with Qwen3-VL support."
            ) from exc

        cls = Qwen3VLForConditionalGeneration

    else:
        raise ValueError(args.mode)

    dtype = getattr(torch, args.dtype)

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": args.attn_implementation,
        "device_map": args.device_map,
    }

    if args.mode in ("cosmos_reason2_32b", "qwen3_8_27b"):
        kwargs["dtype"] = dtype
    else:
        kwargs["torch_dtype"] = dtype

    print(f"Loading {model_name} from:")
    print(path)

    model = cls.from_pretrained(
        str(path),
        **kwargs,
    )

    model.eval()

    provenance = [
        f"Mode: {args.mode}",
        f"Model: {model_name}",
        f"Checkpoint: {path}",
        f"dtype: {args.dtype}",
        f"attention: {args.attn_implementation}",
        f"device_map: {args.device_map}",
        f"Backend: {cls.__name__}",
    ]

    return model, processor, provenance


# ---------------------------------------------------------------------
# Video message construction
# ---------------------------------------------------------------------

def build_messages(
    sample: Sample,
    prompt: str,
    fps: float,
    qwen38_native: bool = False,
):
    content = []

    # Label views in text so the model knows which stream is which.
    if len(sample.videos) == 1:
        content.append(
            {
                "type": "text",
                "text": f"Camera view: {sample.view.upper()}",
            }
        )

        content.append(
            {
                "type": "video",
                # Qwen3.8 native Transformers processing expects a plain
                # local path. qwen_vl_utils used by the older models expects
                # a file:// URI.
                "video": (
                    str(sample.videos[0])
                    if qwen38_native
                    else f"file://{sample.videos[0]}"
                ),
                "fps": fps,
            }
        )

    else:
        content.append(
            {
                "type": "text",
                "text": "VIDEO 1: FRONT VIEW",
            }
        )

        content.append(
            {
                "type": "video",
                "video": (
                    str(sample.videos[0])
                    if qwen38_native
                    else f"file://{sample.videos[0]}"
                ),
                "fps": fps,
            }
        )

        content.append(
            {
                "type": "text",
                "text": "VIDEO 2: SIDE VIEW",
            }
        )

        content.append(
            {
                "type": "video",
                "video": (
                    str(sample.videos[1])
                    if qwen38_native
                    else f"file://{sample.videos[1]}"
                ),
                "fps": fps,
            }
        )

    content.append(
        {
            "type": "text",
            "text": prompt,
        }
    )

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


# ---------------------------------------------------------------------
# Processor handling
# ---------------------------------------------------------------------

def process_qwen_family(
    processor,
    messages,
):
    """
    Qwen2.5-VL / RoboBrain-family path using qwen_vl_utils.
    """
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise ImportError(
            "qwen-vl-utils is required for video inference. "
            "Install it in this environment."
        ) from exc

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    result = process_vision_info(messages)

    # Compatibility with common qwen-vl-utils versions.
    if isinstance(result, tuple) and len(result) >= 2:
        image_inputs = result[0]
        video_inputs = result[1]
    else:
        raise RuntimeError(
            "Unexpected return value from process_vision_info()."
        )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    return inputs


def process_cosmos(
    processor,
    messages,
    fps: float,
):
    """
    Cosmos-Reason2 / Qwen3-VL path.

    First try native chat-template multimodal processing.
    If the installed processor does not decode the video through the
    template path, fall back to qwen_vl_utils.
    """
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            fps=fps,
        )

        # A valid multimodal result normally contains pixel/video tensors.
        has_visual = any(
            k in inputs
            for k in (
                "pixel_values",
                "pixel_values_videos",
                "video_grid_thw",
                "image_grid_thw",
            )
        )

        if has_visual:
            return inputs

    except Exception as native_exc:
        print(
            "[Cosmos] native video template processing failed; "
            "trying qwen_vl_utils fallback."
        )
        print(f"[Cosmos] native error: {native_exc}")

    # qwen_vl_utils commonly expects local videos as file:// URIs.
    fallback_messages = deepcopy(messages)
    for message in fallback_messages:
        for item in message.get("content", []):
            if item.get("type") == "video":
                value = str(item.get("video", ""))
                if value and not value.startswith(("file://", "http://", "https://")):
                    item["video"] = f"file://{value}"

    return process_qwen_family(
        processor=processor,
        messages=fallback_messages,
    )


def process_qwen38(
    processor,
    messages,
    fps: float,
    enable_thinking: bool,
):
    """Native Qwen3.8 multimodal processing with a compatibility fallback."""
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
            preserve_thinking=False,
            processor_kwargs={"fps": fps},
        )

        has_visual = any(
            key in inputs
            for key in (
                "pixel_values",
                "pixel_values_videos",
                "video_grid_thw",
                "image_grid_thw",
            )
        )
        if has_visual:
            return inputs
    except Exception as native_exc:
        print(
            "[Qwen3.8] native video processing failed; "
            "trying qwen_vl_utils fallback."
        )
        print(f"[Qwen3.8] native error: {native_exc}")

    fallback_messages = deepcopy(messages)
    for message in fallback_messages:
        for item in message.get("content", []):
            if item.get("type") == "video":
                value = str(item.get("video", ""))
                if value and not value.startswith(("file://", "http://", "https://")):
                    item["video"] = f"file://{value}"

    return process_qwen_family(
        processor=processor,
        messages=fallback_messages,
    )


def prepare_inputs(
    model,
    processor,
    sample: Sample,
    args,
    prompt: str,
):
    messages = build_messages(
        sample=sample,
        prompt=prompt,
        fps=args.fps,
        qwen38_native=args.mode == "qwen3_8_27b",
    )

    if args.mode == "cosmos_reason2_32b":
        inputs = process_cosmos(
            processor=processor,
            messages=messages,
            fps=args.fps,
        )
    elif args.mode == "qwen3_8_27b":
        inputs = process_qwen38(
            processor=processor,
            messages=messages,
            fps=args.fps,
            enable_thinking=args.qwen38_thinking,
        )
    else:
        inputs = process_qwen_family(
            processor=processor,
            messages=messages,
        )

    inputs = inputs.to(input_device(model))

    return inputs


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------

def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_memory_mb():
    if not torch.cuda.is_available():
        return {
            "allocated": 0.0,
            "reserved": 0.0,
            "peak_allocated": 0.0,
            "peak_reserved": 0.0,
        }

    scale = 1024 ** 2

    return {
        "allocated": torch.cuda.memory_allocated() / scale,
        "reserved": torch.cuda.memory_reserved() / scale,
        "peak_allocated": torch.cuda.max_memory_allocated() / scale,
        "peak_reserved": torch.cuda.max_memory_reserved() / scale,
    }


def generate(
    model,
    processor,
    sample: Sample,
    args,
    prompt: str,
):
    total_start = time.perf_counter()

    inputs = prepare_inputs(
        model=model,
        processor=processor,
        sample=sample,
        args=args,
        prompt=prompt,
    )

    sync_cuda()

    preprocess_seconds = time.perf_counter() - total_start

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }

    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature

    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            **generation_kwargs,
        )

    sync_cuda()

    generation_seconds = time.perf_counter() - start

    input_tokens = int(inputs.input_ids.shape[1])

    generated_ids = output[:, input_tokens:]

    reasoning = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    output_tokens = int(generated_ids.shape[1])

    total_seconds = time.perf_counter() - total_start

    return reasoning, {
        "preprocess_seconds": preprocess_seconds,
        "generation_seconds": generation_seconds,
        "end_to_end_seconds": total_seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": (
            output_tokens / generation_seconds
            if generation_seconds > 0
            else 0.0
        ),
    }


# ---------------------------------------------------------------------
# Output parsing / automatic basic metrics
# ---------------------------------------------------------------------

LABELS = [
    "Task-relevant scene:",
    "Distractors:",
    "Target:",
    "Action sequence:",
    "Task progress:",
    "Outcome:",
    "Failure detected:",
    "Failure time:",
    "Failure type:",
    "Failure evidence:",
    "Recovery needed:",
    "Recovery entry trigger:",
    "Recovery action:",
    "Recovery exit condition:",
    "Next action:",
]

def clean_output(text: str) -> str:
    text = text.replace("**", "")
    return text


def structured_answer(text: str) -> str:
    """
    Parse only the final labeled answer block.

    Some reasoning models may mention labels in an analysis preamble even when
    instructed not to. Using the last Task-relevant scene label prevents that
    preamble from contaminating the benchmark fields.
    """
    matches = list(
        re.finditer(r"(?im)^\s*Task-relevant scene\s*:", text)
    )
    return text[matches[-1].start():].strip() if matches else text.strip()

def extract_field(
    text: str,
    label: str,
) -> str:
    """
    Extract text from one label until the next known label.
    Supports multiline values.
    """
    labels_pattern = "|".join(
        re.escape(x) for x in LABELS
    )

    pattern = (
        rf"(?is){re.escape(label)}\s*"
        rf"(.*?)"
        rf"(?=\n\s*(?:{labels_pattern})|\Z)"
    )

    m = re.search(pattern, text)

    if not m:
        return ""

    return " ".join(
        m.group(1).strip().split()
    )


def output_complete(text: str) -> bool:
    low = text.lower()
    return all(
        label.lower() in low
        for label in LABELS
    )


def normalize_choice(text: str) -> str:
    text = re.sub(r"^[\s`*_#>-]+", "", text).strip().upper()

    for choice in (
        "SUCCESS",
        "FAILURE",
        "UNCERTAIN",
        "YES",
        "NO",
    ):
        if text.startswith(choice):
            return choice

    return text

def normalize_outcome(text: str) -> str:
    t = re.sub(r"^[\s`*_#>-]+", "", text).strip().upper()

    # Prefer an explicit leading class when the model follows the prompt.
    leading = re.match(r"^(SUCCESS|FAILURE|UNCERTAIN)\b", t)
    if leading:
        return leading.group(1)

    # Normalize semantically equivalent free-form answers. Check uncertainty
    # before failure because text such as "uncertain whether it failed" must
    # not be converted to FAILURE merely because it contains "fail".
    if re.search(r"\b(UNCERTAIN|UNKNOWN|UNSURE|CANNOT DETERMINE)\b", t):
        return "UNCERTAIN"
    if re.search(r"\b(FAIL|FAILS|FAILED|FAILURE|UNSUCCESSFUL|MISSED)\b", t):
        return "FAILURE"
    if re.search(r"\b(SUCCESS|SUCCESSFUL|SUCCEEDED|COMPLETED|COMPLETE)\b", t):
        return "SUCCESS"
    return t


def normalize_yes_no_uncertain(text: str) -> str:
    """Normalize strict labels and common free-form equivalents."""
    t = re.sub(r"^[\s`*_#>-]+", "", text).strip().upper()
    leading = re.match(r"^(YES|NO|UNCERTAIN)\b", t)
    if leading:
        return leading.group(1)
    if re.search(r"\b(UNCERTAIN|UNKNOWN|UNSURE|CANNOT DETERMINE)\b", t):
        return "UNCERTAIN"
    if re.search(r"\b(NO FAILURE|NOT DETECTED|NONE)\b", t):
        return "NO"
    if re.search(r"\b(FAILURE DETECTED|FAILURE IS VISIBLE|NEEDS? RECOVERY)\b", t):
        return "YES"
    return t


def repetition_score(text: str) -> float:
    words = re.findall(r"\w+", text.lower())

    if not words:
        return 1.0

    return 1.0 - len(set(words)) / len(words)


def max_token_run(text: str) -> int:
    words = re.findall(r"\w+", text.lower())

    if not words:
        return 0

    best = 1
    current = 1

    for a, b in zip(words, words[1:]):
        if a == b:
            current += 1
        else:
            current = 1

        best = max(best, current)

    return best


def percentile(values, fraction):
    values = sorted(values)
    return values[
        round((len(values) - 1) * fraction)
    ]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_name, default_model_path, short_name = MODE_INFO[args.mode]

    model_path = (
        args.model_path
        if args.model_path is not None
        else Path(default_model_path)
    )

    model_path = model_path.expanduser().resolve()

    if not model_path.is_dir():
        raise NotADirectoryError(
            f"Model path does not exist:\n{model_path}"
        )

    grouped = discover_videos(args.video_dir)

    print("\nDiscovered videos:")
    for episode, views in grouped.items():
        print(f"  {episode}:")
        for view, path in sorted(views.items()):
            print(f"    {view}: {path.name}")

    samples = build_samples(
        grouped=grouped,
        view_mode=args.view_mode,
    )

    if args.limit:
        samples = samples[:args.limit]

    if not samples:
        raise RuntimeError("No benchmark samples created.")

    print("\nBenchmark samples:")
    for sample in samples:
        print(
            f"  {sample.sample_id}: "
            + ", ".join(x.name for x in sample.videos)
        )

    multi_view_possible = any(
        len(sample.videos) == 2
        for sample in samples
    )

    # Each sample gets the appropriate prompt because individual and
    # paired runs are mixed when --view_mode both.
    output_root = args.output_dir.expanduser().resolve()

    logs_dir = output_root / "logs"
    metrics_dir = output_root / "metrics"

    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    suffix = (
        args.run_name
        if args.run_name
        else time.strftime("%Y%m%d_%H%M%S")
    )

    log_path = (
        logs_dir
        / f"{short_name}_{args.view_mode}_{suffix}.log"
    )

    csv_path = (
        metrics_dir
        / f"{short_name}_{args.view_mode}_{suffix}_per_sample.csv"
    )

    summary_path = (
        metrics_dir
        / f"{short_name}_{args.view_mode}_{suffix}_summary.json"
    )

    torch.cuda.empty_cache()
    reset_peak_memory()
    sync_cuda()

    load_start = time.perf_counter()

    model, processor, provenance = load_model(
        model_path,
        args,
    )

    sync_cuda()

    model_load_seconds = (
        time.perf_counter() - load_start
    )

    load_memory = gpu_memory_mb()

    print("\n" + "\n".join(provenance))
    print(f"Model load: {model_load_seconds:.2f}s")
    print(f"FPS: {args.fps}")
    print(f"Task: {args.task_instruction}")
    print(f"View mode: {args.view_mode}")

    if args.warmup > 0:
        warmup_sample = samples[0]

        warmup_prompt = build_prompt(
            args.task_instruction,
            multi_view=len(warmup_sample.videos) == 2,
        )

        for i in range(args.warmup):
            print(
                f"Warm-up {i + 1}/{args.warmup}: "
                f"{warmup_sample.sample_id}"
            )

            generate(
                model,
                processor,
                warmup_sample,
                args,
                warmup_prompt,
            )

    log_path.write_text("", encoding="utf-8")

    rows: list[dict[str, Any]] = []

    for idx, sample in enumerate(samples, 1):
        prompt = build_prompt(
            args.task_instruction,
            multi_view=len(sample.videos) == 2,
        )

        reset_peak_memory()

        before = gpu_memory_mb()

        reasoning, timing = generate(
            model=model,
            processor=processor,
            sample=sample,
            args=args,
            prompt=prompt,
        )

        reasoning = clean_output(reasoning)
        parse_text = structured_answer(reasoning)

        after = gpu_memory_mb()

        pred_outcome_raw = extract_field(
            parse_text,
            "Outcome:",
        )

        pred_outcome = normalize_choice(
            pred_outcome_raw
        )
        pred_outcome = normalize_outcome(
            pred_outcome
        )

        pred_failure = normalize_yes_no_uncertain(
            extract_field(
                parse_text,
                "Failure detected:",
            )
        )

        gt_outcome = (
            normalize_outcome(sample.gt_outcome)
            if sample.gt_outcome is not None
            else None
        )

        outcome_correct = (
            gt_outcome is not None
            and pred_outcome == gt_outcome
        )

        failure_correct = (
            sample.gt_failure is not None
            and pred_failure == sample.gt_failure
        )

        row = {
            "mode": args.mode,
            "model": model_name,
            "sample_id": sample.sample_id,
            "episode": sample.episode,
            "view": sample.view,
            "video_1": str(sample.videos[0]),
            "video_2": (
                str(sample.videos[1])
                if len(sample.videos) > 1
                else ""
            ),
            "gt_outcome": sample.gt_outcome or "",
            "pred_outcome": pred_outcome,
            "outcome_correct": outcome_correct,
            "gt_failure_detected": sample.gt_failure or "",
            "pred_failure_detected": pred_failure,
            "failure_detection_correct": failure_correct,
            "failure_time": extract_field(
                parse_text,
                "Failure time:",
            ),
            "failure_type": extract_field(
                parse_text,
                "Failure type:",
            ),
            "failure_evidence": extract_field(
                parse_text,
                "Failure evidence:",
            ),
            "recovery_needed": normalize_yes_no_uncertain(
                extract_field(
                    parse_text,
                    "Recovery needed:",
                )
            ),
            "recovery_entry_trigger": extract_field(
                parse_text,
                "Recovery entry trigger:",
            ),
            "recovery_action": extract_field(
                parse_text,
                "Recovery action:",
            ),
            "recovery_exit_condition": extract_field(
                parse_text,
                "Recovery exit condition:",
            ),
            "next_action": extract_field(
                parse_text,
                "Next action:",
            ),
            "distractors": extract_field(
                parse_text,
                "Distractors:",
            ),
            "task_progress": extract_field(
                parse_text,
                "Task progress:",
            ),
            "fps": args.fps,
            "input_tokens": timing["input_tokens"],
            "output_tokens": timing["output_tokens"],
            "preprocess_seconds": timing[
                "preprocess_seconds"
            ],
            "generation_seconds": timing[
                "generation_seconds"
            ],
            "end_to_end_seconds": timing[
                "end_to_end_seconds"
            ],
            "tokens_per_second": timing[
                "tokens_per_second"
            ],
            "gpu_allocated_before_mb": before[
                "allocated"
            ],
            "peak_allocated_mb": after[
                "peak_allocated"
            ],
            "peak_reserved_mb": after[
                "peak_reserved"
            ],
            "output_complete": output_complete(
                parse_text
            ),
            "repetition_score": repetition_score(
                reasoning
            ),
            "max_token_run": max_token_run(
                reasoning
            ),
        }

        rows.append(row)

        print(
            f"\n[{idx}/{len(samples)}] {sample.sample_id}"
        )

        print(
            f"  GT outcome={sample.gt_outcome}, "
            f"prediction={pred_outcome}, "
            f"correct={outcome_correct}"
        )

        print(
            f"  GT failure={sample.gt_failure}, "
            f"prediction={pred_failure}, "
            f"correct={failure_correct}"
        )

        print(
            f"  time={timing['end_to_end_seconds']:.2f}s, "
            f"{timing['tokens_per_second']:.2f} token/s"
        )

        block = "\n".join(
            [
                "=" * 100,
                f"Sample: {sample.sample_id}",
                f"Episode: {sample.episode}",
                f"View: {sample.view}",
                "Videos:",
                *[
                    f"  - {x}"
                    for x in sample.videos
                ],
                f"Ground truth outcome: {sample.gt_outcome}",
                f"Ground truth failure: {sample.gt_failure}",
                *provenance,
                f"FPS: {args.fps}",
                f"Task: {args.task_instruction}",
                "",
                "=== MODEL OUTPUT ===",
                reasoning,
                "",
                "=== PARSED METRICS ===",
                json.dumps(
                    row,
                    indent=2,
                    ensure_ascii=False,
                ),
                "",
            ]
        )

        with log_path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(block + "\n")

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    labeled_rows = [
        r
        for r in rows
        if r["gt_outcome"]
    ]

    outcome_accuracy = (
        sum(
            bool(r["outcome_correct"])
            for r in labeled_rows
        )
        / len(labeled_rows)
        if labeled_rows
        else None
    )

    failure_accuracy = (
        sum(
            bool(r["failure_detection_correct"])
            for r in labeled_rows
        )
        / len(labeled_rows)
        if labeled_rows
        else None
    )

    individual_rows = [
        r
        for r in rows
        if r["view"] in ("front", "side")
    ]

    paired_rows = [
        r
        for r in rows
        if r["view"] == "front+side"
    ]

    failure_gt_rows = [
        r for r in labeled_rows
        if r["gt_outcome"] == "FAILURE"
    ]

    success_gt_rows = [
        r for r in labeled_rows
        if r["gt_outcome"] == "SUCCESS"
    ]

    def accuracy_for(subset, field):
        if not subset:
            return None

        return sum(
            bool(r[field])
            for r in subset
        ) / len(subset)

    summary = {
        "mode": args.mode,
        "model": model_name,
        "model_path": str(model_path),
        "video_dir": str(
            args.video_dir.expanduser().resolve()
        ),
        "task_instruction": args.task_instruction,
        "view_mode": args.view_mode,
        "fps": args.fps,
        "num_samples": len(rows),
        "num_individual_samples": len(
            individual_rows
        ),
        "num_paired_samples": len(
            paired_rows
        ),
        "outcome_accuracy": outcome_accuracy,
        "failure_detection_accuracy": (
            failure_accuracy
        ),
        "individual_outcome_accuracy": (
            accuracy_for(
                individual_rows,
                "outcome_correct",
            )
        ),
        "individual_failure_detection_accuracy": (
            accuracy_for(
                individual_rows,
                "failure_detection_correct",
            )
        ),
        "paired_outcome_accuracy": (
            accuracy_for(
                paired_rows,
                "outcome_correct",
            )
        ),
        "paired_failure_detection_accuracy": (
            accuracy_for(
                paired_rows,
                "failure_detection_correct",
            )
        ),
        # Failure-only metrics are intentionally separate from overall
        # accuracy. This avoids reporting true negatives on successful runs
        # as if they were successful failure detections.
        "num_failure_samples": len(failure_gt_rows),
        "num_success_samples": len(success_gt_rows),
        "failure_recall": (
            sum(
                r["pred_failure_detected"] == "YES"
                for r in failure_gt_rows
            ) / len(failure_gt_rows)
            if failure_gt_rows else None
        ),
        "failure_outcome_recall": (
            sum(
                r["pred_outcome"] == "FAILURE"
                for r in failure_gt_rows
            ) / len(failure_gt_rows)
            if failure_gt_rows else None
        ),
        "failure_uncertain_rate": (
            sum(
                r["pred_failure_detected"] == "UNCERTAIN"
                or r["pred_outcome"] == "UNCERTAIN"
                for r in failure_gt_rows
            ) / len(failure_gt_rows)
            if failure_gt_rows else None
        ),
        "success_specificity": (
            sum(
                r["pred_failure_detected"] == "NO"
                for r in success_gt_rows
            ) / len(success_gt_rows)
            if success_gt_rows else None
        ),
        "completion_rate": (
            sum(
                bool(r["output_complete"])
                for r in rows
            )
            / len(rows)
        ),
        "mean_end_to_end_seconds": statistics.mean(
            r["end_to_end_seconds"]
            for r in rows
        ),
        "median_end_to_end_seconds": statistics.median(
            r["end_to_end_seconds"]
            for r in rows
        ),
        "p95_end_to_end_seconds": percentile(
            [
                r["end_to_end_seconds"]
                for r in rows
            ],
            0.95,
        ),
        "mean_generation_seconds": statistics.mean(
            r["generation_seconds"]
            for r in rows
        ),
        "mean_tokens_per_second": statistics.mean(
            r["tokens_per_second"]
            for r in rows
        ),
        "model_load_seconds": model_load_seconds,
        "model_allocated_after_load_mb": (
            load_memory["allocated"]
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpus": (
            [
                torch.cuda.get_device_name(i)
                for i in range(
                    torch.cuda.device_count()
                )
            ]
            if torch.cuda.is_available()
            else []
        ),
        "ground_truth": GROUND_TRUTH,
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Log:     {log_path}")
    print(f"CSV:     {csv_path}")
    print(f"Summary: {summary_path}")
    print()
    print(
        f"Outcome accuracy: "
        f"{outcome_accuracy if outcome_accuracy is not None else 'N/A'}"
    )
    print(
        f"Failure detection accuracy: "
        f"{failure_accuracy if failure_accuracy is not None else 'N/A'}"
    )
    if failure_gt_rows:
        failure_recall = sum(
            r["pred_failure_detected"] == "YES"
            for r in failure_gt_rows
        ) / len(failure_gt_rows)
        print(f"Failure-only recall: {failure_recall}")


if __name__ == "__main__":
    main()
