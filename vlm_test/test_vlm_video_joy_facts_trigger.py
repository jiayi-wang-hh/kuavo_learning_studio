#!/usr/bin/env python3
"""Causal stage-1 VLM trigger that decides only CONTINUE or PAUSE.

This detector intentionally does not classify failure types, plan recovery,
track object state, or decide task completion.  PAUSE hands a short video
buffer to a stronger stage-2 verifier; false pauses may therefore be resumed
by stage 2.
"""

from __future__ import annotations

import os
import sys

# Qwen3.5 needs a newer Transformers stack than the current lingbotvla env.
# Keep the current environment untouched: when --mode qwen35 is requested,
# restart this same script with the dedicated qwen35 Python interpreter.
QWEN35_PYTHON = "/home/kuavo/miniforge3/envs/qwen35/bin/python"

def _maybe_reexec_in_qwen35_env() -> None:
    if "--mode" not in sys.argv:
        return

    try:
        mode_index = sys.argv.index("--mode") + 1
        mode = sys.argv[mode_index]
    except (ValueError, IndexError):
        return

    if mode not in {"qwen35", "qwen35_9b"}:
        return

    target_python = os.path.realpath(QWEN35_PYTHON)
    current_python = os.path.realpath(sys.executable)

    if current_python == target_python:
        return

    if not os.path.isfile(target_python):
        raise FileNotFoundError(
            f"Dedicated Qwen3.5 Python not found: {target_python}"
        )

    print(
        f"[env-switch] qwen35 requested; re-launching with {target_python}",
        flush=True,
    )
    os.execv(
        target_python,
        [target_python, os.path.abspath(__file__), *sys.argv[1:]],
    )

_maybe_reexec_in_qwen35_env()

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import transformers

# from vlm_test import test_vlm_video_joy_trigger as common
import test_vlm_video_joy_trigger as common


ALLOWED_DECISIONS = {"CONTINUE", "PAUSE"}
ALLOWED_CONFIDENCE = {"LOW", "MEDIUM", "HIGH"}
ALLOWED_FACT_VALUES = {"YES", "NO", "UNCERTAIN"}


def parse_args():
    """Parse pause-trigger options, then reuse the shared evaluation options."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--trigger-current-command",
        default="",
        help="Live controller command, e.g. CLOSE_GRIPPER_AND_LIFT.",
    )
    p.add_argument(
        "--trigger-expected-effect",
        default="",
        help="Expected visible result of the current controller command.",
    )
    p.add_argument(
        "--trigger-contact-frames",
        type=int,
        default=4,
        help="Chronological frames rendered into each contact sheet.",
    )
    p.add_argument("--detection-max-new-tokens", type=int, default=192)
    p.add_argument(
        "--trigger-use-contact-sheets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use image contact sheets rather than raw video inputs.",
    )
    p.add_argument(
        "--stop-on-pause",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stop evaluating later windows after a confirmed PAUSE.",
    )
    trigger_args, remaining = p.parse_known_args()
    if trigger_args.trigger_contact_frames < 2:
        p.error("--trigger-contact-frames must be >= 2")
    sys.argv = [sys.argv[0], *remaining]
    # Extend the shared CLI with an explicit Qwen2.5-7B alias for this trigger script.
    # common.parse_args() builds --mode choices from common.MODE_INFO, so add the alias first.
    if "qwen25" not in common.MODE_INFO:
        common.MODE_INFO["qwen25"] = (
            "Qwen2.5-VL-7B-Instruct",
            "/media/data/jiayi/hf_model/Qwen2.5-VL-7B-Instruct",
            "qwen25_vl_7b",
        )

    # Short CLI alias for Qwen3.5-4B.
    # The shared implementation already knows how to load qwen35_4b, so after
    # parsing we map this convenience alias back to the canonical shared mode.
    if "qwen35" not in common.MODE_INFO:
        common.MODE_INFO["qwen35"] = (
            "Qwen3.5-4B",
            "/media/data/jiayi/hf_model/Qwen3.5-4B",
            "qwen35_4b",
        )

    if "qwen35_9b" not in common.MODE_INFO:
        common.MODE_INFO["qwen35_9b"] = (
            "Qwen3.5-9B",
            "/media/data/jiayi/hf_model/Qwen3.5-9B",
            "qwen35_9b",
        )

    args = common.parse_args()
    if args.mode == "qwen35":
        args.mode = "qwen35_4b"
    args.trigger_current_command = trigger_args.trigger_current_command
    args.trigger_expected_effect = trigger_args.trigger_expected_effect
    args.trigger_contact_frames = trigger_args.trigger_contact_frames
    args.trigger_use_contact_sheets = trigger_args.trigger_use_contact_sheets
    args.detection_max_new_tokens = trigger_args.detection_max_new_tokens
    # Keep --stop-on-interrupt working for compatibility with the shared CLI,
    # but expose the stage-1 meaning under its own name.
    args.stop_on_pause = (
        trigger_args.stop_on_pause
        if trigger_args.stop_on_pause is not None
        else args.stop_on_interrupt
    )
    return args


def build_pause_prompt_default(
    task: str,
    views: list[str],
    start_s: float,
    end_s: float,
    current_command: str,
    expected_effect: str,
) -> str:
    controller_context = (
        f"Current controller command: {current_command}\n"
        f"Expected visible effect: {expected_effect}"
        if current_command or expected_effect
        else (
            "Current controller command: unavailable. "
            "Do not assume that a grasp, lift, or placement has completed "
            "unless completion is directly visible."
        )
    )

    return f"""You are a fast stage-1 visual trigger for robot manipulation.

You receive chronological contact sheets. Within each sheet, tiles run from
earliest (left) to latest (right). The first sheet is the full HEAD view; the
next two sheets are crops of the left and right workspaces.

The observation window covers {start_s:.1f} to {end_s:.1f} seconds.
Use only visible changes inside this window.

Task:
{task}

{controller_context}

Your only decision is:
- CONTINUE: there is no direct visual evidence that execution must be interrupted.
- PAUSE: there is direct visual evidence of failure, abnormal motion, or persistent no-progress.

Do not classify the failure type and do not plan recovery.

Decision procedure:

1. First compare the EARLIEST and LATEST tiles.
2. Determine what visibly changed:
   - Did a gripper move?
   - Did a target object move with the gripper?
   - Did an object visibly fall, slip, or separate?
   - Did a placement visibly finish outside the basket?
   - Was there clearly abnormal contact or collision?
3. Return PAUSE only if the observed change itself provides direct evidence
   that execution has failed or become abnormal.

PAUSE may be appropriate when:
- after an actually visible grasp/lift attempt, the gripper moves away while
  the relevant object visibly stays at its previous location;
- an object that was visibly moving with a gripper later separates, slips, or falls;
- a placement visibly finishes and the object remains outside or unstable;
- clearly abnormal contact or collision is visible;
- after meaningful motion has already begun, neither the robot nor target shows
  meaningful progress throughout the observation window.

Return CONTINUE when:
- the robot is approaching, aligning, positioning, or preparing to grasp;
- a grasp/lift/transport/placement is still in progress;
- the object remains on the table before a completed grasp attempt is visible;
- the gripper moves but there is not enough evidence to know whether a grasp
  attempt has completed;
- the visual evidence is ambiguous;
- execution appears normal.

Critical grounding rules:
- Do NOT copy or paraphrase a PAUSE criterion as evidence.
- Evidence must describe a concrete observation from THIS window.
- Evidence must identify the visible actor/object and the actual change observed.
- Never claim "the target remains behind" unless you can visibly compare its
  position before and after the gripper motion.
- Never claim a grasp/lift failure unless a grasp/lift attempt is visibly established.
- Gripper motion alone is NOT evidence of grasp failure.
- An object being on the table is NOT evidence of failure.
- If you cannot identify a specific visible failure event, return CONTINUE.
- If the evidence sentence could be reused unchanged for many different videos,
  it is too generic and must not justify PAUSE.

Examples of acceptable evidence:
- "The left gripper rises between the first and last tiles while the red toy
  remains at the same table position."
- "The right toy is held in an earlier tile but is visibly back on the table
  in the final tile."
- "Both grippers are still approaching the toys and no completed grasp is visible."

Examples of unacceptable evidence:
- "A gripper moves away while its target remains behind."
- "The grasp failed."
- "There is no progress."
- Any sentence copied from the decision rules above.

Return exactly one JSON object:

{{
  "trigger_decision": "CONTINUE | PAUSE",
  "confidence": "LOW | MEDIUM | HIGH",
  "evidence": "one concrete observation from this window"
}}
"""


def build_pause_prompt_qwen35(
    task: str,
    views: list[str],
    start_s: float,
    end_s: float,
    current_command: str,
    expected_effect: str,
) -> str:
    """Qwen3.5-specific trigger prompt.

    Qwen3.5 was observed to ground evidence better than Qwen2.5, but to be too
    conservative: it often described a likely grasp miss correctly and still
    returned CONTINUE. This prompt therefore defines grasp attempts from the
    visible motion sequence rather than requiring the exact closure frame.
    """
    controller_context = (
        f"Current controller command: {current_command}\n"
        f"Expected visible effect: {expected_effect}"
        if current_command or expected_effect
        else (
            "Current controller command: unavailable. Infer action phase from the "
            "visible motion sequence only; do not require seeing the exact instant "
            "of gripper closure."
        )
    )

    return f"""You are a fast stage-1 visual trigger for robot manipulation.

You receive chronological contact sheets. Within each sheet, tiles run from
earliest (left) to latest (right). The first sheet is the full HEAD view; the
next two sheets are crops of the left and right workspaces.

The observation window covers {start_s:.1f} to {end_s:.1f} seconds.
Use only visible evidence inside this window.

Task:
{task}

{controller_context}

Your only decision is:
- CONTINUE: execution is visibly normal or the manipulation attempt has not yet
  produced evidence of failure.
- PAUSE: the visible motion sequence provides evidence that a manipulation
  attempt failed, became abnormal, or stalled.

Do not classify the failure type and do not plan recovery.

Important temporal rule:
A grasp attempt does NOT require seeing the exact gripper-closing frame.
Treat a grasp attempt as established when the visible sequence is consistent with:

    approach target -> reach/contact target region -> move away or upward

Then check the object outcome.

Return PAUSE when any of these is visible:
- The gripper reaches the target region and then moves away/up, while the intended
  object remains at approximately the same table position.
- The gripper clearly departs after an attempted grasp without carrying the target.
- An object that was visibly moving with a gripper later slips, falls, or separates.
- A placement visibly finishes with the object outside the basket or unstable on
  its edge.
- After a manipulation attempt has begun, robot/object motion shows persistent
  no-progress across the observation window.
- Clearly abnormal contact, collision, or unsafe motion occurs.

Return CONTINUE when:
- The gripper is still approaching and has not yet reached the target region.
- The target is visibly moving together with the gripper.
- Transport or placement is visibly progressing normally.
- The sequence genuinely does not show whether target interaction occurred.
- Execution appears normal and there is no concrete failure evidence.

Decision procedure:
1. Compare earliest and latest tiles.
2. For each side, determine:
   a. Did the gripper reach the target region?
   b. Did it subsequently move away/up?
   c. Did the target move with the gripper?
3. If (a) and (b) are true but (c) is false, this is sufficient evidence for PAUSE.
4. Do not downgrade that pattern to "still approaching" merely because the exact
   closure instant is not visible.

Grounding rules:
- Evidence must describe a concrete observation from THIS window.
- Name the visible side/object when possible.
- Do not copy or paraphrase the decision rules as generic evidence.
- Gripper motion alone is not enough; relate it to the target object's motion.
- An object merely being on the table before interaction is not a failure.
- If the evidence shows "gripper reached target -> departed -> object stayed",
  return PAUSE.
- If the evidence sentence could be reused unchanged for many different videos,
  it is too generic.

Examples:

CONTINUE:
{{"trigger_decision":"CONTINUE","confidence":"MEDIUM",
  "evidence":"The left gripper is still approaching the red toy and has not yet moved away from the target region."}}

PAUSE:
{{"trigger_decision":"PAUSE","confidence":"HIGH",
  "evidence":"The left gripper reaches the red toy and then rises away, while the red toy remains at the same table position."}}

Return exactly one JSON object:

{{
  "trigger_decision": "CONTINUE | PAUSE",
  "confidence": "LOW | MEDIUM | HIGH",
  "evidence": "one concrete observation from this window"
}}
"""


def build_pause_prompt(
    mode: str,
    task: str,
    views: list[str],
    start_s: float,
    end_s: float,
    current_command: str,
    expected_effect: str,
) -> str:
    if mode in {"qwen35", "qwen35_4b", "qwen35_9b"}:
        return build_pause_prompt_qwen35(
            task,
            views,
            start_s,
            end_s,
            current_command,
            expected_effect,
        )
    return build_pause_prompt_default(
        task,
        views,
        start_s,
        end_s,
        current_command,
        expected_effect,
    )



def build_visual_facts_prompt(
    task: str,
    views: list[str],
    start_s: float,
    end_s: float,
    current_command: str,
    expected_effect: str,
) -> str:
    controller_context = (
        f"Current controller command: {current_command}\n"
        f"Expected visible effect: {expected_effect}"
        if current_command or expected_effect
        else "Current controller command: unavailable."
    )

    return f"""You are a visual observer for robot manipulation.

IMPORTANT: You do NOT decide PAUSE or CONTINUE.
You only report directly visible facts from the observation window.

You receive chronological contact sheets. In each sheet, tiles run from
earliest (left) to latest (right). The first sheet is the full HEAD view; the
next two sheets are left and right workspace crops.

Observation window: {start_s:.1f} to {end_s:.1f} seconds.

Task:
{task}

{controller_context}

Object-role rules:
- The small movable toys/objects are manipulation TARGETS.
- The baskets/containers are DESTINATIONS, not grasp targets.
- Never report a basket remaining stationary as evidence about whether a grasp
  of a toy succeeded.
- Evaluate LEFT and RIGHT sides independently.
- Do not infer an event merely because it is plausible. Use UNCERTAIN when the
  contact sheet does not establish it.

For each side, report:

1. attempt_visible
   YES only if the gripper visibly reaches/interacts with the corresponding
   movable target and the sequence shows an actual manipulation attempt.
   NO if it is clearly still approaching or never reaches the target.
   UNCERTAIN if contact/attempt cannot be established.

2. target_following
   YES if the movable target visibly moves together with the gripper after the
   attempt.
   NO if an attempt is visible and the gripper subsequently departs while the
   movable target clearly stays behind.
   UNCERTAIN if the relationship cannot be established.
   If attempt_visible is NO, use UNCERTAIN here.

3. unexpected_drop_visible
    YES only if the object unintentionally slips, falls, or separates
    outside the intended placement.

    A deliberate release into the correct basket is NOT a drop failure
    and must be NO.

Also report:
- placement_failure_visible:
  YES only if a placement visibly finishes with a target outside a basket or
  unstable on its edge.
- unsafe_motion_visible:
  YES only for a clearly abnormal collision or unsafe motion.

Grounding requirements:
- Describe observations, not conclusions.
- Do NOT output "grasp failed", "task failed", "PAUSE", or "CONTINUE".
- Do NOT copy a rule as evidence.
- Distinguish movable toys from baskets.
- If the sequence is ambiguous, use UNCERTAIN instead of guessing.

Return exactly one JSON object and nothing else:

{{
  "left_attempt_visible": "YES | NO | UNCERTAIN",
  "left_target_following": "YES | NO | UNCERTAIN",
  "left_unexpected_drop_visible": "YES | NO | UNCERTAIN",
  "right_attempt_visible": "YES | NO | UNCERTAIN",
  "right_target_following": "YES | NO | UNCERTAIN",
  "right_unexpected_drop_visible": "YES | NO | UNCERTAIN",
  "placement_failure_visible": "YES | NO | UNCERTAIN",
  "unsafe_motion_visible": "YES | NO | UNCERTAIN",
  "evidence": "one short concrete description of the visible motion"
}}
"""


FACT_KEYS = (
    "left_attempt_visible",
    "left_target_following",
    "left_unexpected_drop_visible",
    "right_attempt_visible",
    "right_target_following",
    "right_unexpected_drop_visible",
    "placement_failure_visible",
    "unsafe_motion_visible",
)


def normalize_fact(value: Any) -> str:
    value = common.norm(value)
    return value if value in ALLOWED_FACT_VALUES else ""


def validate_visual_facts(parsed: dict[str, Any]) -> tuple[bool, dict[str, str]]:
    facts = {key: normalize_fact(parsed.get(key)) for key in FACT_KEYS}
    valid = (
        all(facts[key] in ALLOWED_FACT_VALUES for key in FACT_KEYS)
        and bool(str(parsed.get("evidence") or "").strip())
    )
    return valid, facts


def decision_from_visual_facts(
    parsed: dict[str, Any],
) -> tuple[str, str, bool, str, dict[str, str]]:
    """Deterministic trigger policy over VLM-produced visual facts."""
    valid, facts = validate_visual_facts(parsed)
    if not valid:
        # Safety fail-safe for malformed output.
        return "PAUSE", "INVALID_FACTS_FAIL_SAFE", False, "LOW", facts

    hard_failure = (
        facts["left_unexpected_drop_visible"] == "YES"
        or facts["right_unexpected_drop_visible"] == "YES"
        or facts["placement_failure_visible"] == "YES"
        or facts["unsafe_motion_visible"] == "YES"
    )

    left_grasp_miss = (
        facts["left_attempt_visible"] == "YES"
        and facts["left_target_following"] == "NO"
    )
    right_grasp_miss = (
        facts["right_attempt_visible"] == "YES"
        and facts["right_target_following"] == "NO"
    )

    if hard_failure:
        return "PAUSE", "FACT_HARD_FAILURE", True, "HIGH", facts
    if left_grasp_miss or right_grasp_miss:
        return "PAUSE", "FACT_GRASP_MISS", True, "HIGH", facts

    # UNCERTAIN does not itself trigger a pause in this baseline.
    return "CONTINUE", "FACTS_NO_FAILURE", True, "MEDIUM", facts


def make_contact_sheet(
    source: Path,
    destination: Path,
    frame_count: int,
    crop: str | None = None,
) -> None:
    """Render evenly spaced frames into one left-to-right chronological image."""
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    duration = common.video_duration(source)
    fps = frame_count / max(duration, 1e-3)
    filters = [f"fps={fps:.6f}"]
    if crop:
        filters.append(crop)
    filters.extend(["scale=320:240", f"tile={frame_count}x1"])
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            ",".join(filters),
            "-frames:v",
            "1",
            str(destination),
        ],
        check=True,
    )


def build_contact_sheets(window, stream, cache_root: Path, frame_count: int) -> list[tuple[str, Path]]:
    """Build overview plus left/right workspace sheets for each supplied view."""
    sheets: list[tuple[str, Path]] = []
    stamp = f"{int(round(window.start_s * 1000)):08d}_{int(round(window.end_s * 1000)):08d}"
    for label, source in zip(stream.view.split("+"), window.videos):
        root = cache_root / "pause_contact_sheets" / stream.rollout / label
        overview = root / f"{stamp}_overview.jpg"
        make_contact_sheet(source, overview, frame_count)
        sheets.append((f"{label.upper()} OVERVIEW", overview))
        if label == "head":
            # The head camera is 640x480 in this dataset.  Relative crop
            # expressions keep this valid if a later recording changes size.
            left = root / f"{stamp}_left_workspace.jpg"
            right = root / f"{stamp}_right_workspace.jpg"
            make_contact_sheet(
                source,
                left,
                frame_count,
                "crop=iw/2:ih*3/4:0:ih/4",
            )
            make_contact_sheet(
                source,
                right,
                frame_count,
                "crop=iw/2:ih*3/4:iw/2:ih/4",
            )
            sheets.extend(
                [
                    ("HEAD LEFT WORKSPACE", left),
                    ("HEAD RIGHT WORKSPACE", right),
                ]
            )
    return sheets


def build_image_messages(sheets: list[tuple[str, Path]], prompt: str):
    content: list[dict[str, str]] = []
    for label, path in sheets:
        content.append({"type": "text", "text": label})
        content.append({"type": "image", "image": f"file://{path}"})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def generate_from_contact_sheets(model, processor, sheets, prompt: str, args):
    """Run image inference; contact sheets make temporal comparison explicit."""
    messages = build_image_messages(sheets, prompt)
    inputs = common.process_qwen_utils(
        processor, messages, common.thinking_kwargs(args)
    ).to(common.input_device(model))
    common.sync_cuda()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.detection_max_new_tokens,
            do_sample=args.temperature > 0,
            **({"temperature": args.temperature} if args.temperature > 0 else {}),
        )
    common.sync_cuda()
    generation_seconds = time.perf_counter() - start
    input_tokens = int(inputs.input_ids.shape[1])
    generated = output[:, input_tokens:]
    text = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return text, {
        "preprocess_seconds": 0.0,
        "generation_seconds": generation_seconds,
        "end_to_end_seconds": generation_seconds,
        "input_tokens": input_tokens,
        "output_tokens": int(generated.shape[1]),
        "tokens_per_second": int(generated.shape[1]) / generation_seconds if generation_seconds else 0.0,
    }


def validate_trigger_output(parsed: dict[str, Any]) -> tuple[bool, str, str]:
    decision = common.norm(parsed.get("trigger_decision"))
    confidence = common.norm(parsed.get("confidence"))
    valid = (
        {"trigger_decision", "confidence", "evidence"}.issubset(parsed)
        and decision in ALLOWED_DECISIONS
        and confidence in ALLOWED_CONFIDENCE
        and bool(str(parsed.get("evidence") or "").strip())
    )
    return valid, decision, confidence


def final_trigger_decision(parsed: dict[str, Any]) -> tuple[str, str, bool, str]:
    valid, raw_decision, confidence = validate_trigger_output(parsed)
    if not valid:
        # A malformed safety-detector response cannot authorize continuation.
        return "PAUSE", "INVALID_OUTPUT_FAIL_SAFE", False, confidence
    return raw_decision, "NONE", True, confidence


def main() -> None:
    args = parse_args()
    common.require_ffmpeg()
    if args.trigger_use_contact_sheets and args.view_mode != "head":
        raise ValueError(
            "Contact-sheet mode currently expects --view-mode head so it can "
            "supply one overview and two workspace crops."
        )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_name, default_path, short_name = common.MODE_INFO[args.mode]
    model_path = (args.model_path or Path(default_path)).expanduser().resolve()
    if not model_path.is_dir():
        raise NotADirectoryError(f"Model path does not exist: {model_path}")

    annotations = common.load_annotations(args.annotations)
    grouped = common.discover_videos(args.video_dir)
    if args.limit_rollouts:
        keep = set(sorted(grouped)[: args.limit_rollouts])
        grouped = {key: value for key, value in grouped.items() if key in keep}
    streams = common.build_streams(grouped, annotations, args.view_mode)
    if not streams:
        raise RuntimeError("No streams were created. Check filenames and view mode.")

    output_root = args.output_dir.expanduser().resolve()
    logs_dir = output_root / "logs"
    metrics_dir = output_root / "metrics"
    cache_root = output_root / "window_cache"
    for directory in (logs_dir, metrics_dir, cache_root):
        directory.mkdir(parents=True, exist_ok=True)

    suffix = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    base_name = f"{short_name}_{args.view_mode}_facts_trigger_{suffix}"
    log_path = logs_dir / f"{base_name}.log"
    window_csv = metrics_dir / f"{base_name}_windows.csv"
    rollout_csv = metrics_dir / f"{base_name}_rollouts.csv"
    summary_path = metrics_dir / f"{base_name}_summary.json"
    log_path.write_text("", encoding="utf-8")

    torch.cuda.empty_cache()
    common.reset_peak_memory()
    common.sync_cuda()
    load_start = time.perf_counter()
    model, processor, provenance = common.load_model(model_path, args)
    common.sync_cuda()
    model_load_seconds = time.perf_counter() - load_start
    load_memory = common.gpu_memory_mb()

    window_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    warmed = False

    for stream_index, stream in enumerate(streams, 1):
        duration = min(common.video_duration(path) for path in stream.videos)
        ends = common.decision_times(
            duration, args.start_time, args.window_seconds, args.stride_seconds
        )
        if args.max_windows_per_rollout:
            ends = ends[: args.max_windows_per_rollout]

        gt_failure = common.norm(stream.gt.get("failure_detected"))
        gt_time = common.parse_time(stream.gt.get("first_failure_time_s"))
        first_pause_time = None
        first_inference_s = None
        pause_streak = 0
        print(
            f"\n[{stream_index}/{len(streams)}] {stream.stream_id}: "
            f"duration={duration:.2f}s, windows={len(ends)}"
        )

        for window_index, end_s in enumerate(ends, 1):
            clip_start = time.perf_counter()
            window = common.build_window(stream, end_s, args, cache_root)
            clip_seconds = time.perf_counter() - clip_start
            visual_start = time.perf_counter()
            sheets = (
                build_contact_sheets(
                    window, stream, cache_root, args.trigger_contact_frames
                )
                if args.trigger_use_contact_sheets
                else []
            )
            contact_sheet_seconds = time.perf_counter() - visual_start
            prompt = build_visual_facts_prompt(
                args.task_instruction,
                stream.view.split("+"),
                window.start_s,
                window.end_s,
                args.trigger_current_command,
                args.trigger_expected_effect,
            )

            if not warmed and args.warmup:
                for warm_index in range(args.warmup):
                    print(f"Warm-up {warm_index + 1}/{args.warmup}")
                    if args.trigger_use_contact_sheets:
                        generate_from_contact_sheets(
                            model, processor, sheets, prompt, args
                        )
                    else:
                        common.generate(
                            model,
                            processor,
                            stream.view,
                            window.videos,
                            prompt,
                            args,
                            args.detection_max_new_tokens,
                        )
                warmed = True

            common.reset_peak_memory()
            if args.trigger_use_contact_sheets:
                raw, timing = generate_from_contact_sheets(
                    model, processor, sheets, prompt, args
                )
            else:
                raw, timing = common.generate(
                    model,
                    processor,
                    stream.view,
                    window.videos,
                    prompt,
                    args,
                    args.detection_max_new_tokens,
                )
            gpu_memory = common.gpu_memory_mb()
            parsed, json_only = common.extract_json(raw)
            (
                guarded_decision,
                guard_reason,
                schema_valid,
                confidence,
                visual_facts,
            ) = decision_from_visual_facts(parsed)
            # In facts mode, the VLM does not output a trigger decision.
            # This field records the deterministic Python decision for compatibility
            # with the existing metrics pipeline.
            raw_decision = guarded_decision

            if guarded_decision == "PAUSE":
                pause_streak += 1
            else:
                pause_streak = 0
            if (
                guarded_decision == "PAUSE"
                and pause_streak < args.interrupt_confirmations
            ):
                final_decision = "CONTINUE"
                confirmation_status = "PENDING_CONFIRMATION"
            else:
                final_decision = guarded_decision
                confirmation_status = (
                    "CONFIRMED" if final_decision == "PAUSE" else "NOT_APPLICABLE"
                )

            expected_decision = ""
            if gt_failure == "NO":
                expected_decision = "CONTINUE"
            elif gt_failure == "YES" and gt_time is not None:
                expected_decision = "PAUSE" if end_s >= gt_time else "CONTINUE"

            is_pause = final_decision == "PAUSE"
            if is_pause and first_pause_time is None:
                first_pause_time = end_s
                first_inference_s = timing["end_to_end_seconds"]

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
                "expected_trigger_decision": expected_decision,
                "raw_trigger_decision": raw_decision,
                "guarded_trigger_decision": guarded_decision,
                "final_trigger_decision": final_decision,
                "confidence": confidence,
                "guard_reason": guard_reason,
                "pause_streak": pause_streak,
                "confirmation_status": confirmation_status,
                "window_correct": (
                    final_decision == expected_decision if expected_decision else ""
                ),
                "evidence": str(parsed.get("evidence") or ""),
                **visual_facts,
                "first_pause": is_pause and first_pause_time == end_s,
                "false_pause": is_pause and bool(gt_time is None or end_s < gt_time),
                "json_only": json_only,
                "schema_valid": schema_valid,
                "clip_seconds": clip_seconds,
                "contact_sheet_seconds": contact_sheet_seconds,
                "contact_sheets": " | ".join(str(path) for _, path in sheets),
                **timing,
                "peak_allocated_mb": gpu_memory["peak_allocated"],
                "peak_reserved_mb": gpu_memory["peak_reserved"],
            }
            window_rows.append(row)
            print(
                f"  t={end_s:5.1f}s [{window.start_s:4.1f},{end_s:4.1f}] "
                f"GT={expected_decision or '-':8s} raw={raw_decision or '-':8s} "
                f"final={final_decision:8s} confidence={confidence or '-':6s} "
                f"guard={guard_reason} inference={timing['end_to_end_seconds']:.2f}s"
            )
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    "=" * 100
                    + f"\nStream: {stream.stream_id}"
                    + f"\nWindow: {window.start_s:.3f}-{end_s:.3f}s"
                    + f"\nVideos: {' | '.join(str(path) for path in window.videos)}"
                    + f"\nContact sheets: {' | '.join(str(path) for _, path in sheets)}"
                    + f"\nGround truth: {json.dumps(stream.gt, ensure_ascii=False)}"
                    + "\n"
                    + "\n".join(provenance)
                    + f"\nFPS: {args.fps}\n\n=== TRIGGER OUTPUT ===\n{raw}"
                    + f"\n\n=== TRIGGER METRICS ===\n{json.dumps(row, indent=2, ensure_ascii=False)}"
                    + "\n\n"
                )
            if is_pause and args.stop_on_pause:
                print("  [STOP] confirmed PAUSE; later windows are not evaluated")
                break

        triggered_after_gt = bool(
            gt_failure == "YES"
            and gt_time is not None
            and first_pause_time is not None
            and first_pause_time >= gt_time
        )
        false_pause = bool(
            first_pause_time is not None
            and (gt_time is None or first_pause_time < gt_time)
        )
        delay = (
            first_pause_time - gt_time
            if triggered_after_gt and gt_time is not None
            else None
        )
        if gt_failure == "YES":
            if first_pause_time is None:
                outcome = "MISSED"
            elif false_pause:
                outcome = "EARLY_FALSE_PAUSE"
            elif delay is not None and delay <= args.timely_threshold_seconds:
                outcome = "TIMELY_PAUSE"
            else:
                outcome = "LATE_PAUSE"
        else:
            outcome = "CORRECT_CONTINUE" if first_pause_time is None else "FALSE_PAUSE"

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
                "first_pause_time_s": first_pause_time if first_pause_time is not None else "",
                "triggered_after_gt": triggered_after_gt,
                "false_pause": false_pause,
                "trigger_outcome": outcome,
                "detection_delay_s": delay if delay is not None else "",
                "detector_inference_at_pause_s": first_inference_s if first_inference_s is not None else "",
                "estimated_reaction_delay_s": (
                    delay + first_inference_s
                    if delay is not None and first_inference_s is not None
                    else ""
                ),
            }
        )

    for path, rows in ((window_csv, window_rows), (rollout_csv, rollout_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    failures = [r for r in rollout_rows if r["gt_failure_detected"] == "YES"]
    successes = [r for r in rollout_rows if r["gt_failure_detected"] == "NO"]
    valid_pauses = sum(bool(r["triggered_after_gt"]) for r in failures)
    false_pauses = sum(bool(r["false_pause"]) for r in rollout_rows)
    delays = [float(r["detection_delay_s"]) for r in failures if r["detection_delay_s"] != ""]
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
        "trigger_policy": "vlm_visual_facts_plus_python_rule",
        "trigger_use_contact_sheets": args.trigger_use_contact_sheets,
        "trigger_contact_frames": args.trigger_contact_frames,
        "trigger_current_command": args.trigger_current_command or None,
        "trigger_expected_effect": args.trigger_expected_effect or None,
        "pause_confirmations": args.interrupt_confirmations,
        "stop_on_pause": args.stop_on_pause,
        "timely_threshold_seconds": args.timely_threshold_seconds,
        "num_streams": len(rollout_rows),
        "num_windows_evaluated": len(window_rows),
        "pause_recall": valid_pauses / len(failures) if failures else None,
        "pause_precision": (
            valid_pauses / (valid_pauses + false_pauses)
            if valid_pauses + false_pauses
            else None
        ),
        "success_false_pause_rate": (
            sum(bool(r["false_pause"]) for r in successes) / len(successes)
            if successes
            else None
        ),
        "timely_pause_recall": (
            sum(r["trigger_outcome"] == "TIMELY_PAUSE" for r in failures)
            / len(failures)
            if failures
            else None
        ),
        "mean_detection_delay_s": common.mean_or_none(delays),
        "mean_detector_end_to_end_s": common.mean_or_none(
            [r["end_to_end_seconds"] for r in window_rows]
        ),
        "json_only_rate": sum(bool(r["json_only"]) for r in window_rows) / len(window_rows),
        "schema_valid_rate": sum(bool(r["schema_valid"]) for r in window_rows) / len(window_rows),
        "trigger_outcome_counts": {
            label: sum(r["trigger_outcome"] == label for r in rollout_rows)
            for label in (
                "TIMELY_PAUSE",
                "LATE_PAUSE",
                "EARLY_FALSE_PAUSE",
                "MISSED",
                "FALSE_PAUSE",
                "CORRECT_CONTINUE",
            )
        },
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