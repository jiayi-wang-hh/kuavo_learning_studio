#!/usr/bin/env python3
"""Causal stage-1 VLM trigger that decides only CONTINUE or PAUSE.

This detector intentionally does not classify failure types, plan recovery,
track object state, or decide task completion.  PAUSE hands a short video
buffer to a stronger stage-2 verifier; false pauses may therefore be resumed
by stage 2.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import transformers

from vlm_test import test_vlm_video_joy_trigger as common


ALLOWED_DECISIONS = {"CONTINUE", "PAUSE"}
ALLOWED_CONFIDENCE = {"LOW", "MEDIUM", "HIGH"}
ALLOWED_PHASES = {
    "APPROACH",
    "GRASP",
    "LIFT",
    "TRANSPORT",
    "PLACE",
    "DONE",
    "UNCERTAIN",
}


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
    args = common.parse_args()
    args.trigger_current_command = trigger_args.trigger_current_command
    args.trigger_expected_effect = trigger_args.trigger_expected_effect
    args.trigger_contact_frames = trigger_args.trigger_contact_frames
    args.trigger_use_contact_sheets = trigger_args.trigger_use_contact_sheets
    # Keep --stop-on-interrupt working for compatibility with the shared CLI,
    # but expose the stage-1 meaning under its own name.
    args.stop_on_pause = (
        trigger_args.stop_on_pause
        if trigger_args.stop_on_pause is not None
        else args.stop_on_interrupt
    )
    return args


def build_pause_prompt(
    task: str,
    views: list[str],
    start_s: float,
    end_s: float,
    current_command: str,
    expected_effect: str,
) -> str:
    """Build a one-pass phase-aware failure-detection prompt.

    The VLM first infers the current manipulation phase from the same recent
    visual window, then reasons about the phase-specific expected state change,
    compares it with the observed state change, and finally emits CONTINUE/PAUSE.
    No second VLM call is required for phase estimation.
    """
    controller_context = (
        f"Current controller command: {current_command}\\n"
        f"Controller-provided expected effect: {expected_effect}"
        if current_command or expected_effect
        else (
            "Current controller command: unavailable. Infer the current phase "
            "and its expected visible effect from the task goal and visible motion."
        )
    )

    return f"""You are a fast stage-1 monitor for robot manipulation.

You receive recent chronological visual observations. When contact sheets are
used, tiles run from earliest (left) to latest (right). Compare the beginning
and end of the observation window and reason about how the task state changed.

The observation window covers {start_s:.1f} to {end_s:.1f} seconds.
Use only evidence visible inside this window. Never predict future events.

Task goal:
{task}

Available views:
{", ".join(views)}

{controller_context}

Your job is to make ONE phase-aware decision in this order:

1. Infer the CURRENT MANIPULATION PHASE from the visible behavior.
2. State the observable effect that SHOULD happen in that phase.
3. State the observable effect that ACTUALLY happened in this window.
4. Compare expected versus observed effect.
5. Decide CONTINUE or PAUSE.

Allowed phases:
- APPROACH: the gripper/hand is moving toward or aligning with the correct target.
- PREGRASP: the gripper is approximately aligned with the target in the image, but may still be above it or adjusting its orientation. No grasp attempt has clearly started yet.
- GRASP: A grasp attempt has actually started: the gripper is closing, contacting the object, or is visibly at grasp height.
- LIFT: a grasped object should leave the support surface and move with the gripper.
- TRANSPORT: a held object should move with the gripper toward the correct destination.
- PLACE: the object should enter/reach the correct destination and be stably released.
- DONE: the required manipulation is visibly completed.
- UNCERTAIN: the current phase cannot be inferred reliably from this window.

Critical progress rule:
ROBOT MOTION ALONE IS NOT TASK PROGRESS.
Progress must be supported by an observable task-state change that is appropriate
for the inferred phase.

Examples:
- APPROACH progress: gripper gets meaningfully closer/aligned with the correct target.
- GRASP progress: the target becomes securely acquired by the gripper.
- LIFT progress: the target leaves the support surface and moves together with the gripper.
- TRANSPORT progress: the held target moves with the gripper toward the correct destination.
- PLACE progress: the target enters/reaches the correct destination and is released stably.

Return PAUSE when a completed or clearly attempted manipulation does NOT produce
its expected state change, including:
- the gripper closes or moves away but the target remains behind,
- the target does not move together with the gripper after a grasp/lift attempt,
- the object slips, falls, or is dropped,
- the robot repeatedly moves but the task state does not improve,
- the wrong object is manipulated,
- an object is transported toward the wrong destination,
- a placement attempt ends with the object outside the destination or unstable on its edge,
- unsafe contact, collision, or clearly abnormal motion occurs,
- the outcome of an important completed grasp/lift/place attempt is visually
  ambiguous or cannot be verified.

Return CONTINUE when:
- the current action is still genuinely in progress and no failed outcome is yet visible,
- the phase-specific expected state change is visibly occurring,
- transport/place is proceeding normally,
- or the task is visibly completed successfully.

Important:
- Do NOT use "the arm is still moving", "the hand is near the object", or
  "the robot appears to be trying" as sufficient evidence for CONTINUE.
- For GRASP/LIFT, explicitly check whether the OBJECT moves with the gripper.
- If the robot has moved away after an attempted grasp and the object stayed
  where it was, this is PAUSE even if the robot continues moving afterward.
- If current_phase is UNCERTAIN, do not invent a phase. Use directly visible
  evidence and prefer PAUSE only when a suspicious completed outcome is visible
  or important outcome verification is impossible.
- Do not diagnose a detailed failure type and do not plan recovery.
- Keep expected_effect, observed_effect, and evidence short and concrete.
- Output exactly one JSON object and nothing else.

{{
  "current_phase": "APPROACH | GRASP | LIFT | TRANSPORT | PLACE | DONE | UNCERTAIN",
  "expected_effect": "short phase-specific visible state change that should occur",
  "observed_effect": "short visible state change that actually occurred",
  "trigger_decision": "CONTINUE | PAUSE",
  "confidence": "LOW | MEDIUM | HIGH",
  "evidence": "one short directly visible observation supporting the decision"
}}
"""


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


def validate_trigger_output(
    parsed: dict[str, Any],
) -> tuple[bool, str, str, str, str, str]:
    """Validate the phase-aware stage-1 output."""
    phase = common.norm(parsed.get("current_phase"))
    decision = common.norm(parsed.get("trigger_decision"))
    confidence = common.norm(parsed.get("confidence"))
    expected_effect = str(parsed.get("expected_effect") or "").strip()
    observed_effect = str(parsed.get("observed_effect") or "").strip()
    evidence = str(parsed.get("evidence") or "").strip()

    required = {
        "current_phase",
        "expected_effect",
        "observed_effect",
        "trigger_decision",
        "confidence",
        "evidence",
    }
    valid = (
        required.issubset(parsed)
        and phase in ALLOWED_PHASES
        and decision in ALLOWED_DECISIONS
        and confidence in ALLOWED_CONFIDENCE
        and bool(expected_effect)
        and bool(observed_effect)
        and bool(evidence)
    )
    return (
        valid,
        phase,
        decision,
        confidence,
        expected_effect,
        observed_effect,
    )


def final_trigger_decision(
    parsed: dict[str, Any],
) -> tuple[str, str, bool, str, str, str, str]:
    """Return the guarded phase-aware decision.

    The model remains the primary reasoner. This function only:
    1) fails safe on malformed output; and
    2) catches a small set of direct textual contradictions where the model
       explicitly describes an obvious failed state change but says CONTINUE.
    """
    (
        valid,
        phase,
        raw_decision,
        confidence,
        expected_effect,
        observed_effect,
    ) = validate_trigger_output(parsed)

    if not valid:
        return (
            "PAUSE",
            "INVALID_OUTPUT_FAIL_SAFE",
            False,
            phase,
            confidence,
            expected_effect,
            observed_effect,
        )

    evidence = str(parsed.get("evidence") or "").strip()
    failure_text = f"{observed_effect} {evidence}".lower()

    # Intentionally narrow: the prompt is the main semantic reasoner.
    direct_failure_signals = (
        "remains on the table",
        "remained on the table",
        "remains behind",
        "remained behind",
        "left behind",
        "did not grasp",
        "failed to grasp",
        "missed the object",
        "not acquired",
        "was not acquired",
        "did not move with",
        "does not move with",
        "slipped",
        "dropped",
        "fell",
        "wrong object",
        "wrong destination",
        "outside the container",
        "outside the basket",
        "on the edge",
        "no task progress",
        "no progress",
    )

    if raw_decision == "CONTINUE" and any(
        signal in failure_text for signal in direct_failure_signals
    ):
        return (
            "PAUSE",
            "OBSERVED_EFFECT_CONTRADICTION",
            True,
            phase,
            confidence,
            expected_effect,
            observed_effect,
        )

    return (
        raw_decision,
        "NONE",
        True,
        phase,
        confidence,
        expected_effect,
        observed_effect,
    )


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
    base_name = f"{short_name}_{args.view_mode}_pause_trigger_{suffix}"
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
            prompt = build_pause_prompt(
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
            raw_decision = common.norm(parsed.get("trigger_decision"))
            (
                guarded_decision,
                guard_reason,
                schema_valid,
                current_phase,
                confidence,
                inferred_expected_effect,
                observed_effect,
            ) = final_trigger_decision(parsed)

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
                "current_phase": current_phase,
                "inferred_expected_effect": inferred_expected_effect,
                "observed_effect": observed_effect,
                "confidence": confidence,
                "guard_reason": guard_reason,
                "pause_streak": pause_streak,
                "confirmation_status": confirmation_status,
                "window_correct": (
                    final_decision == expected_decision if expected_decision else ""
                ),
                "evidence": str(parsed.get("evidence") or ""),
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
                f"final={final_decision:8s} phase={current_phase or '-':9s} "
                f"confidence={confidence or '-':6s} guard={guard_reason} "
                f"inference={timing['end_to_end_seconds']:.2f}s"
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
        "trigger_use_contact_sheets": args.trigger_use_contact_sheets,
        "trigger_contact_frames": args.trigger_contact_frames,
        "trigger_current_command": args.trigger_current_command or None,
        "trigger_expected_effect": args.trigger_expected_effect or None,
        "phase_aware_trigger": True,
        "allowed_phases": sorted(ALLOWED_PHASES),
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
        "phase_counts": {
            phase: sum(r["current_phase"] == phase for r in window_rows)
            for phase in sorted(ALLOWED_PHASES)
        },
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