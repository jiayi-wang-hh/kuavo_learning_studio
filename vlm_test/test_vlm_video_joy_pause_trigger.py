#!/usr/bin/env python3
"""Causal stage-1 VLM trigger that decides only CONTINUE or PAUSE.

This detector intentionally does not classify failure types, plan recovery,
track object state, or decide task completion.  PAUSE hands a short video
buffer to a stronger stage-2 verifier; false pauses may therefore be resumed
by stage 2.
"""

from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
import transformers

import test_vlm_video_joy_trigger as common


ALLOWED_DECISIONS = {"CONTINUE", "PAUSE"}
ALLOWED_CONFIDENCE = {"LOW", "MEDIUM", "HIGH"}


def build_pause_prompt(
    task: str,
    views: list[str],
    start_s: float,
    end_s: float,
) -> str:
    return f"""You are a fast stage-1 safety trigger for robot manipulation.

{common.view_statement(views)} The video window covers {start_s:.1f} to {end_s:.1f} seconds.
Use only this video window. Never predict what happens after it.

Task:
{task}

Decide whether execution may CONTINUE or should PAUSE for review by a stronger
vision model. Do not diagnose the failure type and do not plan recovery.

Return PAUSE when any of these is visible:
- A gripper closes and then lifts or moves away while its target remains behind.
- An object slips, falls, or separates unexpectedly from a gripper.
- A placement attempt finishes but the object is visibly outside the container
  or resting unstably on its edge.
- Unsafe contact, collision, or another clearly abnormal motion occurs.
- The outcome of a completed grasp, lift, or placement attempt is important but
  cannot be verified because it is occluded or visually ambiguous.

Return CONTINUE when:
- The robot is approaching, aligning, or performing an action that is still in
  progress and no suspicious outcome is visible.
- A grasp, transport, or placement is visibly proceeding normally.
- The task is visibly successful. Successful completion is not a failure.

Important rules:
- Prefer PAUSE for a suspicious or unverifiable completed action; stage 2 will
  reject false alarms and resume execution.
- A toy merely being on the table is not by itself suspicious.
- Do not require seeing the exact instant of gripper closure. If the gripper has
  moved up or away and the target was left behind, return PAUSE.
- Evidence must describe the directly visible reason in one short sentence.
- Output exactly one JSON object and nothing else.

{{
  "trigger_decision": "CONTINUE | PAUSE",
  "confidence": "LOW | MEDIUM | HIGH",
  "evidence": "one short directly visible observation"
}}
"""


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
    args = common.parse_args()
    common.require_ffmpeg()
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
            prompt = build_pause_prompt(
                args.task_instruction,
                stream.view.split("+"),
                window.start_s,
                window.end_s,
            )

            if not warmed and args.warmup:
                for warm_index in range(args.warmup):
                    print(f"Warm-up {warm_index + 1}/{args.warmup}")
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
            guarded_decision, guard_reason, schema_valid, confidence = (
                final_trigger_decision(parsed)
            )

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
                "first_pause": is_pause and first_pause_time == end_s,
                "false_pause": is_pause and bool(gt_time is None or end_s < gt_time),
                "json_only": json_only,
                "schema_valid": schema_valid,
                "clip_seconds": clip_seconds,
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
                    + f"\nGround truth: {json.dumps(stream.gt, ensure_ascii=False)}"
                    + "\n"
                    + "\n".join(provenance)
                    + f"\nFPS: {args.fps}\n\n=== TRIGGER OUTPUT ===\n{raw}"
                    + f"\n\n=== TRIGGER METRICS ===\n{json.dumps(row, indent=2, ensure_ascii=False)}"
                    + "\n\n"
                )
            if is_pause and args.stop_on_interrupt:
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
        "pause_confirmations": args.interrupt_confirmations,
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
