#!/usr/bin/env python3
"""Replay causal runtime chunk blending and audit timing and arm safety limits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from offline_chunk_blending_eval import (
    KUAVO_ARM_INDICES,
    KUAVO_GRIPPER_INDICES,
    apply_gripper_hysteresis,
    blend_weights,
    summarize,
)


def causal_blend(
    predicted: np.ndarray,
    *,
    execution_horizon: int,
    blend_steps: int,
    mode: str,
    start_weight: float,
) -> np.ndarray:
    """Blend in arrival order, retaining only the previous raw chunk as runtime state."""
    model_horizon = predicted.shape[1]
    effective_blend = min(blend_steps, execution_horizon, model_horizon - execution_horizon)
    if effective_blend < 1:
        raise ValueError("Model horizon must leave at least one unexecuted tail action")
    alpha = blend_weights(effective_blend, mode, start_weight)[:, None]
    executed: list[np.ndarray] = []
    previous_raw: np.ndarray | None = None
    for current_raw in predicted:
        current = current_raw[:execution_horizon].copy()
        if previous_raw is not None:
            old_tail = previous_raw[
                execution_horizon : execution_horizon + effective_blend,
                KUAVO_ARM_INDICES,
            ]
            new_head = current[:effective_blend, KUAVO_ARM_INDICES]
            current[:effective_blend, KUAVO_ARM_INDICES] = (
                (1.0 - alpha) * old_tail + alpha * new_head
            )
        executed.append(current)
        previous_raw = current_raw.copy()
    return np.stack(executed)


def replay_metrics(executed: np.ndarray, *, fps: float) -> dict[str, Any]:
    arm = executed[:, :, KUAVO_ARM_INDICES]
    flat = arm.reshape(-1, arm.shape[-1])
    step = np.linalg.norm(np.diff(flat, axis=0), axis=-1)
    velocity = np.diff(flat, axis=0) * fps
    acceleration = np.diff(flat, n=2, axis=0) * fps * fps
    return {
        "arm_step_l2": summarize(step),
        "arm_velocity_l2_per_s": summarize(np.linalg.norm(velocity, axis=-1)),
        "arm_acceleration_l2_per_s2": summarize(np.linalg.norm(acceleration, axis=-1)),
    }


def count_limit_violations(
    executed: np.ndarray,
    *,
    fps: float,
    max_step_l2: float,
    max_velocity_l2: float,
    max_acceleration_l2: float,
) -> dict[str, int]:
    arm = executed[:, :, KUAVO_ARM_INDICES].reshape(-1, len(KUAVO_ARM_INDICES))
    step = np.linalg.norm(np.diff(arm, axis=0), axis=-1)
    velocity = step * fps
    acceleration = np.linalg.norm(np.diff(arm, n=2, axis=0), axis=-1) * fps * fps
    return {
        "arm_step_l2": int(np.sum(step > max_step_l2)),
        "arm_velocity_l2_per_s": int(np.sum(velocity > max_velocity_l2)),
        "arm_acceleration_l2_per_s2": int(np.sum(acceleration > max_acceleration_l2)),
    }


def timing_rows(
    latency_ms: np.ndarray,
    *,
    input_label: str,
    execution_horizon: int,
    fps: float,
) -> list[dict[str, Any]]:
    budget_ms = execution_horizon / fps * 1000.0
    rows = []
    for chunk_id, latency in enumerate(latency_ms):
        warmup = chunk_id == 0
        margin = None if warmup else float(budget_ms - latency)
        rows.append(
            {
                "input": input_label,
                "chunk_id": chunk_id,
                "warmup": warmup,
                "latency_ms": float(latency),
                "budget_ms": budget_ms,
                "deadline_margin_ms": margin,
                "deadline_miss": bool(not warmup and latency > budget_ms),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--blend-steps", type=int, default=8)
    parser.add_argument("--blend-mode", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--new-chunk-start-weight", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.35)
    parser.add_argument("--gripper-close-threshold", type=float, default=0.65)
    parser.add_argument("--max-arm-step-l2", type=float, default=0.25)
    parser.add_argument("--max-arm-velocity-l2", type=float, default=2.5)
    parser.add_argument("--max-arm-acceleration-l2", type=float, default=25.0)
    parser.add_argument("--reference-matrix-dir", type=Path)
    parser.add_argument("--reference-candidate", default="C_cosine8_w050")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.execution_horizon < 2 or args.blend_steps < 1 or args.fps <= 0:
        parser.error("Invalid execution horizon, blend steps, or FPS")
    if not 0.0 <= args.new_chunk_start_weight <= 1.0:
        parser.error("--new-chunk-start-weight must be in [0,1]")
    if min(args.max_arm_step_l2, args.max_arm_velocity_l2, args.max_arm_acceleration_l2) <= 0:
        parser.error("Safety limits must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    all_timing_rows: list[dict[str, Any]] = []
    labels: set[str] = set()
    for input_dir in args.input_dir:
        resolved = input_dir.expanduser().resolve()
        label = f"{resolved.parent.name}_{resolved.name}"
        if label in labels:
            parser.error(f"Input label collision: {label}")
        labels.add(label)
        data_path = resolved / "chunks.npz"
        if not data_path.is_file():
            parser.error(f"Missing diagnostic input: {data_path}")
        source = np.load(data_path, allow_pickle=True)
        predicted = np.asarray(source["kuavo_action_chunks"], dtype=np.float64)
        latency_ms = np.asarray(source["request_latency_ms"], dtype=np.float64)
        executed = causal_blend(
            predicted,
            execution_horizon=args.execution_horizon,
            blend_steps=args.blend_steps,
            mode=args.blend_mode,
            start_weight=args.new_chunk_start_weight,
        )
        executed, gripper_switches = apply_gripper_hysteresis(
            executed,
            open_threshold=args.gripper_open_threshold,
            close_threshold=args.gripper_close_threshold,
        )

        reference_max_error = None
        if args.reference_matrix_dir:
            reference_path = (
                args.reference_matrix_dir.expanduser().resolve()
                / label
                / args.reference_candidate
                / "blended_chunks.npz"
            )
            if not reference_path.is_file():
                parser.error(f"Missing offline reference: {reference_path}")
            reference = np.load(reference_path)["blended_executed_chunks"]
            reference_max_error = float(np.max(np.abs(executed - reference)))

        rows = timing_rows(
            latency_ms,
            input_label=label,
            execution_horizon=args.execution_horizon,
            fps=args.fps,
        )
        all_timing_rows.extend(rows)
        measured = np.asarray([row["latency_ms"] for row in rows if not row["warmup"]])
        margins = np.asarray([row["deadline_margin_ms"] for row in rows if not row["warmup"]])
        reports[label] = {
            "sample_count": int(predicted.shape[0]),
            "causal_reference_max_abs_error": reference_max_error,
            "metrics": replay_metrics(executed, fps=args.fps),
            "limit_violations": count_limit_violations(
                executed,
                fps=args.fps,
                max_step_l2=args.max_arm_step_l2,
                max_velocity_l2=args.max_arm_velocity_l2,
                max_acceleration_l2=args.max_arm_acceleration_l2,
            ),
            "gripper_switch_count": gripper_switches,
            "timing": {
                "chunk_budget_ms": args.execution_horizon / args.fps * 1000.0,
                "latency_ms": summarize(measured),
                "deadline_margin_ms": summarize(margins),
                "deadline_miss_count": sum(row["deadline_miss"] for row in rows),
                "warmup_latency_ms": float(latency_ms[0]),
            },
        }
        np.savez_compressed(
            output_dir / f"{label}_runtime_replay.npz",
            executed_chunks=executed,
            frame_indices=source["frame_indices"],
        )

    with (output_dir / "timing.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_timing_rows[0]))
        writer.writeheader()
        writer.writerows(all_timing_rows)
    report = {
        "experiment": {
            "execution_horizon": args.execution_horizon,
            "blend_steps": args.blend_steps,
            "blend_mode": args.blend_mode,
            "new_chunk_start_weight": args.new_chunk_start_weight,
            "fps": args.fps,
            "safety_limits": {
                "max_arm_step_l2": args.max_arm_step_l2,
                "max_arm_velocity_l2": args.max_arm_velocity_l2,
                "max_arm_acceleration_l2": args.max_arm_acceleration_l2,
            },
        },
        "inputs": reports,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote runtime replay to {output_dir}")


if __name__ == "__main__":
    main()
