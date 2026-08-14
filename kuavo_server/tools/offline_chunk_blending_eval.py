#!/usr/bin/env python3
"""Evaluate lightweight action-chunk blending from saved jitter diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


KUAVO_ARM_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14])
KUAVO_GRIPPER_INDICES = np.asarray([7, 15])
CANDIDATE_MATRIX = (
    ("A_cosine8_w000", "cosine", 8, 0.00),
    ("B_cosine8_w025", "cosine", 8, 0.25),
    ("C_cosine8_w050", "cosine", 8, 0.50),
    ("D_linear8_w025", "linear", 8, 0.25),
)


def summarize(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None, "p05": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def blend_weights(length: int, mode: str, new_chunk_start_weight: float) -> np.ndarray:
    if not 0.0 <= new_chunk_start_weight <= 1.0:
        raise ValueError("new_chunk_start_weight must be in [0, 1]")
    if length < 1:
        return np.empty((0,), dtype=np.float64)
    if length == 1:
        return np.asarray([new_chunk_start_weight], dtype=np.float64)
    progress = np.linspace(0.0, 1.0, length)
    if mode == "linear":
        base = progress
    elif mode == "cosine":
        base = 0.5 - 0.5 * np.cos(np.pi * progress)
    else:
        raise ValueError(f"Unknown blend mode: {mode}")
    return new_chunk_start_weight + (1.0 - new_chunk_start_weight) * base


def apply_gripper_hysteresis(
    executed: np.ndarray,
    *,
    open_threshold: float,
    close_threshold: float,
) -> tuple[np.ndarray, dict[str, int]]:
    if not 0.0 <= open_threshold < close_threshold <= 1.0:
        raise ValueError("Require 0 <= open_threshold < close_threshold <= 1")
    result = executed.copy()
    switches: dict[str, int] = {}
    for gripper_index in KUAVO_GRIPPER_INDICES:
        first = float(result[0, 0, gripper_index])
        state = 1.0 if first >= 0.5 else 0.0
        switch_count = 0
        for chunk_index in range(result.shape[0]):
            for step_index in range(result.shape[1]):
                command = float(result[chunk_index, step_index, gripper_index])
                previous = state
                if command >= close_threshold:
                    state = 1.0
                elif command <= open_threshold:
                    state = 0.0
                if state != previous:
                    switch_count += 1
                result[chunk_index, step_index, gripper_index] = state
        switches[str(gripper_index)] = switch_count
    return result, switches


def construct_executed_chunks(
    predicted_chunks: np.ndarray,
    *,
    execution_horizon: int,
    blend_steps: int,
    mode: str,
    new_chunk_start_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    if predicted_chunks.ndim != 3:
        raise ValueError(f"Expected predicted chunks [N,T,D], got {predicted_chunks.shape}")
    model_horizon = predicted_chunks.shape[1]
    if execution_horizon < 2 or execution_horizon > model_horizon:
        raise ValueError("execution_horizon must be in [2, model_horizon]")
    available_tail = model_horizon - execution_horizon
    effective_blend = min(blend_steps, execution_horizon, available_tail)
    if effective_blend < 1:
        raise ValueError(
            "No overlap available: blend_steps, execution_horizon, and model tail must be positive"
        )

    baseline = predicted_chunks[:, :execution_horizon, :].copy()
    blended = baseline.copy()
    alpha = blend_weights(effective_blend, mode, new_chunk_start_weight)[:, None]
    for chunk_index in range(1, predicted_chunks.shape[0]):
        previous_tail = predicted_chunks[
            chunk_index - 1,
            execution_horizon : execution_horizon + effective_blend,
            :,
        ][:, KUAVO_ARM_INDICES]
        current_head = predicted_chunks[
            chunk_index,
            :effective_blend,
            :,
        ][:, KUAVO_ARM_INDICES]
        blended_head = blended[chunk_index, :effective_blend, :]
        blended_head[:, KUAVO_ARM_INDICES] = (
            (1.0 - alpha) * previous_tail + alpha * current_head
        )
    return baseline, blended


def trajectory_metrics(
    executed: np.ndarray,
    dataset_action: np.ndarray,
    *,
    fps: float,
) -> tuple[dict[str, Any], list[dict[str, float | int]]]:
    arm = executed[:, :, KUAVO_ARM_INDICES]
    dt = 1.0 / fps
    acceleration = np.diff(arm, n=2, axis=1) / (dt * dt)
    acceleration_norm = np.linalg.norm(acceleration, axis=-1)

    boundary_delta = arm[1:, 0, :] - arm[:-1, -1, :]
    boundary_jump = np.linalg.norm(boundary_delta, axis=-1)
    previous_velocity = arm[:-1, -1, :] - arm[:-1, -2, :]
    next_velocity = arm[1:, 1, :] - arm[1:, 0, :]
    velocity_cosine = np.sum(previous_velocity * next_velocity, axis=-1) / (
        np.linalg.norm(previous_velocity, axis=-1)
        * np.linalg.norm(next_velocity, axis=-1)
        + 1e-8
    )

    comparable = min(dataset_action.shape[-1], executed.shape[-1])
    action_dims = KUAVO_ARM_INDICES[KUAVO_ARM_INDICES < comparable]
    first_action_error = np.linalg.norm(
        executed[:, 0, action_dims] - dataset_action[:, action_dims], axis=-1
    )
    metrics = {
        "arm_intra_chunk_acceleration": summarize(acceleration_norm),
        "arm_boundary_position_jump": summarize(boundary_jump),
        "arm_boundary_velocity_cosine": summarize(velocity_cosine),
        "arm_first_action_vs_dataset_l2": summarize(first_action_error),
    }
    rows = [
        {
            "boundary_id": index,
            "position_jump_l2": float(boundary_jump[index]),
            "velocity_cosine": float(velocity_cosine[index]),
        }
        for index in range(boundary_jump.shape[0])
    ]
    return metrics, rows


def relative_change(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or abs(before) < 1e-12:
        return None
    return float((after - before) / abs(before))


def write_boundary_csv(
    path: Path,
    baseline_rows: list[dict[str, float | int]],
    blended_rows: list[dict[str, float | int]],
) -> None:
    rows = []
    for baseline, blended in zip(baseline_rows, blended_rows):
        rows.append(
            {
                "boundary_id": baseline["boundary_id"],
                "baseline_position_jump_l2": baseline["position_jump_l2"],
                "blended_position_jump_l2": blended["position_jump_l2"],
                "baseline_velocity_cosine": baseline["velocity_cosine"],
                "blended_velocity_cosine": blended["velocity_cosine"],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_input(
    *,
    input_dir: Path,
    output_dir: Path,
    execution_horizon: int,
    blend_steps: int,
    blend_mode: str,
    new_chunk_start_weight: float,
    fps: float,
    gripper_mode: str,
    gripper_open_threshold: float,
    gripper_close_threshold: float,
) -> dict[str, Any]:
    input_path = input_dir.expanduser().resolve() / "chunks.npz"
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing diagnostic input: {input_path}")
    source = np.load(input_path, allow_pickle=True)
    predicted = np.asarray(source["kuavo_action_chunks"], dtype=np.float64)
    dataset_action = np.asarray(source["dataset_action"], dtype=np.float64)
    if predicted.shape[-1] != 16:
        raise ValueError(f"Expected Kuavo 16-D actions, got {predicted.shape}")

    baseline, blended = construct_executed_chunks(
        predicted,
        execution_horizon=execution_horizon,
        blend_steps=blend_steps,
        mode=blend_mode,
        new_chunk_start_weight=new_chunk_start_weight,
    )
    baseline_switches = None
    blended_switches = None
    if gripper_mode == "hysteresis":
        baseline, baseline_switches = apply_gripper_hysteresis(
            baseline,
            open_threshold=gripper_open_threshold,
            close_threshold=gripper_close_threshold,
        )
        blended, blended_switches = apply_gripper_hysteresis(
            blended,
            open_threshold=gripper_open_threshold,
            close_threshold=gripper_close_threshold,
        )

    baseline_metrics, baseline_rows = trajectory_metrics(baseline, dataset_action, fps=fps)
    blended_metrics, blended_rows = trajectory_metrics(blended, dataset_action, fps=fps)
    jump_before = baseline_metrics["arm_boundary_position_jump"]
    jump_after = blended_metrics["arm_boundary_position_jump"]
    cosine_before = baseline_metrics["arm_boundary_velocity_cosine"]
    cosine_after = blended_metrics["arm_boundary_velocity_cosine"]
    report = {
        "experiment": {
            "input_dir": str(input_dir.expanduser().resolve()),
            "sample_count": int(predicted.shape[0]),
            "model_horizon": int(predicted.shape[1]),
            "execution_horizon": execution_horizon,
            "blend_steps": min(blend_steps, execution_horizon, predicted.shape[1] - execution_horizon),
            "blend_mode": blend_mode,
            "new_chunk_start_weight": new_chunk_start_weight,
            "fps": fps,
            "gripper_mode": gripper_mode,
            "gripper_open_threshold": gripper_open_threshold,
            "gripper_close_threshold": gripper_close_threshold,
        },
        "baseline": baseline_metrics,
        "blended": blended_metrics,
        "comparison": {
            "boundary_jump_mean_relative_change": relative_change(jump_before["mean"], jump_after["mean"]),
            "boundary_jump_p95_relative_change": relative_change(jump_before["p95"], jump_after["p95"]),
            "velocity_cosine_p05_change": None if cosine_before["p05"] is None or cosine_after["p05"] is None else float(cosine_after["p05"] - cosine_before["p05"]),
            "velocity_cosine_median_change": None if cosine_before["median"] is None or cosine_after["median"] is None else float(cosine_after["median"] - cosine_before["median"]),
        },
        "gripper_switch_count": {"baseline": baseline_switches, "blended": blended_switches},
    }

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "blended_chunks.npz",
        baseline_executed_chunks=baseline,
        blended_executed_chunks=blended,
        dataset_action=dataset_action,
        frame_indices=source["frame_indices"],
    )
    write_boundary_csv(output_dir / "boundary_comparison.csv", baseline_rows, blended_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    return report


def aggregate_row(input_label: str, candidate: str, report: dict[str, Any]) -> dict[str, Any]:
    baseline = report["baseline"]
    blended = report["blended"]
    comparison = report["comparison"]
    return {
        "input": input_label,
        "candidate": candidate,
        "blend_mode": report["experiment"]["blend_mode"],
        "start_weight": report["experiment"]["new_chunk_start_weight"],
        "boundary_mean_baseline": baseline["arm_boundary_position_jump"]["mean"],
        "boundary_mean_blended": blended["arm_boundary_position_jump"]["mean"],
        "boundary_mean_relative_change": comparison["boundary_jump_mean_relative_change"],
        "boundary_p95_baseline": baseline["arm_boundary_position_jump"]["p95"],
        "boundary_p95_blended": blended["arm_boundary_position_jump"]["p95"],
        "boundary_p95_relative_change": comparison["boundary_jump_p95_relative_change"],
        "velocity_cosine_p05_change": comparison["velocity_cosine_p05_change"],
        "velocity_cosine_median_change": comparison["velocity_cosine_median_change"],
        "acceleration_p95_baseline": baseline["arm_intra_chunk_acceleration"]["p95"],
        "acceleration_p95_blended": blended["arm_intra_chunk_acceleration"]["p95"],
        "first_action_error_mean_baseline": baseline["arm_first_action_vs_dataset_l2"]["mean"],
        "first_action_error_mean_blended": blended["arm_first_action_vs_dataset_l2"]["mean"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more offline_jitter_diagnostic output directories.",
    )
    parser.add_argument(
        "--candidate-matrix",
        action="store_true",
        help="Evaluate the documented A/B/C/D blending matrix for every input directory.",
    )
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--blend-steps", type=int, default=8)
    parser.add_argument("--blend-mode", choices=["linear", "cosine"], default="cosine")
    parser.add_argument(
        "--new-chunk-start-weight",
        type=float,
        default=0.0,
        help="Weight of the new prediction at the first blended step; range [0,1].",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--gripper-mode", choices=["passthrough", "hysteresis"], default="hysteresis"
    )
    parser.add_argument("--gripper-open-threshold", type=float, default=0.35)
    parser.add_argument("--gripper-close-threshold", type=float, default=0.65)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.blend_steps < 1 or args.fps <= 0:
        parser.error("--blend-steps and --fps must be positive")
    if not 0.0 <= args.new_chunk_start_weight <= 1.0:
        parser.error("--new-chunk-start-weight must be in [0,1]")

    output_dir = args.output_dir.expanduser().resolve()
    if not args.candidate_matrix and len(args.input_dir) != 1:
        parser.error("Multiple --input-dir values require --candidate-matrix")

    if not args.candidate_matrix:
        try:
            report = evaluate_input(
                input_dir=args.input_dir[0], output_dir=output_dir,
                execution_horizon=args.execution_horizon, blend_steps=args.blend_steps,
                blend_mode=args.blend_mode, new_chunk_start_weight=args.new_chunk_start_weight,
                fps=args.fps, gripper_mode=args.gripper_mode,
                gripper_open_threshold=args.gripper_open_threshold,
                gripper_close_threshold=args.gripper_close_threshold,
            )
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Wrote blending evaluation to {output_dir}")
        return

    rows: list[dict[str, Any]] = []
    used_labels: set[str] = set()
    for input_dir in args.input_dir:
        resolved = input_dir.expanduser().resolve()
        label = f"{resolved.parent.name}_{resolved.name}"
        if label in used_labels:
            parser.error(f"Input label collision for {resolved}; use distinct parent/input names")
        used_labels.add(label)
        for candidate, mode, steps, weight in CANDIDATE_MATRIX:
            candidate_output = output_dir / label / candidate
            try:
                report = evaluate_input(
                    input_dir=resolved, output_dir=candidate_output,
                    execution_horizon=args.execution_horizon, blend_steps=steps,
                    blend_mode=mode, new_chunk_start_weight=weight,
                    fps=args.fps, gripper_mode=args.gripper_mode,
                    gripper_open_threshold=args.gripper_open_threshold,
                    gripper_close_threshold=args.gripper_close_threshold,
                )
            except (FileNotFoundError, ValueError) as error:
                parser.error(str(error))
            rows.append(aggregate_row(label, candidate, report))
            print(f"[{label}] {candidate} -> {candidate_output}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "aggregate_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "aggregate_summary.json").open("w", encoding="utf-8") as stream:
        json.dump({"candidate_matrix": CANDIDATE_MATRIX, "results": rows}, stream, indent=2)
    print(f"Wrote {len(rows)} evaluations and aggregate summaries to {output_dir}")


if __name__ == "__main__":
    main()
