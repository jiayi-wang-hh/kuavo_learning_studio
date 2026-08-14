#!/usr/bin/env python3
"""Audit whether chunk-blending cosine regressions occur at low-speed boundaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


KUAVO_ARM_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14])


def safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < 1e-12:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def boundary_rows(
    chunks: np.ndarray,
    *,
    input_label: str,
    candidate: str,
    trajectory: str,
    min_velocity_norm: float,
) -> list[dict[str, Any]]:
    arm = chunks[:, :, KUAVO_ARM_INDICES]
    rows: list[dict[str, Any]] = []
    for boundary_id in range(arm.shape[0] - 1):
        previous_velocity = arm[boundary_id, -1] - arm[boundary_id, -2]
        next_velocity = arm[boundary_id + 1, 1] - arm[boundary_id + 1, 0]
        previous_norm = float(np.linalg.norm(previous_velocity))
        next_norm = float(np.linalg.norm(next_velocity))
        valid = min(previous_norm, next_norm) >= min_velocity_norm
        rows.append(
            {
                "input": input_label,
                "candidate": candidate,
                "trajectory": trajectory,
                "boundary_id": boundary_id,
                "position_jump_l2": float(
                    np.linalg.norm(arm[boundary_id + 1, 0] - arm[boundary_id, -1])
                ),
                "previous_velocity_norm": previous_norm,
                "next_velocity_norm": next_norm,
                "min_velocity_norm": min(previous_norm, next_norm),
                "cosine": safe_cosine(previous_velocity, next_velocity),
                "cosine_valid": valid,
            }
        )
    return rows


def summarize_candidate(
    baseline_rows: list[dict[str, Any]],
    blended_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paired: list[dict[str, Any]] = []
    for baseline, blended in zip(baseline_rows, blended_rows):
        both_valid = bool(baseline["cosine_valid"] and blended["cosine_valid"])
        delta = (
            float(blended["cosine"] - baseline["cosine"])
            if both_valid
            else None
        )
        paired.append(
            {
                "input": baseline["input"],
                "candidate": baseline["candidate"],
                "boundary_id": baseline["boundary_id"],
                "baseline_position_jump_l2": baseline["position_jump_l2"],
                "blended_position_jump_l2": blended["position_jump_l2"],
                "baseline_min_velocity_norm": baseline["min_velocity_norm"],
                "blended_min_velocity_norm": blended["min_velocity_norm"],
                "baseline_cosine": baseline["cosine"],
                "blended_cosine": blended["cosine"],
                "both_cosines_valid": both_valid,
                "cosine_delta": delta,
                "direction_reversal": bool(
                    both_valid and baseline["cosine"] >= 0.0 and blended["cosine"] < 0.0
                ),
                "severe_regression": bool(both_valid and delta is not None and delta <= -0.2),
            }
        )
    valid = [row for row in paired if row["both_cosines_valid"]]
    deltas = np.asarray([row["cosine_delta"] for row in valid], dtype=np.float64)
    summary = {
        "boundary_count": len(paired),
        "valid_cosine_count": len(valid),
        "low_speed_or_invalid_count": len(paired) - len(valid),
        "cosine_delta_mean": float(np.mean(deltas)) if deltas.size else None,
        "cosine_delta_median": float(np.median(deltas)) if deltas.size else None,
        "cosine_delta_p05": float(np.percentile(deltas, 5)) if deltas.size else None,
        "cosine_regression_count": sum(row["cosine_delta"] < 0 for row in valid),
        "severe_regression_count": sum(row["severe_regression"] for row in paired),
        "direction_reversal_count": sum(row["direction_reversal"] for row in paired),
    }
    return summary, paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["B_cosine8_w025", "C_cosine8_w050", "D_linear8_w025"],
    )
    parser.add_argument(
        "--min-velocity-norm",
        type=float,
        default=0.01,
        help="Cosine is invalid when either adjacent arm velocity norm is below this value.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.min_velocity_norm <= 0:
        parser.error("--min-velocity-norm must be positive")

    matrix_dir = args.matrix_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    summaries: dict[str, dict[str, Any]] = defaultdict(dict)
    all_rows: list[dict[str, Any]] = []

    input_dirs = sorted(path for path in matrix_dir.iterdir() if path.is_dir())
    if not input_dirs:
        parser.error(f"No matrix input directories found under {matrix_dir}")
    for input_dir in input_dirs:
        for candidate in args.candidates:
            data_path = input_dir / candidate / "blended_chunks.npz"
            if not data_path.is_file():
                parser.error(f"Missing matrix data: {data_path}")
            source = np.load(data_path)
            baseline = np.asarray(source["baseline_executed_chunks"], dtype=np.float64)
            blended = np.asarray(source["blended_executed_chunks"], dtype=np.float64)
            baseline_rows = boundary_rows(
                baseline,
                input_label=input_dir.name,
                candidate=candidate,
                trajectory="baseline",
                min_velocity_norm=args.min_velocity_norm,
            )
            blended_rows = boundary_rows(
                blended,
                input_label=input_dir.name,
                candidate=candidate,
                trajectory="blended",
                min_velocity_norm=args.min_velocity_norm,
            )
            summary, paired = summarize_candidate(baseline_rows, blended_rows)
            summaries[input_dir.name][candidate] = summary
            all_rows.extend(paired)

    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    for task in sorted({row["input"].split("_", 1)[0] for row in all_rows}):
        for candidate in args.candidates:
            rows = [
                row
                for row in all_rows
                if row["input"].startswith(f"{task}_")
                and row["candidate"] == candidate
                and row["both_cosines_valid"]
            ]
            deltas = np.asarray([row["cosine_delta"] for row in rows], dtype=np.float64)
            grouped[task][candidate] = {
                "valid_cosine_count": len(rows),
                "cosine_delta_mean": float(np.mean(deltas)) if deltas.size else None,
                "cosine_delta_median": float(np.median(deltas)) if deltas.size else None,
                "cosine_delta_p05": float(np.percentile(deltas, 5)) if deltas.size else None,
                "cosine_regression_count": sum(row["cosine_delta"] < 0 for row in rows),
                "severe_regression_count": sum(row["severe_regression"] for row in rows),
                "direction_reversal_count": sum(row["direction_reversal"] for row in rows),
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "boundary_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    report = {
        "experiment": {
            "matrix_dir": str(matrix_dir),
            "candidates": args.candidates,
            "min_velocity_norm": args.min_velocity_norm,
        },
        "per_input": summaries,
        "per_task": grouped,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote boundary audit to {output_dir}")


if __name__ == "__main__":
    main()
