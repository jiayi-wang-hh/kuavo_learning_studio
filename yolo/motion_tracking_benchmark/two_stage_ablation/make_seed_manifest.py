#!/usr/bin/env python3
"""Create one tracker-independent left/right toy seed manifest from GT boxes."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def is_visible(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select one visible GT frame per rollout/side as common tracker seeds."
    )
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frame", type=int,
        help="Use this annotated frame for every rollout. Default: earliest frame with both toys visible.",
    )
    args = parser.parse_args()

    rows_by_rollout: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.gt.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("side") in {"left_toy", "right_toy"} and is_visible(row.get("visible", "1")):
                rows_by_rollout[row["rollout"]].append(row)

    seeds: list[dict[str, str]] = []
    missing: list[str] = []
    for rollout, rows in sorted(rows_by_rollout.items()):
        by_frame: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
        for row in rows:
            by_frame[int(row["frame"])][row["side"]] = row
        candidate_frames = [args.frame] if args.frame is not None else sorted(by_frame)
        seed_frame = next(
            (frame for frame in candidate_frames if {"left_toy", "right_toy"} <= set(by_frame[frame])),
            None,
        )
        if seed_frame is None:
            missing.append(rollout)
            continue
        for side in ("left_toy", "right_toy"):
            row = by_frame[seed_frame][side]
            seeds.append({key: row[key] for key in ("rollout", "frame", "side", "x1", "y1", "x2", "y2")})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("rollout", "frame", "side", "x1", "y1", "x2", "y2"))
        writer.writeheader()
        writer.writerows(seeds)
    print(f"Wrote {len(seeds) // 2} rollout seed pairs to {args.output}")
    if missing:
        print("No common visible seed frame for: " + ", ".join(missing))


if __name__ == "__main__":
    main()
