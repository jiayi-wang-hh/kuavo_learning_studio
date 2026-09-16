#!/usr/bin/env python3
"""Convert benchmark tracks.csv files to the canonical two-stage CSV schema."""
from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path


SIDES = {"left_toy", "right_toy"}


def rollout_from_path(path: Path, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(str(path))
    return match.group(1) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True, help="Quoted glob for one condition's tracks.csv files.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-regex", default=r"(rollout[^/\\_]*)", help="Regex with rollout in capture group 1.")
    parser.add_argument("--include-sources", nargs="*", help="Optional allow-list, e.g. SAM2_PROPAGATED TRACKED.")
    args = parser.parse_args()
    pattern = re.compile(args.rollout_regex, re.IGNORECASE)
    allowed = set(args.include_sources or [])
    out_rows: list[dict[str, object]] = []
    skipped: list[str] = []
    for raw_path in sorted(glob.glob(args.input_glob, recursive=True)):
        path = Path(raw_path)
        rollout = rollout_from_path(path, pattern)
        if rollout is None:
            skipped.append(str(path))
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("label") not in SIDES or (allowed and row.get("source") not in allowed):
                    continue
                try:
                    out_rows.append({
                        "rollout": rollout,
                        "frame": int(row["frame"]),
                        "side": row["label"],
                        "x1": float(row["x1"]), "y1": float(row["y1"]),
                        "x2": float(row["x2"]), "y2": float(row["y2"]),
                        "score": float(row.get("confidence") or 0.0),
                        "source": row.get("source", ""),
                    })
                except (KeyError, ValueError):
                    continue
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("rollout", "frame", "side", "x1", "y1", "x2", "y2", "score", "source"))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} canonical predictions to {args.output}")
    if skipped:
        print("Skipped files without a rollout match: " + ", ".join(skipped))


if __name__ == "__main__":
    main()
