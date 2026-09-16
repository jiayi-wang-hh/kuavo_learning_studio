#!/usr/bin/env python3
"""Metric evaluator for the two-stage initializer/tracker ablation."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


KEY = ("rollout", "frame", "side")


def box_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def center_error(a, b):
    return math.hypot((a[0]+a[2]-b[0]-b[2])/2, (a[1]+a[3]-b[1]-b[3])/2)


def load_rows(path, prediction=False):
    rows = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("side") not in {"left_toy", "right_toy"}:
                continue
            if not prediction and str(row.get("visible", "1")).lower() not in {"1", "true", "yes"}:
                continue
            try:
                box = tuple(float(row[name]) for name in ("x1", "y1", "x2", "y2"))
                key = (row["rollout"], int(row["frame"]), row["side"])
            except (KeyError, ValueError):
                continue
            current = rows.get(key)
            # For duplicate detections keep the highest-confidence prediction.
            score = float(row.get("score") or row.get("confidence") or 0.0)
            if current is None or score > current[1]:
                rows[key] = (box, score)
    return rows


def evaluate(gt, pred, iou_threshold):
    ious, errors, matched = [], [], 0
    by_side = defaultdict(lambda: [0, 0])
    for key, (gt_box, _) in gt.items():
        side = key[2]; by_side[side][1] += 1
        if key not in pred:
            continue
        pred_box, _ = pred[key]
        iou = box_iou(gt_box, pred_box)
        ious.append(iou); errors.append(center_error(gt_box, pred_box))
        if iou >= iou_threshold:
            matched += 1; by_side[side][0] += 1
    frame_pairs = defaultdict(set)
    matched_pairs = defaultdict(set)
    for rollout, frame, side in gt:
        frame_pairs[(rollout, frame)].add(side)
    for key, (gt_box, _) in gt.items():
        if key not in pred or box_iou(gt_box, pred[key][0]) < iou_threshold:
            continue
        matched_pairs[(key[0], key[1])].add(key[2])
    both_visible = sum(sides == {"left_toy", "right_toy"} for sides in frame_pairs.values())
    both_matched = sum(
        sides == {"left_toy", "right_toy"} and matched_pairs[pair] == sides
        for pair, sides in frame_pairs.items()
    )
    return {
        "gt_visible_boxes": len(gt),
        "predicted_boxes": len(pred),
        "matched_at_iou": matched,
        "recall_at_iou": matched / len(gt) if gt else None,
        "mean_iou": sum(ious) / len(ious) if ious else None,
        "mean_center_error_px": sum(errors) / len(errors) if errors else None,
        "side_recall": {side: (value[0] / value[1] if value[1] else None) for side, value in by_side.items()},
        "both_toys_recall_at_iou": both_matched / both_visible if both_visible else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("initializer", "tracker"))
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(load_rows(args.gt), load_rows(args.pred, prediction=True), args.iou_threshold)
    result.update({"stage": args.stage, "condition": args.name, "iou_threshold": args.iou_threshold, "gt": str(args.gt), "pred": str(args.pred)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
