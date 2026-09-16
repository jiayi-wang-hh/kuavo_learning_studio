#!/usr/bin/env python3
"""
Evaluate a simple V1 pregrasp-timeout + post-activation stagnation failure trigger.

Inputs
------
1) One or more YOLO-World + CSRT tracking CSVs produced by
   test_yolo_world_visual_tracking.py.

2) The existing JSON GT format used by the joy task, e.g.
   {
     "rollouts": {
       "rollout01": {
         "outcome": "FAILURE",
         "first_failure_time_s": 6.5,
         ...
       },
       ...
     }
   }

Core idea
---------
For each toy independently:
- estimate two static basket centers from all accepted basket detections;
- use distance to the NEAREST basket because either basket is valid;
- PREGRASP: wait until the toy has visibly moved from its initial position;
- if the toy still has not activated by a timeout, emit PAUSE;
- once activated, monitor the toy's own motion over a temporal window;
- if the toy remains almost stationary for long enough, emit PAUSE;
- basket distance is used only as a coarse DONE cue, not as a requirement
  that transport must monotonically approach the basket;
- DETECTED/TRACKED are reliable; PREDICTED/LOST freeze the timer.

This script is intentionally simple: it tests whether "long-term lack of
task progress" is already useful as a Stage-1 trigger before adding
gripper/EE-specific rules.

Outputs
-------
Per rollout:
  *_progress_trigger.csv
  *_progress_trigger_summary.json

Batch:
  batch_summary.csv
  batch_summary.json

Key metrics
-----------
- success_false_pause_rate
- failure_detection_rate
- event_precision
- event_recall
- mean/median trigger delay relative to GT first_failure_time_s
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import statistics
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple


RELIABLE_SOURCES = {"DETECTED", "TRACKED"}
TOY_CLASSES = ("left_toy", "right_toy")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input",
        required=True,
        help=(
            "Tracking CSV or glob, e.g. "
            "'outputs/yolo_world_visual_tracking/*_visual_tracking.csv'"
        ),
    )
    p.add_argument(
        "--gt-json",
        type=Path,
        required=True,
        help="Existing joy annotation JSON.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/progress_trigger_eval"),
    )

    # Trigger parameters.
    p.add_argument(
        "--activation-motion",
        type=float,
        default=0.04,
        help=(
            "Toy displacement from its initial reliable position required to activate "
            "post-grasp monitoring. A slightly larger threshold reduces CSRT-drift activation."
        ),
    )
    p.add_argument(
        "--stagnation-window-s",
        type=float,
        default=1.0,
        help="Lookback window used to measure the toy's own motion.",
    )
    p.add_argument(
        "--min-motion",
        type=float,
        default=0.012,
        help=(
            "Minimum normalized toy displacement over stagnation-window-s. "
            "Below this value the toy is considered stationary."
        ),
    )
    p.add_argument(
        "--pause-after-s",
        type=float,
        default=2.0,
        help="Continuous post-activation stagnation duration required before PAUSE.",
    )
    p.add_argument(
        "--pregrasp-timeout-s",
        type=float,
        default=6.5,
        help=(
            "If a toy has not activated (has not moved enough from its initial "
            "position) by this time, emit a pregrasp-timeout PAUSE. "
            "This catches grasp-miss / no-manipulation-start failures."
        ),
    )
    p.add_argument(
        "--activation-confirm-frames",
        type=int,
        default=3,
        help=(
            "Number of consecutive reliable frames above activation-motion "
            "required before entering transport monitoring."
        ),
    )
    p.add_argument(
        "--success-distance",
        type=float,
        default=0.10,
        help="Nearest-basket center distance below which a toy is considered placed/done.",
    )
    # Reliability / debounce.
    p.add_argument(
        "--allow-predicted",
        action="store_true",
        help="Treat PREDICTED toy boxes as reliable. Default: no.",
    )
    p.add_argument(
        "--confirm-frames",
        type=int,
        default=2,
        help="Consecutive trigger-positive frames required for final PAUSE.",
    )
    p.add_argument(
        "--cooldown-s",
        type=float,
        default=999.0,
        help="Suppress repeated PAUSE events after the first one in a rollout.",
    )

    # Evaluation.
    p.add_argument(
        "--early-tolerance-s",
        type=float,
        default=0.5,
        help=(
            "A trigger up to this many seconds before GT first_failure_time_s "
            "is still counted as a valid detection."
        ),
    )
    p.add_argument(
        "--late-tolerance-s",
        type=float,
        default=3.0,
        help=(
            "A trigger later than GT first_failure_time_s + this value counts "
            "as a miss for event-level evaluation."
        ),
    )

    return p.parse_args()


def is_true(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes"}


def dist_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def median(vals: List[float]) -> Optional[float]:
    return statistics.median(vals) if vals else None


def rollout_key_from_filename(path: Path) -> Optional[str]:
    m = re.search(r"(rollout\d+)", path.stem, flags=re.IGNORECASE)
    return m.group(1).lower() if m else None


def load_gt(path: Path) -> Dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rollouts = data.get("rollouts", {})
    return {str(k).lower(): v for k, v in rollouts.items()}


def load_tracking_csv(path: Path):
    """
    Returns:
      frames: {frame_id: {...}}
      basket_centers: [(x,y), (x,y)] estimated globally for the rollout
    """
    frames = defaultdict(
        lambda: {
            "time_s": None,
            "toys": {},
            "basket_detections": [],
        }
    )

    basket_rows = []

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if not is_true(row.get("accepted", "")):
                continue

            frame = int(row["frame"])
            t = float(row["time_s"])
            cls = row["class"]

            try:
                cx = float(row["side_center_x"])
                cy = float(row["side_center_y"])
            except (TypeError, ValueError):
                continue

            frames[frame]["time_s"] = t

            if cls in TOY_CLASSES:
                frames[frame]["toys"][cls] = {
                    "x": cx,
                    "y": cy,
                    "source": row.get("track_source", "") or "",
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                }

            elif cls == "basket":
                frames[frame]["basket_detections"].append((cx, cy))
                basket_rows.append((cx, cy))

    if not basket_rows:
        raise RuntimeError(
            f"No accepted basket detections found in {path}. "
            "This trigger needs basket locations."
        )

    # Baskets are static. Split detections by x around their global median,
    # then take median center for each cluster.
    xs = [x for x, _ in basket_rows]
    split = statistics.median(xs)

    left = [(x, y) for x, y in basket_rows if x <= split]
    right = [(x, y) for x, y in basket_rows if x > split]

    # If clustering degenerates, fall back to min/max-x halves.
    if not left or not right:
        basket_rows_sorted = sorted(basket_rows, key=lambda p: p[0])
        half = max(1, len(basket_rows_sorted) // 2)
        left = basket_rows_sorted[:half]
        right = basket_rows_sorted[half:]

    if not right:
        # Rare fallback: only one stable basket detected.
        one = (
            statistics.median([x for x, _ in left]),
            statistics.median([y for _, y in left]),
        )
        basket_centers = [one]
    else:
        basket_centers = [
            (
                statistics.median([x for x, _ in left]),
                statistics.median([y for _, y in left]),
            ),
            (
                statistics.median([x for x, _ in right]),
                statistics.median([y for _, y in right]),
            ),
        ]

    return dict(sorted(frames.items())), basket_centers


def toy_reliable(toy: Optional[dict], allow_predicted: bool) -> bool:
    if toy is None:
        return False
    source = toy.get("source", "")
    if source in RELIABLE_SOURCES:
        return True
    return allow_predicted and source == "PREDICTED"


def nearest_basket_distance(
    toy_xy: Tuple[float, float],
    baskets: List[Tuple[float, float]],
) -> Tuple[float, int]:
    ds = [dist_xy(toy_xy, b) for b in baskets]
    idx = min(range(len(ds)), key=lambda i: ds[i])
    return ds[idx], idx


def find_history_at_or_before(
    history: deque,
    target_time: float,
) -> Optional[dict]:
    """Return history item with time closest to target_time."""
    if not history:
        return None
    return min(history, key=lambda h: abs(h["time_s"] - target_time))


def process_rollout(
    path: Path,
    gt: dict,
    args,
):
    frames, baskets = load_tracking_csv(path)

    toy_state = {
        "left": {
            "initial_xy": None,
            "activated": False,
            "activation_streak": 0,
            "done": False,
            "no_progress_s": 0.0,
            "history": deque(maxlen=1000),
        },
        "right": {
            "initial_xy": None,
            "activated": False,
            "activation_streak": 0,
            "done": False,
            "no_progress_s": 0.0,
            "history": deque(maxlen=1000),
        },
    }

    rows = []
    trigger_streak = 0
    last_pause_t = -1e9
    first_pause_t = None
    first_pause_side = None

    prev_time = None

    for frame_id, fr in frames.items():
        t = float(fr["time_s"])
        dt = 0.0 if prev_time is None else max(0.0, t - prev_time)
        prev_time = t

        side_outputs = {}
        any_side_raw_trigger = False
        raw_trigger_sides = []

        for side, toy_cls in (("left", "left_toy"), ("right", "right_toy")):
            st = toy_state[side]
            toy = fr["toys"].get(toy_cls)
            reliable = toy_reliable(toy, args.allow_predicted)

            out = {
                "reliable": reliable,
                "source": toy.get("source", "") if toy else "",
                "distance": None,
                "nearest_basket": None,
                "moved_from_start": None,
                "progress": None,
                "toy_motion": None,
                "activated": st["activated"],
                "done": st["done"],
                "no_progress_s": st["no_progress_s"],
                "raw_trigger": False,
                "reason": "",
            }

            if not reliable:
                # Freeze timer while perception is uncertain.
                out["reason"] = "vision_uncertain"
                side_outputs[side] = out
                continue

            toy_xy = (float(toy["x"]), float(toy["y"]))
            d_now, basket_idx = nearest_basket_distance(toy_xy, baskets)

            if st["initial_xy"] is None:
                st["initial_xy"] = toy_xy

            moved_from_start = dist_xy(toy_xy, st["initial_xy"])

            # Task success / placed side.
            if d_now <= args.success_distance:
                st["done"] = True
                st["no_progress_s"] = 0.0

            # Activate only after the toy itself has visibly moved for several
            # consecutive reliable frames. This avoids CSRT drift causing a false
            # transition out of PREGRASP.
            if not st["activated"] and not st["done"]:
                if moved_from_start >= args.activation_motion:
                    st["activation_streak"] += 1
                else:
                    st["activation_streak"] = 0

                if st["activation_streak"] >= args.activation_confirm_frames:
                    st["activated"] = True
                    st["no_progress_s"] = 0.0
            else:
                st["activation_streak"] = 0

            history = st["history"]
            past = find_history_at_or_before(
                history,
                t - args.stagnation_window_s,
            )

            progress = None  # diagnostic only; no longer used for triggering
            toy_motion = None

            if past is not None:
                progress = past["distance"] - d_now
                toy_motion = dist_xy(past["xy"], toy_xy)

            raw_trigger = False
            reason = ""

            if st["done"]:
                reason = "done_near_basket"

            elif not st["activated"]:
                # PREGRASP phase: before the toy starts moving we do not evaluate
                # basket progress. But if this phase lasts too long, treat it as a
                # generic failure trigger (e.g. grasp miss / no manipulation start).
                if t >= args.pregrasp_timeout_s:
                    raw_trigger = True
                    reason = (
                        f"pregrasp_timeout: t={t:.2f}s, "
                        f"moved_from_start={moved_from_start:.4f}, "
                        f"activation_streak={st['activation_streak']}"
                    )
                else:
                    reason = (
                        f"pregrasp_wait: t={t:.2f}s, "
                        f"moved_from_start={moved_from_start:.4f}, "
                        f"activation_streak={st['activation_streak']}"
                    )

            elif past is None:
                reason = "warming_window"

            else:
                # Post-activation V1 trigger:
                # do NOT require the toy to monotonically approach the basket.
                # Only ask whether the toy itself has become stationary.
                low_motion = toy_motion < args.min_motion

                if low_motion:
                    st["no_progress_s"] += dt
                else:
                    st["no_progress_s"] = 0.0

                if st["no_progress_s"] >= args.pause_after_s:
                    raw_trigger = True
                    reason = (
                        f"stagnation: toy_motion={toy_motion:.4f}, "
                        f"distance_to_basket={d_now:.4f}, "
                        f"timer={st['no_progress_s']:.2f}s"
                    )
                else:
                    reason = (
                        f"monitoring_motion: toy_motion={toy_motion:.4f}, "
                        f"distance_to_basket={d_now:.4f}, "
                        f"timer={st['no_progress_s']:.2f}s"
                    )

            history.append(
                {
                    "time_s": t,
                    "xy": toy_xy,
                    "distance": d_now,
                }
            )

            out.update(
                {
                    "distance": d_now,
                    "nearest_basket": basket_idx,
                    "moved_from_start": moved_from_start,
                    "progress": progress,
                    "toy_motion": toy_motion,
                    "activated": st["activated"],
                    "done": st["done"],
                    "no_progress_s": st["no_progress_s"],
                    "raw_trigger": raw_trigger,
                    "reason": reason,
                }
            )

            side_outputs[side] = out

            if raw_trigger:
                any_side_raw_trigger = True
                raw_trigger_sides.append(side)

        if any_side_raw_trigger:
            trigger_streak += 1
        else:
            trigger_streak = 0

        cooldown_ok = (t - last_pause_t) >= args.cooldown_s
        final_pause = (
            any_side_raw_trigger
            and trigger_streak >= args.confirm_frames
            and cooldown_ok
        )

        if final_pause:
            last_pause_t = t
            if first_pause_t is None:
                first_pause_t = t
                first_pause_side = "+".join(raw_trigger_sides)

        rows.append(
            {
                "frame": frame_id,
                "time_s": f"{t:.3f}",
                "left_source": side_outputs["left"]["source"],
                "right_source": side_outputs["right"]["source"],
                "left_reliable": side_outputs["left"]["reliable"],
                "right_reliable": side_outputs["right"]["reliable"],
                "left_distance": (
                    "" if side_outputs["left"]["distance"] is None
                    else f"{side_outputs['left']['distance']:.5f}"
                ),
                "right_distance": (
                    "" if side_outputs["right"]["distance"] is None
                    else f"{side_outputs['right']['distance']:.5f}"
                ),
                "left_progress": (
                    "" if side_outputs["left"]["progress"] is None
                    else f"{side_outputs['left']['progress']:.5f}"
                ),
                "right_progress": (
                    "" if side_outputs["right"]["progress"] is None
                    else f"{side_outputs['right']['progress']:.5f}"
                ),
                "left_toy_motion": (
                    "" if side_outputs["left"]["toy_motion"] is None
                    else f"{side_outputs['left']['toy_motion']:.5f}"
                ),
                "right_toy_motion": (
                    "" if side_outputs["right"]["toy_motion"] is None
                    else f"{side_outputs['right']['toy_motion']:.5f}"
                ),
                "left_activated": side_outputs["left"]["activated"],
                "right_activated": side_outputs["right"]["activated"],
                "left_activation_streak": toy_state["left"]["activation_streak"],
                "right_activation_streak": toy_state["right"]["activation_streak"],
                "left_done": side_outputs["left"]["done"],
                "right_done": side_outputs["right"]["done"],
                "left_stagnation_s": f"{side_outputs['left']['no_progress_s']:.3f}",
                "right_stagnation_s": f"{side_outputs['right']['no_progress_s']:.3f}",
                "left_raw_trigger": side_outputs["left"]["raw_trigger"],
                "right_raw_trigger": side_outputs["right"]["raw_trigger"],
                "raw_trigger": any_side_raw_trigger,
                "trigger_streak": trigger_streak,
                "final_pause": final_pause,
                "trigger_sides": "+".join(raw_trigger_sides),
                "left_reason": side_outputs["left"]["reason"],
                "right_reason": side_outputs["right"]["reason"],
            }
        )

    rollout_key = rollout_key_from_filename(path)
    outcome = str(gt.get("outcome", "")).upper()
    gt_failure = outcome == "FAILURE" or str(gt.get("failure_detected", "")).upper() == "YES"
    gt_first_failure = gt.get("first_failure_time_s", None)

    pred_failure = first_pause_t is not None

    # Event-level classification.
    if not gt_failure:
        event_result = "FP" if pred_failure else "TN"
        delay = None
        detected_in_window = False
    else:
        if not pred_failure:
            event_result = "FN"
            delay = None
            detected_in_window = False
        else:
            delay = float(first_pause_t) - float(gt_first_failure)
            detected_in_window = (
                -args.early_tolerance_s
                <= delay
                <= args.late_tolerance_s
            )
            event_result = "TP" if detected_in_window else "FN_LATE_OR_EARLY"

    out_csv = args.output_dir / f"{path.stem}_progress_trigger.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "rollout": rollout_key,
        "input_csv": str(path),
        "basket_centers": baskets,
        "gt_outcome": outcome,
        "gt_failure_detected": gt.get("failure_detected"),
        "gt_first_failure_time_s": gt_first_failure,
        "gt_failure_type": gt.get("failure_type"),
        "pred_pause": pred_failure,
        "pred_first_pause_time_s": first_pause_t,
        "pred_first_pause_side": first_pause_side,
        "trigger_delay_s": delay,
        "detected_in_window": detected_in_window,
        "event_result": event_result,
        "params": {
            "activation_motion": args.activation_motion,
            "activation_confirm_frames": args.activation_confirm_frames,
            "pregrasp_timeout_s": args.pregrasp_timeout_s,
            "stagnation_window_s": args.stagnation_window_s,
            "min_motion": args.min_motion,
            "pause_after_s": args.pause_after_s,
            "success_distance": args.success_distance,
            "confirm_frames": args.confirm_frames,
        },
    }

    out_json = args.output_dir / f"{path.stem}_progress_trigger_summary.json"
    out_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return summary


def resolve_inputs(pattern: str) -> List[Path]:
    p = Path(pattern)
    if p.is_file():
        return [p]
    return [Path(x) for x in sorted(glob.glob(pattern)) if Path(x).is_file()]


def aggregate(summaries: List[dict]) -> dict:
    success = [s for s in summaries if s["gt_outcome"] == "SUCCESS"]
    failure = [s for s in summaries if s["gt_outcome"] == "FAILURE"]

    success_fp = sum(bool(s["pred_pause"]) for s in success)
    valid_failure_detections = sum(bool(s["detected_in_window"]) for s in failure)

    tp = valid_failure_detections
    fn = len(failure) - tp
    fp = success_fp

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None

    delays = [
        s["trigger_delay_s"]
        for s in failure
        if s["detected_in_window"] and s["trigger_delay_s"] is not None
    ]

    return {
        "num_rollouts": len(summaries),
        "num_success_rollouts": len(success),
        "num_failure_rollouts": len(failure),
        "success_false_pause_count": success_fp,
        "success_false_pause_rate": (
            success_fp / len(success) if success else None
        ),
        "failure_detected_count": valid_failure_detections,
        "failure_detection_rate": (
            valid_failure_detections / len(failure) if failure else None
        ),
        "event_precision": precision,
        "event_recall": recall,
        "mean_trigger_delay_s": (
            statistics.mean(delays) if delays else None
        ),
        "median_trigger_delay_s": (
            statistics.median(delays) if delays else None
        ),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = resolve_inputs(args.input)
    if not inputs:
        raise FileNotFoundError(f"No tracking CSV matched: {args.input}")

    gt_all = load_gt(args.gt_json)

    summaries = []

    for path in inputs:
        key = rollout_key_from_filename(path)
        if key is None:
            print(f"[SKIP] cannot infer rolloutXX from {path.name}")
            continue

        if key not in gt_all:
            print(f"[SKIP] no GT entry for {key}: {path.name}")
            continue

        summary = process_rollout(
            path=path,
            gt=gt_all[key],
            args=args,
        )
        summaries.append(summary)

        print(
            f"{key}: GT={summary['gt_outcome']} "
            f"gt_t={summary['gt_first_failure_time_s']} "
            f"pause={summary['pred_first_pause_time_s']} "
            f"delay={summary['trigger_delay_s']} "
            f"result={summary['event_result']}"
        )

    if not summaries:
        raise RuntimeError("No rollouts were evaluated.")

    batch_csv = args.output_dir / "batch_summary.csv"
    fieldnames = [
        "rollout",
        "gt_outcome",
        "gt_failure_detected",
        "gt_first_failure_time_s",
        "gt_failure_type",
        "pred_pause",
        "pred_first_pause_time_s",
        "pred_first_pause_side",
        "trigger_delay_s",
        "detected_in_window",
        "event_result",
    ]

    with batch_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s.get(k) for k in fieldnames})

    metrics = aggregate(summaries)

    batch_json = args.output_dir / "batch_summary.json"
    batch_json.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "rollouts": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== Overall ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print(f"\nPer-rollout CSVs: {args.output_dir}")
    print(f"Batch summary CSV: {batch_csv}")
    print(f"Batch summary JSON: {batch_json}")


if __name__ == "__main__":
    main()
