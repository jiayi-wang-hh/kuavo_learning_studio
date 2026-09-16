#!/usr/bin/env python3
"""
Offline failure-trigger evaluator for YOLO-World + temporal-tracker CSV outputs.

Expected tracker CSV columns (from test_yolo_world_visual_tracking.py):
    frame,time_s,class,confidence,x1,y1,x2,y2,accepted,
    rejection_reason,side_center_x,side_center_y,track_source,
    missing_frames,vx_px,vy_px,assoc_dist,visual_tracker,
    tracker_ok,tracker_reason

This script:
1) reconstructs per-frame left/right toy trajectories;
2) reconstructs left/right gripper trajectories from accepted "robot gripper" boxes;
3) computes simple temporal features;
4) fires three prototype failure triggers:
       GRASP_MISS
       OBJECT_DROP
       STAGNATION
5) writes a per-frame CSV and a summary JSON;
6) optionally evaluates against a lightweight GT CSV.

Important:
- This is a V1 offline trigger benchmark, not a final controller.
- DETECTED/TRACKED toy states are treated as reliable observations.
- PREDICTED/LOST are treated as vision-uncertain by default.
- Grasp/drop rules require gripper observations. If grippers are not detected,
  those rules will abstain rather than hallucinate.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


RELIABLE_TOY_SOURCES = {"DETECTED", "TRACKED"}
TOY_CLASSES = ("left_toy", "right_toy")

GRIPPER_CLASS = "robot gripper"


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input",
        required=True,
        help=(
            "One tracker CSV or a glob pattern, e.g. "
            "'outputs/yolo_world_visual_tracking/*_visual_tracking.csv'"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/failure_trigger_eval"),
    )

    # Optional GT file.
    # Columns:
    # rollout,start_s,end_s,label
    # label: NORMAL | GRASP_MISS | OBJECT_DROP | STAGNATION | FAILURE
    p.add_argument(
        "--gt-csv",
        type=Path,
        default=None,
        help="Optional interval GT CSV: rollout,start_s,end_s,label",
    )

    # Temporal windows (seconds).
    p.add_argument("--motion-window-s", type=float, default=0.5)
    p.add_argument("--following-window-s", type=float, default=0.6)
    p.add_argument("--stagnation-window-s", type=float, default=1.0)

    # Reliability.
    p.add_argument(
        "--allow-predicted-toy",
        action="store_true",
        help="Allow PREDICTED toy boxes as trigger evidence. Not recommended initially.",
    )
    p.add_argument(
        "--max-toy-gap-s",
        type=float,
        default=0.35,
        help="Maximum gap when looking back for reliable toy motion.",
    )
    p.add_argument(
        "--max-gripper-gap-s",
        type=float,
        default=0.35,
        help="Maximum gap when looking back for gripper motion.",
    )

    # Normalized image-space thresholds.
    p.add_argument(
        "--near-distance",
        type=float,
        default=0.12,
        help="Toy-gripper center distance considered near/contact-like.",
    )
    p.add_argument(
        "--follow-distance",
        type=float,
        default=0.14,
        help="Toy-gripper distance allowed while considering them coupled.",
    )
    p.add_argument(
        "--gripper-lift-dy",
        type=float,
        default=0.035,
        help=(
            "Minimum upward gripper displacement over motion window. "
            "Image y increases downward, so upward motion is negative dy."
        ),
    )
    p.add_argument(
        "--toy-follow-min-dy",
        type=float,
        default=0.018,
        help="Minimum upward toy displacement expected when it follows a lifting gripper.",
    )
    p.add_argument(
        "--drop-distance-increase",
        type=float,
        default=0.08,
        help="Required increase in toy-gripper distance for OBJECT_DROP.",
    )
    p.add_argument(
        "--drop-down-dy",
        type=float,
        default=0.025,
        help="Required downward toy displacement for OBJECT_DROP.",
    )
    p.add_argument(
        "--following-distance-std",
        type=float,
        default=0.025,
        help="Max std of toy-gripper distance over following window.",
    )
    p.add_argument(
        "--stagnation-motion",
        type=float,
        default=0.015,
        help="Max normalized displacement over stagnation window.",
    )

    # Debounce.
    p.add_argument(
        "--confirm-frames",
        type=int,
        default=2,
        help="Consecutive raw trigger frames required for final PAUSE.",
    )
    p.add_argument(
        "--cooldown-s",
        type=float,
        default=1.0,
        help="After final PAUSE, suppress new final PAUSE events for this long.",
    )

    # Evaluation.
    p.add_argument(
        "--match-tolerance-s",
        type=float,
        default=1.0,
        help="GT-event matching tolerance for trigger delay/event precision-recall.",
    )

    return p.parse_args()


@dataclass
class Obs:
    time_s: float
    frame: int
    cls: str
    x: float
    y: float
    source: str = ""
    confidence: float = 0.0


def finite(v: Optional[float]) -> bool:
    return v is not None and math.isfinite(v)


def dist(a: Optional[Obs], b: Optional[Obs]) -> Optional[float]:
    if a is None or b is None:
        return None
    return math.hypot(a.x - b.x, a.y - b.y)


def displacement(a: Optional[Obs], b: Optional[Obs]) -> Optional[Tuple[float, float, float]]:
    if a is None or b is None:
        return None
    dx = b.x - a.x
    dy = b.y - a.y
    return dx, dy, math.hypot(dx, dy)


def read_tracker_csv(path: Path) -> Dict[int, dict]:
    frames: Dict[int, dict] = defaultdict(
        lambda: {
            "time_s": None,
            "toys": {},
            "grippers": [],
        }
    )

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if str(row.get("accepted", "")).strip().lower() not in {"true", "1", "yes"}:
                continue

            frame = int(row["frame"])
            t = float(row["time_s"])
            cls = row["class"]

            frames[frame]["time_s"] = t

            try:
                x = float(row["side_center_x"])
                y = float(row["side_center_y"])
            except (TypeError, ValueError):
                try:
                    x1 = float(row["x1"])
                    y1 = float(row["y1"])
                    x2 = float(row["x2"])
                    y2 = float(row["y2"])
                    # Cannot normalize without frame size; tracker output already
                    # contains normalized centers for accepted rows, so skip if absent.
                    continue
                except Exception:
                    continue

            obs = Obs(
                time_s=t,
                frame=frame,
                cls=cls,
                x=x,
                y=y,
                source=row.get("track_source", "") or "",
                confidence=float(row.get("confidence", 0.0) or 0.0),
            )

            if cls in TOY_CLASSES:
                frames[frame]["toys"][cls] = obs
            elif cls == GRIPPER_CLASS:
                frames[frame]["grippers"].append(obs)

    return dict(sorted(frames.items()))


def assign_grippers_by_side(frames: Dict[int, dict]) -> None:
    """
    Assign accepted gripper detections to left_gripper/right_gripper by normalized x.
    If only one exists, x<0.5 -> left else right.
    """
    for fr in frames.values():
        gs = sorted(fr["grippers"], key=lambda g: g.x)
        fr["left_gripper"] = None
        fr["right_gripper"] = None

        if len(gs) >= 2:
            fr["left_gripper"] = gs[0]
            fr["right_gripper"] = gs[-1]
        elif len(gs) == 1:
            if gs[0].x < 0.5:
                fr["left_gripper"] = gs[0]
            else:
                fr["right_gripper"] = gs[0]


def is_reliable_toy(obs: Optional[Obs], allow_predicted: bool) -> bool:
    if obs is None:
        return False
    if obs.source in RELIABLE_TOY_SOURCES:
        return True
    return allow_predicted and obs.source == "PREDICTED"


def nearest_past_obs(
    history: deque,
    current_time: float,
    target_age_s: float,
    max_gap_s: float,
) -> Optional[Obs]:
    target_time = current_time - target_age_s
    best = None
    best_err = None

    for obs in history:
        err = abs(obs.time_s - target_time)
        if best_err is None or err < best_err:
            best = obs
            best_err = err

    if best is None:
        return None

    if best_err is not None and best_err > max_gap_s:
        return None

    return best


def window_obs(history: deque, current_time: float, window_s: float) -> List[Obs]:
    cutoff = current_time - window_s
    return [o for o in history if o.time_s >= cutoff]


def mean(vals: Iterable[float]) -> Optional[float]:
    vals = list(vals)
    return sum(vals) / len(vals) if vals else None


def std(vals: Iterable[float]) -> Optional[float]:
    vals = list(vals)
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def coupled_recently(
    toy_hist: deque,
    grip_hist: deque,
    current_time: float,
    window_s: float,
    max_mean_dist: float,
    max_dist_std: float,
) -> Tuple[bool, Optional[float], Optional[float], int]:
    toy_by_frame = {o.frame: o for o in window_obs(toy_hist, current_time, window_s)}
    grip_by_frame = {o.frame: o for o in window_obs(grip_hist, current_time, window_s)}

    ds = []
    for fr in sorted(set(toy_by_frame) & set(grip_by_frame)):
        d = dist(toy_by_frame[fr], grip_by_frame[fr])
        if d is not None:
            ds.append(d)

    if len(ds) < 3:
        return False, None, None, len(ds)

    m = mean(ds)
    s = std(ds)
    coupled = (
        m is not None
        and s is not None
        and m <= max_mean_dist
        and s <= max_dist_std
    )
    return coupled, m, s, len(ds)


def load_gt(gt_path: Optional[Path]) -> Dict[str, List[dict]]:
    if gt_path is None:
        return {}

    out = defaultdict(list)
    with gt_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rollout = row["rollout"]
            out[rollout].append(
                {
                    "start_s": float(row["start_s"]),
                    "end_s": float(row["end_s"]),
                    "label": row["label"].strip().upper(),
                }
            )
    return dict(out)


def gt_label_at(intervals: List[dict], t: float) -> str:
    for it in intervals:
        if it["start_s"] <= t <= it["end_s"]:
            return it["label"]
    return "NORMAL"


def gt_events(intervals: List[dict]) -> List[dict]:
    return [
        it for it in intervals
        if it["label"] != "NORMAL"
    ]


def evaluate_events(
    final_events: List[dict],
    intervals: List[dict],
    tolerance_s: float,
) -> dict:
    gt = gt_events(intervals)
    used = set()
    matches = []

    for gi, event in enumerate(gt):
        best_j = None
        best_abs = None

        for j, pred in enumerate(final_events):
            if j in used:
                continue

            # Accept triggers from GT start until end+tolerance.
            lo = event["start_s"]
            hi = event["end_s"] + tolerance_s

            if lo <= pred["time_s"] <= hi:
                abs_delay = abs(pred["time_s"] - event["start_s"])
                if best_abs is None or abs_delay < best_abs:
                    best_abs = abs_delay
                    best_j = j

        if best_j is not None:
            used.add(best_j)
            pred = final_events[best_j]
            matches.append(
                {
                    "gt_label": event["label"],
                    "gt_start_s": event["start_s"],
                    "pred_type": pred["type"],
                    "pred_time_s": pred["time_s"],
                    "delay_s": pred["time_s"] - event["start_s"],
                }
            )

    tp = len(matches)
    fn = len(gt) - tp
    fp = len(final_events) - tp

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None

    delays = [m["delay_s"] for m in matches]
    return {
        "gt_events": len(gt),
        "pred_events": len(final_events),
        "true_positive_events": tp,
        "false_positive_events": fp,
        "missed_failure_events": fn,
        "event_precision": precision,
        "event_recall": recall,
        "mean_trigger_delay_s": mean(delays),
        "matches": matches,
    }


def process_one(
    path: Path,
    args,
    gt_intervals: List[dict],
) -> dict:
    frames = read_tracker_csv(path)
    assign_grippers_by_side(frames)

    if not frames:
        raise RuntimeError(f"No usable frames in {path}")

    # Per-side histories.
    toy_hist = {
        "left": deque(maxlen=500),
        "right": deque(maxlen=500),
    }
    grip_hist = {
        "left": deque(maxlen=500),
        "right": deque(maxlen=500),
    }

    raw_streak = {
        "GRASP_MISS": 0,
        "OBJECT_DROP": 0,
        "STAGNATION": 0,
    }

    last_pause_time = -1e9
    final_events = []
    rows = []

    total_frames = 0
    reliable_toy_frames = 0
    uncertain_toy_frames = 0
    frames_with_any_gripper = 0

    for frame_id, fr in frames.items():
        t = float(fr["time_s"])
        total_frames += 1

        current = {}

        for side, toy_cls, grip_key in (
            ("left", "left_toy", "left_gripper"),
            ("right", "right_toy", "right_gripper"),
        ):
            toy = fr["toys"].get(toy_cls)
            grip = fr.get(grip_key)

            toy_reliable = is_reliable_toy(
                toy,
                args.allow_predicted_toy,
            )

            if toy_reliable:
                toy_hist[side].append(toy)

            if grip is not None:
                grip_hist[side].append(grip)

            current[side] = {
                "toy": toy,
                "grip": grip,
                "toy_reliable": toy_reliable,
            }

        if any(current[s]["grip"] is not None for s in ("left", "right")):
            frames_with_any_gripper += 1

        reliable_sides = sum(
            current[s]["toy_reliable"]
            for s in ("left", "right")
        )
        if reliable_sides == 2:
            reliable_toy_frames += 1
        else:
            uncertain_toy_frames += 1

        raw_types = []
        evidence = []

        for side in ("left", "right"):
            toy = current[side]["toy"]
            grip = current[side]["grip"]
            toy_ok = current[side]["toy_reliable"]

            # If either current toy or gripper is unavailable, grasp/drop abstain.
            if not toy_ok or grip is None:
                continue

            # Current distance.
            d_now = dist(toy, grip)

            # Motion-window lookback.
            toy_prev = nearest_past_obs(
                toy_hist[side],
                t,
                args.motion_window_s,
                args.max_toy_gap_s,
            )
            grip_prev = nearest_past_obs(
                grip_hist[side],
                t,
                args.motion_window_s,
                args.max_gripper_gap_s,
            )

            toy_disp = displacement(toy_prev, toy)
            grip_disp = displacement(grip_prev, grip)

            # ---- GRASP_MISS ----
            # Vision-only approximation:
            # - gripper is/was near toy,
            # - gripper moves upward,
            # - toy fails to move upward enough.
            if (
                toy_disp is not None
                and grip_disp is not None
                and d_now is not None
            ):
                toy_dx, toy_dy, toy_move = toy_disp
                grip_dx, grip_dy, grip_move = grip_disp

                near_now = d_now <= args.near_distance
                near_prev = (
                    dist(toy_prev, grip_prev) is not None
                    and dist(toy_prev, grip_prev) <= args.near_distance
                )

                grip_lifted = grip_dy <= -args.gripper_lift_dy
                toy_followed_up = toy_dy <= -args.toy_follow_min_dy

                if (
                    (near_now or near_prev)
                    and grip_lifted
                    and not toy_followed_up
                ):
                    raw_types.append("GRASP_MISS")
                    evidence.append(
                        f"{side}: grip_dy={grip_dy:.3f}, "
                        f"toy_dy={toy_dy:.3f}, d={d_now:.3f}"
                    )

            # ---- OBJECT_DROP ----
            coupled, mean_d, std_d, n_pairs = coupled_recently(
                toy_hist[side],
                grip_hist[side],
                t,
                args.following_window_s,
                args.follow_distance,
                args.following_distance_std,
            )

            if coupled:
                # Compare to motion-window past.
                if (
                    toy_disp is not None
                    and d_now is not None
                    and mean_d is not None
                ):
                    toy_dx, toy_dy, toy_move = toy_disp
                    distance_increase = d_now - mean_d

                    if (
                        distance_increase >= args.drop_distance_increase
                        and toy_dy >= args.drop_down_dy
                    ):
                        raw_types.append("OBJECT_DROP")
                        evidence.append(
                            f"{side}: d_now={d_now:.3f}, "
                            f"recent_mean_d={mean_d:.3f}, "
                            f"toy_dy={toy_dy:.3f}"
                        )

        # ---- STAGNATION ----
        # Conservative: only if BOTH toy tracks are reliable, and if gripper
        # observations exist for at least one side.
        stagnation_votes = []

        for side in ("left", "right"):
            toy = current[side]["toy"]
            if not current[side]["toy_reliable"]:
                continue

            toy_prev = nearest_past_obs(
                toy_hist[side],
                t,
                args.stagnation_window_s,
                args.max_toy_gap_s,
            )
            td = displacement(toy_prev, toy)

            grip = current[side]["grip"]
            gd = None
            if grip is not None:
                grip_prev = nearest_past_obs(
                    grip_hist[side],
                    t,
                    args.stagnation_window_s,
                    args.max_gripper_gap_s,
                )
                gd = displacement(grip_prev, grip)

            if td is None:
                continue

            toy_still = td[2] <= args.stagnation_motion

            if gd is not None:
                grip_still = gd[2] <= args.stagnation_motion
                stagnation_votes.append(toy_still and grip_still)
            else:
                # Without gripper evidence, do not infer stagnation from toy alone.
                stagnation_votes.append(False)

        if stagnation_votes and all(stagnation_votes):
            raw_types.append("STAGNATION")
            evidence.append("all observed sides: toy+gripper motion below threshold")

        # Deduplicate and choose priority.
        raw_types = list(dict.fromkeys(raw_types))
        priority = ["OBJECT_DROP", "GRASP_MISS", "STAGNATION"]

        raw_type = "NORMAL"
        for typ in priority:
            if typ in raw_types:
                raw_type = typ
                break

        # Debounce streak.
        for typ in raw_streak:
            if typ == raw_type:
                raw_streak[typ] += 1
            else:
                raw_streak[typ] = 0

        cooldown_ok = (t - last_pause_time) >= args.cooldown_s
        final_pause = (
            raw_type != "NORMAL"
            and raw_streak[raw_type] >= args.confirm_frames
            and cooldown_ok
        )

        if final_pause:
            final_events.append(
                {
                    "frame": frame_id,
                    "time_s": t,
                    "type": raw_type,
                    "evidence": " | ".join(evidence),
                }
            )
            last_pause_time = t

        gt_label = gt_label_at(gt_intervals, t)

        rows.append(
            {
                "frame": frame_id,
                "time_s": f"{t:.3f}",
                "left_toy_source": (
                    current["left"]["toy"].source
                    if current["left"]["toy"] is not None
                    else ""
                ),
                "right_toy_source": (
                    current["right"]["toy"].source
                    if current["right"]["toy"] is not None
                    else ""
                ),
                "left_gripper_visible": current["left"]["grip"] is not None,
                "right_gripper_visible": current["right"]["grip"] is not None,
                "left_toy_reliable": current["left"]["toy_reliable"],
                "right_toy_reliable": current["right"]["toy_reliable"],
                "raw_trigger": raw_type,
                "raw_streak": (
                    raw_streak[raw_type]
                    if raw_type != "NORMAL"
                    else 0
                ),
                "final_pause": final_pause,
                "evidence": " | ".join(evidence),
                "gt_label": gt_label,
            }
        )

    out_csv = args.output_dir / f"{path.stem}_failure_trigger.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Frame-level diagnostic accuracy if GT exists.
    frame_eval = {}
    if gt_intervals:
        tp = fp = tn = fn = 0

        for r in rows:
            gt_failure = r["gt_label"] != "NORMAL"
            pred_failure = bool(r["final_pause"])

            if pred_failure and gt_failure:
                tp += 1
            elif pred_failure and not gt_failure:
                fp += 1
            elif not pred_failure and gt_failure:
                fn += 1
            else:
                tn += 1

        frame_eval = {
            "frame_tp": tp,
            "frame_fp": fp,
            "frame_tn": tn,
            "frame_fn": fn,
        }

    event_eval = (
        evaluate_events(
            final_events,
            gt_intervals,
            args.match_tolerance_s,
        )
        if gt_intervals
        else {}
    )

    summary = {
        "input_csv": str(path),
        "num_frames": total_frames,
        "reliable_both_toys_rate": (
            reliable_toy_frames / total_frames
            if total_frames
            else None
        ),
        "vision_uncertain_rate": (
            uncertain_toy_frames / total_frames
            if total_frames
            else None
        ),
        "any_gripper_visible_rate": (
            frames_with_any_gripper / total_frames
            if total_frames
            else None
        ),
        "num_final_pause_events": len(final_events),
        "final_events": final_events,
        "frame_eval": frame_eval,
        "event_eval": event_eval,
        "parameters": {
            "motion_window_s": args.motion_window_s,
            "following_window_s": args.following_window_s,
            "stagnation_window_s": args.stagnation_window_s,
            "near_distance": args.near_distance,
            "follow_distance": args.follow_distance,
            "gripper_lift_dy": args.gripper_lift_dy,
            "toy_follow_min_dy": args.toy_follow_min_dy,
            "drop_distance_increase": args.drop_distance_increase,
            "drop_down_dy": args.drop_down_dy,
            "stagnation_motion": args.stagnation_motion,
            "confirm_frames": args.confirm_frames,
            "cooldown_s": args.cooldown_s,
        },
    }

    out_json = args.output_dir / f"{path.stem}_summary.json"
    out_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "rollout": path.stem,
        "input_csv": str(path),
        "frame_csv": str(out_csv),
        "summary_json": str(out_json),
        "num_frames": total_frames,
        "reliable_both_toys_rate": summary["reliable_both_toys_rate"],
        "vision_uncertain_rate": summary["vision_uncertain_rate"],
        "any_gripper_visible_rate": summary["any_gripper_visible_rate"],
        "num_final_pause_events": len(final_events),
        "event_precision": event_eval.get("event_precision"),
        "event_recall": event_eval.get("event_recall"),
        "mean_trigger_delay_s": event_eval.get("mean_trigger_delay_s"),
    }


def resolve_inputs(pattern: str) -> List[Path]:
    p = Path(pattern)
    if p.is_file():
        return [p]

    paths = [Path(x) for x in sorted(glob.glob(pattern))]
    return [x for x in paths if x.is_file()]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = resolve_inputs(args.input)
    if not inputs:
        raise FileNotFoundError(
            f"No input CSV matched: {args.input}"
        )

    gt = load_gt(args.gt_csv)

    all_summaries = []

    for path in inputs:
        rollout_key = path.stem
        intervals = gt.get(rollout_key, [])

        # Also support GT rollout name without common suffixes.
        if not intervals:
            simplified = rollout_key
            for suffix in (
                "_yoloworld_visual_tracking",
                "_yoloworld_left_right_temporal",
                "_yoloworld_left_right",
            ):
                simplified = simplified.replace(suffix, "")
            intervals = gt.get(simplified, [])

        print(f"\n=== {path.name} ===")
        result = process_one(
            path,
            args,
            intervals,
        )
        all_summaries.append(result)

        print(
            f"frames={result['num_frames']} "
            f"reliable_toys={result['reliable_both_toys_rate']:.3f} "
            f"vision_uncertain={result['vision_uncertain_rate']:.3f} "
            f"gripper_visible={result['any_gripper_visible_rate']:.3f} "
            f"pause_events={result['num_final_pause_events']}"
        )

        if result["event_precision"] is not None:
            print(
                f"event_precision={result['event_precision']:.3f} "
                f"event_recall={result['event_recall']:.3f} "
                f"mean_delay={result['mean_trigger_delay_s']}"
            )

    batch_csv = args.output_dir / "batch_summary.csv"
    with batch_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(all_summaries[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_summaries)

    batch_json = args.output_dir / "batch_summary.json"
    batch_json.write_text(
        json.dumps(
            all_summaries,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nDONE")
    print(f"Batch summary CSV:  {batch_csv}")
    print(f"Batch summary JSON: {batch_json}")


if __name__ == "__main__":
    main()
