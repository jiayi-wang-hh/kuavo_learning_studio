#!/usr/bin/env python3
"""Stage-1 failure trigger using visual toy tracks and robot EE state.

V5 deliberately does not infer a gripper from the image.  It pauses only when
an end-effector is demonstrably moving while a reliably observed toy is not:

    ee_displacement / window >= ee_motion_min
    toy_displacement           <= toy_motion_max
    condition duration          >= trigger_duration_s

The two input streams are synchronized by timestamp, so the state CSV may be
recorded at a different rate.  State rows farther than ``--sync-max-gap-s``
from a video frame are considered unavailable and reset/freeze the candidate.

Robot-state CSV contract
------------------------
The CSV must contain a time column (default ``time_s``) and either a shared
EE position (``--ee-columns x,y,z``), or per-side positions
(``--left-ee-columns left_ee_x,left_ee_y,left_ee_z`` and likewise right).
Column names are configurable to make this usable with simulator logs.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import glob
import json
import math
import re
import statistics
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


RELIABLE_SOURCES = {"DETECTED", "TRACKED"}
SIDES = (("left", "left_toy"), ("right", "right_toy"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Visual tracking CSV or glob")
    p.add_argument("--robot-state", required=True, help="Robot-state CSV or glob")
    p.add_argument("--gt-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/progress_trigger_v5_robot_state"))
    p.add_argument("--time-column", default="time_s", help="Time column used in robot-state CSV")
    p.add_argument("--ee-columns", default=None, help="Shared EE XYZ columns, comma separated")
    p.add_argument("--left-ee-columns", default=None, help="Left EE XYZ columns, comma separated")
    p.add_argument("--right-ee-columns", default=None, help="Right EE XYZ columns, comma separated")
    p.add_argument("--sync-max-gap-s", type=float, default=0.10)
    p.add_argument("--motion-window-s", type=float, default=0.50,
                   help="Lookback window for EE and toy displacement")
    p.add_argument("--ee-motion-min", type=float, default=0.020,
                   help="Minimum EE speed (state-unit/s) considered active")
    p.add_argument("--toy-motion-max", type=float, default=0.012,
                   help="Maximum normalized image toy displacement over the window")
    p.add_argument("--trigger-duration-s", type=float, default=1.0)
    p.add_argument("--confirm-frames", type=int, default=2)
    p.add_argument("--cooldown-s", type=float, default=999.0)
    p.add_argument("--allow-predicted", action="store_true")
    p.add_argument("--early-tolerance-s", type=float, default=0.5)
    p.add_argument("--late-tolerance-s", type=float, default=3.0)
    return p.parse_args()


def split_columns(value: Optional[str]) -> Optional[Tuple[str, str, str]]:
    if value is None:
        return None
    cols = tuple(x.strip() for x in value.split(",") if x.strip())
    if len(cols) != 3:
        raise ValueError("EE column arguments must contain exactly three comma-separated names")
    return cols  # type: ignore[return-value]


def key_from_path(path: Path) -> Optional[str]:
    match = re.search(r"(rollout\d+)", path.stem, re.IGNORECASE)
    return match.group(1).lower() if match else None


def resolve(pattern: str) -> List[Path]:
    direct = Path(pattern)
    return [direct] if direct.is_file() else [Path(x) for x in sorted(glob.glob(pattern)) if Path(x).is_file()]


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def load_gt(path: Path) -> Dict[str, dict]:
    return {str(k).lower(): v for k, v in json.loads(path.read_text()).get("rollouts", {}).items()}


def load_visual(path: Path) -> Dict[int, dict]:
    frames: Dict[int, dict] = defaultdict(lambda: {"time_s": None, "toys": {}})
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("accepted", "")).lower() not in {"1", "true", "yes"}:
                continue
            if row.get("class") not in {"left_toy", "right_toy"}:
                continue
            try:
                frame, time_s = int(row["frame"]), float(row["time_s"])
                xy = (float(row["side_center_x"]), float(row["side_center_y"]))
            except (KeyError, TypeError, ValueError):
                continue
            frames[frame]["time_s"] = time_s
            frames[frame]["toys"][row["class"]] = {"xy": xy, "source": row.get("track_source", "")}
    if not frames:
        raise RuntimeError(f"No accepted toy observations in {path}")
    return dict(sorted(frames.items()))


def load_robot_state(path: Path, time_column: str, columns: Tuple[str, str, str]) -> List[Tuple[float, Tuple[float, float, float]]]:
    samples = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set((time_column, *columns)) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing robot-state columns: {sorted(missing)}")
        for row in reader:
            try:
                samples.append((float(row[time_column]), tuple(float(row[c]) for c in columns)))
            except (TypeError, ValueError):
                continue
    if len(samples) < 2:
        raise RuntimeError(f"{path}: need at least two valid EE samples")
    return sorted(samples)


def nearest_sample(samples: List[tuple], t: float, max_gap: float) -> Optional[tuple]:
    times = [s[0] for s in samples]
    i = bisect.bisect_left(times, t)
    candidates = samples[max(0, i - 1): i + 1]
    if not candidates:
        return None
    result = min(candidates, key=lambda s: abs(s[0] - t))
    return result if abs(result[0] - t) <= max_gap else None


def history_at_or_before(history: deque, target: float) -> Optional[dict]:
    candidates = [item for item in history if item["time_s"] <= target]
    return candidates[-1] if candidates else None


def process_rollout(visual_path: Path, state_path: Path, gt: dict, args: argparse.Namespace, shared_cols, left_cols, right_cols) -> dict:
    frames = load_visual(visual_path)
    # Per-side mappings override the shared mapping; a shared EE can be used for
    # single-arm simulations or when the task does not distinguish arms.
    shared = load_robot_state(state_path, args.time_column, shared_cols) if shared_cols else None
    streams = {
        "left": load_robot_state(state_path, args.time_column, left_cols) if left_cols else shared,
        "right": load_robot_state(state_path, args.time_column, right_cols) if right_cols else shared,
    }
    if streams["left"] is None and streams["right"] is None:
        raise ValueError("Provide --ee-columns or at least one per-side EE column mapping")

    state = {side: {"toy_history": deque(maxlen=2000), "ee_history": deque(maxlen=2000), "candidate_s": 0.0} for side, _ in SIDES}
    rows, first_pause_t, first_pause_side = [], None, None
    trigger_streak, last_pause_t, previous_t = 0, -float("inf"), None

    for frame, visual in frames.items():
        t = float(visual["time_s"])
        dt = 0.0 if previous_t is None else max(0.0, t - previous_t)
        previous_t = t
        side_out, raw_sides = {}, []
        for side, toy_class in SIDES:
            st, toy = state[side], visual["toys"].get(toy_class)
            reliable = toy is not None and (toy["source"] in RELIABLE_SOURCES or (args.allow_predicted and toy["source"] == "PREDICTED"))
            ee = nearest_sample(streams[side], t, args.sync_max_gap_s) if streams[side] else None
            out = {"reliable": reliable, "ee_available": ee is not None, "toy_motion": None, "ee_speed": None, "candidate_s": st["candidate_s"], "raw": False, "reason": ""}
            if not reliable:
                st["candidate_s"] = 0.0
                out["reason"] = "vision_uncertain"
            elif ee is None:
                st["candidate_s"] = 0.0
                out["reason"] = "robot_state_unsynchronized"
            else:
                st["toy_history"].append({"time_s": t, "xy": toy["xy"]})
                st["ee_history"].append({"time_s": t, "xyz": ee[1]})
                past_toy = history_at_or_before(st["toy_history"], t - args.motion_window_s)
                past_ee = history_at_or_before(st["ee_history"], t - args.motion_window_s)
                if past_toy is None or past_ee is None:
                    st["candidate_s"] = 0.0
                    out["reason"] = "warming_window"
                else:
                    toy_motion = distance(toy["xy"], past_toy["xy"])
                    ee_dt = t - past_ee["time_s"]
                    ee_speed = distance(ee[1], past_ee["xyz"]) / ee_dt if ee_dt > 0 else 0.0
                    inconsistent = ee_speed >= args.ee_motion_min and toy_motion <= args.toy_motion_max
                    st["candidate_s"] = st["candidate_s"] + dt if inconsistent else 0.0
                    out.update(toy_motion=toy_motion, ee_speed=ee_speed, candidate_s=st["candidate_s"], raw=inconsistent and st["candidate_s"] >= args.trigger_duration_s)
                    out["reason"] = "ee_moves_toy_static" if inconsistent else "signals_consistent"
            out["candidate_s"] = st["candidate_s"]
            side_out[side] = out
            if out["raw"]:
                raw_sides.append(side)

        raw = bool(raw_sides)
        trigger_streak = trigger_streak + 1 if raw else 0
        final_pause = raw and trigger_streak >= args.confirm_frames and t - last_pause_t >= args.cooldown_s
        if final_pause:
            last_pause_t = t
            if first_pause_t is None:
                first_pause_t, first_pause_side = t, "+".join(raw_sides)
        rows.append({"frame": frame, "time_s": f"{t:.6f}", "left_reliable": side_out["left"]["reliable"], "right_reliable": side_out["right"]["reliable"], "left_ee_available": side_out["left"]["ee_available"], "right_ee_available": side_out["right"]["ee_available"], "left_toy_motion": "" if side_out["left"]["toy_motion"] is None else f"{side_out['left']['toy_motion']:.6f}", "right_toy_motion": "" if side_out["right"]["toy_motion"] is None else f"{side_out['right']['toy_motion']:.6f}", "left_ee_speed": "" if side_out["left"]["ee_speed"] is None else f"{side_out['left']['ee_speed']:.6f}", "right_ee_speed": "" if side_out["right"]["ee_speed"] is None else f"{side_out['right']['ee_speed']:.6f}", "left_candidate_s": f"{side_out['left']['candidate_s']:.6f}", "right_candidate_s": f"{side_out['right']['candidate_s']:.6f}", "raw_trigger": raw, "trigger_streak": trigger_streak, "final_pause": final_pause, "trigger_sides": "+".join(raw_sides), "left_reason": side_out["left"]["reason"], "right_reason": side_out["right"]["reason"]})

    outcome, first_gt = str(gt.get("outcome", "")).upper(), gt.get("first_failure_time_s")
    failure = outcome == "FAILURE" or str(gt.get("failure_detected", "")).upper() == "YES"
    delay = float(first_pause_t) - float(first_gt) if failure and first_pause_t is not None and first_gt is not None else None
    in_window = delay is not None and -args.early_tolerance_s <= delay <= args.late_tolerance_s
    event = "FP" if not failure and first_pause_t is not None else "TN" if not failure else "TP" if in_window else "FN"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / f"{visual_path.stem}_robot_state_trigger.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = {"rollout": key_from_path(visual_path), "input_csv": str(visual_path), "robot_state_csv": str(state_path), "gt_outcome": outcome, "gt_first_failure_time_s": first_gt, "pred_pause": first_pause_t is not None, "pred_first_pause_time_s": first_pause_t, "pred_first_pause_side": first_pause_side, "trigger_delay_s": delay, "detected_in_window": in_window, "event_result": event, "params": {k: getattr(args, k) for k in ("motion_window_s", "ee_motion_min", "toy_motion_max", "trigger_duration_s", "sync_max_gap_s", "confirm_frames")}}
    (args.output_dir / f"{visual_path.stem}_robot_state_trigger_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    shared, left, right = split_columns(args.ee_columns), split_columns(args.left_ee_columns), split_columns(args.right_ee_columns)
    visuals, states = resolve(args.input), resolve(args.robot_state)
    if not visuals or not states:
        raise FileNotFoundError("No visual tracking or robot-state CSV matched")
    state_by_key = {key_from_path(p): p for p in states if key_from_path(p)}
    if len(states) == 1:
        state_by_key.setdefault(None, states[0])
    gt_all, summaries = load_gt(args.gt_json), []
    for visual in visuals:
        key, state_path = key_from_path(visual), state_by_key.get(key, state_by_key.get(None))
        if state_path is None:
            print(f"[SKIP] no robot-state CSV for {visual.name}"); continue
        summaries.append(process_rollout(visual, state_path, gt_all.get(key or "", {}), args, shared, left, right))
    if not summaries:
        raise RuntimeError("No rollout had both input streams")
    with (args.output_dir / "batch_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"num_rollouts": len(summaries), "results": summaries}, f, indent=2)
    print(json.dumps({"num_rollouts": len(summaries), "events": {x: sum(s["event_result"] == x for s in summaries) for x in ("TP", "FP", "TN", "FN")}}, indent=2))


if __name__ == "__main__":
    main()
