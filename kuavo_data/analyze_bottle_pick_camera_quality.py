#!/usr/bin/env python3
"""Batch-audit compressed camera streams in the bottle-pick ROS bags."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rosbag


DEFAULT_BAG_DIR = Path("/media/data/training_data/task1_bottle_pick")
DEFAULT_DATASET_ROOT = Path(
    "/media/data/training_data/task1_bottle_pick_lerobot_v3.0/lerobot"
)

CAMERAS = {
    "head": "/cam_h/color/image_raw/compressed",
    "left": "/cam_l/color/image_raw/compressed",
    "right": "/cam_r/color/image_raw/compressed",
}

METRICS = (
    "decoded_count",
    "decode_failures",
    "effective_fps",
    "median_fps",
    "max_gap",
    "gaps_gt_0.1",
    "gaps_gt_0.2",
    "gaps_gt_0.5",
    "gaps_gt_1.0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-dir", type=Path, default=DEFAULT_BAG_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def empty_camera_accumulator() -> dict[str, Any]:
    return {
        "decoded_timestamps": [],
        "decode_failures": 0,
    }


def summarize_camera(accumulator: dict[str, Any]) -> dict[str, int | float]:
    timestamps = np.asarray(accumulator["decoded_timestamps"], dtype=np.float64)
    decoded_count = int(timestamps.size)
    decode_failures = int(accumulator["decode_failures"])

    if decoded_count < 2:
        return {
            "decoded_count": decoded_count,
            "decode_failures": decode_failures,
            "effective_fps": float("nan"),
            "median_fps": float("nan"),
            "max_gap": float("nan"),
            "gaps_gt_0.1": 0,
            "gaps_gt_0.2": 0,
            "gaps_gt_0.5": 0,
            "gaps_gt_1.0": 0,
        }

    intervals = np.diff(timestamps)
    positive_intervals = intervals[intervals > 0]
    duration = float(timestamps[-1] - timestamps[0])

    effective_fps = (
        float((decoded_count - 1) / duration) if duration > 0 else float("nan")
    )
    median_interval = (
        float(np.median(positive_intervals))
        if positive_intervals.size
        else float("nan")
    )
    median_fps = (
        float(1.0 / median_interval)
        if np.isfinite(median_interval) and median_interval > 0
        else float("nan")
    )

    return {
        "decoded_count": decoded_count,
        "decode_failures": decode_failures,
        "effective_fps": effective_fps,
        "median_fps": median_fps,
        "max_gap": float(np.max(intervals)),
        "gaps_gt_0.1": int(np.count_nonzero(intervals > 0.1)),
        "gaps_gt_0.2": int(np.count_nonzero(intervals > 0.2)),
        "gaps_gt_0.5": int(np.count_nonzero(intervals > 0.5)),
        "gaps_gt_1.0": int(np.count_nonzero(intervals > 1.0)),
    }


def scan_bag(bag_path: Path) -> dict[str, Any]:
    topic_to_camera = {topic: camera for camera, topic in CAMERAS.items()}
    accumulators = {
        camera: empty_camera_accumulator() for camera in CAMERAS
    }

    row: dict[str, Any] = {
        "bag_filename": bag_path.name,
        "bag_path": str(bag_path.resolve()),
        "scan_error": "",
    }

    try:
        with rosbag.Bag(str(bag_path), "r") as bag:
            for topic, message, record_time in bag.read_messages(
                topics=list(topic_to_camera)
            ):
                camera = topic_to_camera[topic]
                accumulator = accumulators[camera]
                encoded = np.frombuffer(message.data, dtype=np.uint8)
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if decoded is None:
                    accumulator["decode_failures"] += 1
                else:
                    accumulator["decoded_timestamps"].append(record_time.to_sec())
    except Exception as exc:  # Preserve a row for unreadable bags.
        row["scan_error"] = f"{type(exc).__name__}: {exc}"

    for camera, accumulator in accumulators.items():
        summary = summarize_camera(accumulator)
        for metric, value in summary.items():
            row[f"{camera}_{metric}"] = value

    return row


def write_quality_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = ["bag_filename", "bag_path", "scan_error"]
    fieldnames.extend(
        f"{camera}_{metric}" for camera in CAMERAS for metric in METRICS
    )

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_episode_map_if_confirmed(
    bag_files: list[Path], dataset_root: Path, output_path: Path
) -> bool:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return False

    with info_path.open(encoding="utf-8") as info_file:
        dataset_info = json.load(info_file)

    total_episodes = dataset_info.get("total_episodes")
    if total_episodes != len(bag_files):
        return False

    # CvtRosbag2Lerobot uses glob('*.bag') followed by list.sort(), then saves
    # one episode for each successfully processed bag. Equal source/episode
    # counts therefore establish the direct lexicographic mapping for this run.
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=("episode_index", "bag_filename", "source_bag_path"),
        )
        writer.writeheader()
        for episode_index, bag_path in enumerate(bag_files):
            writer.writerow(
                {
                    "episode_index": episode_index,
                    "bag_filename": bag_path.name,
                    "source_bag_path": str(bag_path.resolve()),
                }
            )
    return True


def main() -> None:
    args = parse_args()
    bag_dir = args.bag_dir.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else bag_dir
    )

    if not bag_dir.is_dir():
        raise FileNotFoundError(f"Bag directory does not exist: {bag_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    bag_files = sorted(bag_dir.glob("*.bag"))
    if not bag_files:
        raise FileNotFoundError(f"No .bag files found in: {bag_dir}")

    quality_path = output_dir / "bottle_pick_camera_quality.csv"
    episode_map_path = output_dir / "bottle_pick_episode_bag_map.csv"

    rows = [scan_bag(bag_path) for bag_path in bag_files]
    write_quality_csv(rows, quality_path)
    mapping_created = write_episode_map_if_confirmed(
        bag_files, dataset_root, episode_map_path
    )

    summary = {
        "bag_count": len(bag_files),
        "quality_csv": str(quality_path),
        "episode_map_created": mapping_created,
        "episode_map_csv": str(episode_map_path) if mapping_created else None,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
