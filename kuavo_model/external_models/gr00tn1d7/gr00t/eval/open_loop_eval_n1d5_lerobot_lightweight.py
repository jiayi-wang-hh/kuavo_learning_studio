#!/usr/bin/env python
"""Lightweight open-loop evaluation for LeRobot GR00T N1.5 checkpoints.

Ground-truth actions and episode metadata are read directly from parquet-backed
rows. Video is decoded only at action-chunk replanning steps, matching the
lightweight access pattern used by the native GR00T open-loop evaluator.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot.modeling_groot import GrootPolicy

from open_loop_eval_n1d5_lerobot import (
    make_policy_input,
    plot_episode,
    scalar_int,
    to_numpy_action,
)


LOGGER = logging.getLogger("open_loop_eval_n1d5_lerobot_lightweight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lightweight open-loop evaluation for a LeRobot GR00T N1.5 "
            "checkpoint. Videos are decoded only when replanning."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to checkpoints/<step>/pretrained_model.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Local root of the LeRobot v3 dataset.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default="local/open_loop_eval",
        help="Dataset repo_id used by LeRobot metadata.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[0],
        help="Episode indices to evaluate.",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=16,
        help="Number of predicted actions used before replanning.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="Maximum evaluated steps per episode; <=0 evaluates all steps.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device. With CUDA_VISIBLE_DEVICES set, normally use cuda.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/open_loop_eval_n1d5_lightweight"),
        help="Directory for plots and metrics.json.",
    )
    parser.add_argument(
        "--video-backend",
        default="pyav",
        choices=("pyav", "video_reader", "torchcodec"),
        help="Video backend used only at replanning steps (default: pyav).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {args.model_path}")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.dataset_root}")


def load_policy_and_processors(
    model_path: Path,
    device: torch.device,
) -> tuple[GrootPolicy, Any, Any]:
    LOGGER.info("Loading LeRobot GR00T N1.5 checkpoint: %s", model_path)
    policy = GrootPolicy.from_pretrained(model_path)
    policy = policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=model_path,
        dataset_stats=None,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, postprocessor


def evaluate(
    args: argparse.Namespace,
    dataset: LeRobotDataset,
    policy: GrootPolicy,
    preprocessor: Any,
    postprocessor: Any,
) -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]], int]:
    episode_ground_truth: dict[int, list[np.ndarray]] = defaultdict(list)
    episode_predictions: dict[int, list[np.ndarray]] = defaultdict(list)
    episode_steps: dict[int, int] = defaultdict(int)
    current_episode: int | None = None
    predicted_chunk: list[np.ndarray] = []
    video_decode_count = 0

    with torch.inference_mode():
        for index in range(len(dataset)):
            # Raw rows come from parquet and deliberately skip all video decoding.
            raw_sample = dataset.get_raw_item(index)
            episode_index = scalar_int(raw_sample["episode_index"])
            if episode_index not in args.episodes:
                continue
            if args.max_steps > 0 and episode_steps[episode_index] >= args.max_steps:
                continue

            if current_episode != episode_index:
                current_episode = episode_index
                predicted_chunk = []
                policy.reset()

            if not predicted_chunk:
                # Full access decodes camera frames, but only once per horizon.
                inference_sample = dataset[index]
                video_decode_count += 1
                policy_input = make_policy_input(inference_sample, policy)
                processed = preprocessor(policy_input)
                raw_chunk = policy.predict_action_chunk(processed)
                horizon = min(args.action_horizon, raw_chunk.shape[1])
                predicted_chunk = [
                    to_numpy_action(postprocessor(raw_chunk[:, offset]))
                    for offset in range(horizon)
                ]
                # Do not retain decoded frames or GPU processor outputs while
                # consuming the CPU action chunk over the following steps.
                del inference_sample, policy_input, processed, raw_chunk

            prediction = predicted_chunk.pop(0)
            ground_truth = to_numpy_action(raw_sample["action"])
            if prediction.shape != ground_truth.shape:
                raise ValueError(
                    "Prediction/target shape mismatch: "
                    f"{prediction.shape} vs {ground_truth.shape}"
                )

            episode_predictions[episode_index].append(prediction)
            episode_ground_truth[episode_index].append(ground_truth)
            episode_steps[episode_index] += 1

    return episode_ground_truth, episode_predictions, video_decode_count


def save_results(
    args: argparse.Namespace,
    episode_ground_truth: dict[int, list[np.ndarray]],
    episode_predictions: dict[int, list[np.ndarray]],
    video_decode_count: int,
) -> None:
    results: dict[str, Any] = {
        "model_path": str(args.model_path.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "action_horizon": args.action_horizon,
        "video_decode_count": video_decode_count,
        "episodes": {},
    }
    all_errors: list[np.ndarray] = []

    for episode_index in args.episodes:
        if not episode_ground_truth[episode_index]:
            LOGGER.warning("No frames evaluated for episode %d", episode_index)
            continue
        ground_truth = np.stack(episode_ground_truth[episode_index])
        predictions = np.stack(episode_predictions[episode_index])
        errors = predictions - ground_truth
        all_errors.append(errors)
        mse = float(np.mean(np.square(errors)))
        mae = float(np.mean(np.abs(errors)))
        results["episodes"][str(episode_index)] = {
            "steps": len(ground_truth),
            "mse": mse,
            "mae": mae,
        }
        plot_path = args.output_dir / f"episode_{episode_index:06d}.png"
        plot_episode(episode_index, ground_truth, predictions, plot_path)
        LOGGER.info(
            "Episode %d: steps=%d MSE=%.8f MAE=%.8f plot=%s",
            episode_index,
            len(ground_truth),
            mse,
            mae,
            plot_path,
        )

    if not all_errors:
        raise RuntimeError("No samples were evaluated; check --episodes and dataset metadata")

    combined_errors = np.concatenate(all_errors, axis=0)
    results["overall"] = {
        "steps": int(combined_errors.shape[0]),
        "mse": float(np.mean(np.square(combined_errors))),
        "mae": float(np.mean(np.abs(combined_errors))),
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info(
        "Overall metrics: %s; full video decodes: %d",
        results["overall"],
        video_decode_count,
    )
    LOGGER.info("Saved metrics to %s", metrics_path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    policy, preprocessor, postprocessor = load_policy_and_processors(args.model_path, device)
    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        episodes=args.episodes,
        video_backend=args.video_backend,
    )
    LOGGER.info("Loaded %d parquet rows from episodes %s", len(dataset), args.episodes)

    ground_truth, predictions, video_decode_count = evaluate(
        args,
        dataset,
        policy,
        preprocessor,
        postprocessor,
    )
    save_results(args, ground_truth, predictions, video_decode_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
