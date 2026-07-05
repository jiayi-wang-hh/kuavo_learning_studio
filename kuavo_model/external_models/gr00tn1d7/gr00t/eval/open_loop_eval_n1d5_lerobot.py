#!/usr/bin/env python
"""Open-loop evaluation for LeRobot GR00T N1.5 checkpoints.

This script is intentionally separate from ``open_loop_eval.py``. The latter
loads native GR00T N1.7 Hugging Face checkpoints, while this script loads the
LeRobot ``GrootPolicy`` wrapper and its serialized processor pipelines.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot.modeling_groot import GrootPolicy


LOGGER = logging.getLogger("open_loop_eval_n1d5_lerobot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a LeRobot GR00T N1.5 checkpoint against ground-truth "
            "actions from a LeRobot dataset."
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
        help="Local root of the LeRobot dataset.",
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
        default=Path("/tmp/open_loop_eval_n1d5"),
        help="Directory for plots and metrics.json.",
    )
    parser.add_argument(
        "--video-backend",
        default="pyav",
        help="LeRobot video backend (default: pyav, avoiding torchcodec ABI issues).",
    )
    return parser.parse_args()


def scalar_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    if isinstance(value, np.ndarray):
        return int(value.item())
    return int(value)


def to_numpy_action(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        if "action" not in value:
            raise KeyError(f"Postprocessor returned a dict without 'action': {value.keys()}")
        value = value["action"]
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f"Expected one action vector, got shape {array.shape}")
    return array


def make_policy_input(sample: dict[str, Any], policy: GrootPolicy) -> dict[str, Any]:
    missing = [key for key in policy.config.input_features if key not in sample]
    if missing:
        raise KeyError(
            "Dataset sample is missing checkpoint input features: "
            f"{missing}. Available keys: {sorted(sample)}"
        )

    observation = {key: sample[key] for key in policy.config.input_features}
    # VLA processor pipelines commonly consume the natural-language task under
    # this key. Preserve it when present without leaking ground-truth actions.
    if "task" in sample:
        observation["task"] = sample["task"]
    return observation


def plot_episode(
    episode_index: int,
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
) -> None:
    action_dim = ground_truth.shape[-1]
    figure, axes = plt.subplots(
        action_dim,
        1,
        figsize=(11, max(3, action_dim * 2.2)),
        sharex=True,
        squeeze=False,
    )
    x = np.arange(len(ground_truth))
    for dim in range(action_dim):
        axis = axes[dim, 0]
        axis.plot(x, ground_truth[:, dim], label="ground truth", linewidth=1.2)
        axis.plot(x, predictions[:, dim], label="prediction", linewidth=1.0)
        axis.set_ylabel(f"a[{dim}]")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(loc="best")
    axes[-1, 0].set_xlabel("episode step")
    figure.suptitle(f"GR00T N1.5 open-loop evaluation — episode {episode_index}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {args.model_path}")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.dataset_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    LOGGER.info("Loading LeRobot GR00T N1.5 checkpoint: %s", args.model_path)
    policy = GrootPolicy.from_pretrained(args.model_path)
    policy = policy.to(device).eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.model_path,
        dataset_stats=None,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    dataset_kwargs: dict[str, Any] = {
        "repo_id": args.dataset_repo_id,
        "root": args.dataset_root,
        "episodes": args.episodes,
    }
    dataset_kwargs["video_backend"] = args.video_backend
    dataset = LeRobotDataset(**dataset_kwargs)
    LOGGER.info("Loaded %d frames from episodes %s", len(dataset), args.episodes)

    episode_ground_truth: dict[int, list[np.ndarray]] = defaultdict(list)
    episode_predictions: dict[int, list[np.ndarray]] = defaultdict(list)
    episode_steps: dict[int, int] = defaultdict(int)
    current_episode: int | None = None
    predicted_chunk: list[np.ndarray] = []

    with torch.inference_mode():
        for sample in dataset:
            episode_index = scalar_int(sample["episode_index"])
            if episode_index not in args.episodes:
                continue
            if args.max_steps > 0 and episode_steps[episode_index] >= args.max_steps:
                continue

            if current_episode != episode_index:
                current_episode = episode_index
                predicted_chunk = []
                policy.reset()

            if not predicted_chunk:
                policy_input = make_policy_input(sample, policy)
                processed = preprocessor(policy_input)
                raw_chunk = policy.predict_action_chunk(processed)
                horizon = min(args.action_horizon, raw_chunk.shape[1])
                predicted_chunk = [
                    to_numpy_action(postprocessor(raw_chunk[:, offset]))
                    for offset in range(horizon)
                ]

            prediction = predicted_chunk.pop(0)
            ground_truth = to_numpy_action(sample["action"])
            if prediction.shape != ground_truth.shape:
                raise ValueError(
                    "Prediction/target shape mismatch: "
                    f"{prediction.shape} vs {ground_truth.shape}"
                )

            episode_predictions[episode_index].append(prediction)
            episode_ground_truth[episode_index].append(ground_truth)
            episode_steps[episode_index] += 1

    results: dict[str, Any] = {
        "model_path": str(args.model_path.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "action_horizon": args.action_horizon,
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
    LOGGER.info("Overall metrics: %s", results["overall"])
    LOGGER.info("Saved metrics to %s", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
