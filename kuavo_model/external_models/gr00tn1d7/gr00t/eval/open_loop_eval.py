# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Any
import warnings
import json

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy import BasePolicy
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyClient
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tyro


warnings.simplefilter("ignore", category=FutureWarning)

"""
Example commands:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

"""


def plot_trajectory_results(
    state_joints_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    traj_id: int,
    state_keys: list[str],
    action_keys: list[str],
    action_horizon: int,
    save_plot_path: str,
) -> None:
    """
    Plot and save trajectory results comparing ground truth and predicted actions.

    Args:
        state_joints_across_time: Array of state joints over time
        gt_action_across_time: Ground truth actions over time
        pred_action_across_time: Predicted actions over time
        traj_id: Trajectory ID
        state_keys: List of state modality keys
        action_keys: List of action modality keys
        action_horizon: Action horizon used for inference
        save_plot_path: Path to save the plot
    """
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]

    indices_to_plot = list(range(action_dim))

    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logging.warning("No valid indices to plot")
        return

    # Always plot and save
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))

    # Handle case where there's only one subplot
    if num_plots == 1:
        axes = [axes]

    # Add a global title showing the modality keys
    fig.suptitle(
        f"Trajectory {traj_id} - State: {', '.join(state_keys)} | Action: {', '.join(action_keys)}",
        fontsize=16,
        color="blue",
    )

    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]

        # The dimensions of state_joints and action are the same
        # only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, action_idx], label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")
        ax.plot(pred_action_across_time[:, action_idx], label="pred action")

        # put a dot every ACTION_HORIZON
        for j in range(0, actual_steps, action_horizon):
            if j == 0:
                ax.plot(
                    j,
                    gt_action_across_time[j, action_idx],
                    "ro",
                    label="inference point",
                )
            else:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro")

        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()

    # Create filename with trajectory ID
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)

    plt.close()  # Close the figure to free memory


def parse_observation_gr00t(
    obs: dict[str, Any], modality_configs: dict[str, Any]
) -> dict[str, Any]:
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = key
            else:
                parsed_key = f"{modality}.{key}"
            arr = obs[parsed_key]
            # Add batch dimension
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]
    return new_obs


def parse_action_gr00t(action: dict[str, Any]) -> dict[str, Any]:
    # Unbatch and add prefix
    return {f"action.{key}": action[key][0] for key in action}


def _blend_weights(length: int, mode: str, start_weight: float) -> np.ndarray:
    progress = np.linspace(0.0, 1.0, length)
    if mode == "linear":
        base = progress
    elif mode == "cosine":
        base = 0.5 - 0.5 * np.cos(np.pi * progress)
    else:
        raise ValueError(f"Unsupported chunk blend mode: {mode}")
    return start_weight + (1.0 - start_weight) * base


def _prepare_executed_chunk(
    action_chunk: dict[str, np.ndarray],
    previous_raw_chunk: dict[str, np.ndarray] | None,
    action_keys: list[str],
    *,
    execution_horizon: int,
    blend_mode: str,
    blend_steps: int,
    start_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened baseline/blended chunks while retaining raw model predictions."""
    baseline_parts = []
    blended_parts = []
    for key in action_keys:
        full_key = f"action.{key}"
        current = np.asarray(action_chunk[full_key])
        if current.ndim == 1:
            current = current[:, None]
        if current.shape[0] < execution_horizon:
            raise ValueError(
                f"Action {key} has horizon {current.shape[0]}, shorter than {execution_horizon}"
            )
        baseline = current[:execution_horizon].copy()
        blended = baseline.copy()
        if blend_mode != "none" and previous_raw_chunk is not None and "arm" in key.lower():
            previous = np.asarray(previous_raw_chunk[full_key])
            if previous.ndim == 1:
                previous = previous[:, None]
            effective_blend = min(
                blend_steps,
                execution_horizon,
                previous.shape[0] - execution_horizon,
            )
            if effective_blend < 1:
                raise ValueError(
                    f"No previous tail available for {key}; model horizon must exceed execution horizon"
                )
            alpha = _blend_weights(effective_blend, blend_mode, start_weight)[:, None]
            old_tail = previous[
                execution_horizon : execution_horizon + effective_blend
            ]
            blended[:effective_blend] = (
                (1.0 - alpha) * old_tail + alpha * baseline[:effective_blend]
            )
        baseline_parts.append(baseline)
        blended_parts.append(blended)
    return np.concatenate(baseline_parts, axis=-1), np.concatenate(blended_parts, axis=-1)


def _apply_gripper_hysteresis(
    chunks: np.ndarray,
    action_keys: list[str],
    action_widths: list[int],
    *,
    open_threshold: float,
    close_threshold: float,
) -> np.ndarray:
    result = chunks.copy()
    offset = 0
    for key, width in zip(action_keys, action_widths):
        if "gripper" in key.lower():
            for dimension in range(offset, offset + width):
                state = 1.0 if result[0, 0, dimension] >= 0.5 else 0.0
                for chunk_index in range(result.shape[0]):
                    for step_index in range(result.shape[1]):
                        command = result[chunk_index, step_index, dimension]
                        if command >= close_threshold:
                            state = 1.0
                        elif command <= open_threshold:
                            state = 0.0
                        result[chunk_index, step_index, dimension] = state
        offset += width
    return result


def _trajectory_metrics(
    chunks: np.ndarray,
    ground_truth: np.ndarray,
    arm_indices: np.ndarray,
) -> dict[str, float]:
    predicted = chunks.reshape(-1, chunks.shape[-1])[: len(ground_truth)]
    arm = chunks[:, :, arm_indices]
    boundary_jump = np.linalg.norm(arm[1:, 0] - arm[:-1, -1], axis=-1)
    previous_velocity = arm[:-1, -1] - arm[:-1, -2]
    next_velocity = arm[1:, 1] - arm[1:, 0]
    cosine = np.sum(previous_velocity * next_velocity, axis=-1) / (
        np.linalg.norm(previous_velocity, axis=-1)
        * np.linalg.norm(next_velocity, axis=-1)
        + 1e-8
    )
    acceleration = np.linalg.norm(np.diff(arm, n=2, axis=1), axis=-1)
    return {
        "mse": float(np.mean((ground_truth - predicted) ** 2)),
        "mae": float(np.mean(np.abs(ground_truth - predicted))),
        "arm_boundary_jump_mean": float(np.mean(boundary_jump)),
        "arm_boundary_jump_p95": float(np.percentile(boundary_jump, 95)),
        "arm_velocity_cosine_p05": float(np.percentile(cosine, 5)),
        "arm_velocity_cosine_median": float(np.median(cosine)),
        "arm_intra_chunk_acceleration_p95_per_step2": float(
            np.percentile(acceleration, 95)
        ),
    }


def evaluate_single_trajectory(
    policy: BasePolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    modality_keys: list[str] | None = None,
    steps=300,
    action_horizon=16,
    save_plot_path=None,
    chunk_blend_mode="none",
    chunk_blend_steps=8,
    new_chunk_start_weight=0.5,
    gripper_hysteresis=False,
    gripper_open_threshold=0.35,
    gripper_close_threshold=0.65,
    save_metrics_path=None,
):
    # Ensure steps doesn't exceed trajectory length
    traj = loader[traj_id]
    traj_length = len(traj)
    actual_steps = min(steps, traj_length)
    logging.info(
        f"Using {actual_steps} steps (requested: {steps}, trajectory length: {traj_length})"
    )

    baseline_chunks = []
    blended_chunks = []
    previous_raw_chunk = None

    # Extract state and action keys separately and sort for consistent order
    state_keys = loader.modality_configs["state"].modality_keys
    action_keys = (
        loader.modality_configs["action"].modality_keys if modality_keys is None else modality_keys
    )

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")
    for step_count in range(0, actual_steps, action_horizon):
        data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag)
        logging.info(f"inferencing at step: {step_count}")
        obs = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = v  # (T, D)
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)  # (T, H, W, C)
        for language_key in loader.modality_configs["language"].modality_keys:
            obs[language_key] = data_point.text
        parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)
        _action_chunk, _ = policy.get_action(parsed_obs)
        action_chunk = parse_action_gr00t(_action_chunk)
        baseline, blended = _prepare_executed_chunk(
            action_chunk,
            previous_raw_chunk,
            action_keys,
            execution_horizon=action_horizon,
            blend_mode=chunk_blend_mode,
            blend_steps=chunk_blend_steps,
            start_weight=new_chunk_start_weight,
        )
        baseline_chunks.append(baseline)
        blended_chunks.append(blended)
        previous_raw_chunk = {key: np.asarray(value).copy() for key, value in action_chunk.items()}

    def extract_state_joints(traj: pd.DataFrame, columns: list[str]):
        np_dict = {}
        for column in columns:
            np_dict[column] = np.vstack([arr for arr in traj[column]])
        return np.concatenate([np_dict[column] for column in columns], axis=-1)

    # plot the joints
    state_joints_across_time = extract_state_joints(traj, [f"state.{key}" for key in state_keys])
    gt_action_across_time = extract_state_joints(traj, [f"action.{key}" for key in action_keys])[
        :actual_steps
    ]
    baseline_chunks = np.stack(baseline_chunks)
    blended_chunks = np.stack(blended_chunks)
    action_widths = []
    for key in action_keys:
        raw_action = np.asarray(action_chunk[f"action.{key}"])
        action_widths.append(1 if raw_action.ndim == 1 else int(raw_action.shape[-1]))
    if gripper_hysteresis:
        baseline_chunks = _apply_gripper_hysteresis(
            baseline_chunks,
            action_keys,
            action_widths,
            open_threshold=gripper_open_threshold,
            close_threshold=gripper_close_threshold,
        )
        blended_chunks = _apply_gripper_hysteresis(
            blended_chunks,
            action_keys,
            action_widths,
            open_threshold=gripper_open_threshold,
            close_threshold=gripper_close_threshold,
        )
    pred_action_across_time = blended_chunks.reshape(-1, blended_chunks.shape[-1])[:actual_steps]
    assert gt_action_across_time.shape == pred_action_across_time.shape, (
        f"gt_action: {gt_action_across_time.shape}, pred_action: {pred_action_across_time.shape}"
    )

    arm_indices = []
    offset = 0
    for key, width in zip(action_keys, action_widths):
        if "arm" in key.lower():
            arm_indices.extend(range(offset, offset + width))
        offset += width
    if not arm_indices:
        arm_indices = list(range(pred_action_across_time.shape[-1]))
    arm_indices_array = np.asarray(arm_indices)
    baseline_metrics = _trajectory_metrics(
        baseline_chunks, gt_action_across_time, arm_indices_array
    )
    blended_metrics = _trajectory_metrics(
        blended_chunks, gt_action_across_time, arm_indices_array
    )
    mse = blended_metrics["mse"]
    mae = blended_metrics["mae"]
    logging.info(f"Unnormalized Action MSE across single traj: {mse}")
    logging.info(f"Unnormalized Action MAE across single traj: {mae}")
    logging.info(f"Baseline metrics: {baseline_metrics}")
    logging.info(f"Blended metrics: {blended_metrics}")
    if save_metrics_path:
        metrics_path = Path(save_metrics_path)
        if len(loader) > 1:
            metrics_path = metrics_path.with_name(
                f"{metrics_path.stem}_traj{traj_id}{metrics_path.suffix or '.json'}"
            )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "trajectory_id": traj_id,
                    "execution_horizon": action_horizon,
                    "chunk_blend_mode": chunk_blend_mode,
                    "chunk_blend_steps": chunk_blend_steps,
                    "new_chunk_start_weight": new_chunk_start_weight,
                    "gripper_hysteresis": gripper_hysteresis,
                    "baseline": baseline_metrics,
                    "blended": blended_metrics,
                },
                stream,
                indent=2,
            )

    logging.info(f"state_joints vs time {state_joints_across_time.shape}")
    logging.info(f"gt_action_joints vs time {gt_action_across_time.shape}")
    logging.info(f"pred_action_joints vs time {pred_action_across_time.shape}")

    # Plot trajectory results
    plot_trajectory_results(
        state_joints_across_time=state_joints_across_time,
        gt_action_across_time=gt_action_across_time,
        pred_action_across_time=pred_action_across_time,
        traj_id=traj_id,
        state_keys=state_keys,
        action_keys=action_keys,
        action_horizon=action_horizon,
        save_plot_path=save_plot_path or f"/tmp/open_loop_eval/traj_{traj_id}.jpeg",
    )

    return mse, mae


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "127.0.0.1"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    steps: int = 200
    """Maximum number of steps to evaluate (will be capped by trajectory length)."""

    traj_ids: list[int] = field(default_factory=lambda: [0])
    """List of trajectory IDs to evaluate."""

    action_horizon: int = 16
    """Actions executed before the next inference (8 for candidate C)."""

    chunk_blend_mode: str = "none"
    """Cross-chunk arm blending: none, linear, or cosine."""

    chunk_blend_steps: int = 8
    """Number of overlap steps used for cross-chunk blending."""

    new_chunk_start_weight: float = 0.5
    """New prediction weight at the first blended step."""

    gripper_hysteresis: bool = False
    """Apply independent gripper hysteresis instead of arm blending."""

    gripper_open_threshold: float = 0.35
    """Gripper hysteresis open threshold."""

    gripper_close_threshold: float = 0.65
    """Gripper hysteresis close threshold."""

    save_metrics_path: str | None = None
    """Optional JSON path for baseline and blended metrics."""

    dataset_path: str = "demo_data/cube_to_bowl_5/"
    """Path to the dataset."""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (name or value, case-insensitive). Run with --help to see known tags."""

    model_path: str | None = None
    """Path to the model checkpoint."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    save_plot_path: str | None = None
    """Path to save the plot to."""

    modality_keys: list[str] | None = None
    """List of modality keys to plot. If None, plot all keys."""


def main(args: ArgsConfig):
    args.embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    if args.chunk_blend_mode not in {"none", "linear", "cosine"}:
        raise ValueError("chunk_blend_mode must be one of: none, linear, cosine")
    if not 0.0 <= args.new_chunk_start_weight <= 1.0:
        raise ValueError("new_chunk_start_weight must be in [0, 1]")

    # Download model checkpoint if it's an S3 path
    local_model_path = args.model_path

    # Extract global_step and checkpoint directory name from checkpoint path
    global_step = None
    if local_model_path:
        # Search for pattern "checkpoint-{number}" anywhere in the path
        match = re.search(r"checkpoint-(\d+)", local_model_path)
        if match:
            try:
                global_step = int(match.group(1))
                logging.info(f"Extracted global_step {global_step} from checkpoint path")
            except ValueError:
                logging.warning(
                    f"Could not parse step number from checkpoint path: {local_model_path}"
                )
        else:
            logging.warning(f"Could not find checkpoint-<step> pattern in path: {local_model_path}")

    if local_model_path is not None:
        import torch

        policy = Gr00tPolicy(
            embodiment_tag=args.embodiment_tag,
            model_path=local_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        policy = PolicyClient(host=args.host, port=args.port)

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    logging.info(f"Current modality config: \n{modality}")

    # Create the dataset
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )

    logging.info(f"Dataset length: {len(dataset)}")
    logging.info(f"Running evaluation on trajectories: {args.traj_ids}")

    all_mse = []
    all_mae = []

    for traj_id in args.traj_ids:
        if traj_id >= len(dataset):
            logging.warning(f"Trajectory ID {traj_id} is out of range. Skipping.")
            continue

        logging.info(f"Running trajectory: {traj_id}")
        mse, mae = evaluate_single_trajectory(
            policy,
            dataset,
            traj_id,
            args.embodiment_tag,
            args.modality_keys,
            steps=args.steps,
            action_horizon=args.action_horizon,
            save_plot_path=args.save_plot_path,
            chunk_blend_mode=args.chunk_blend_mode,
            chunk_blend_steps=args.chunk_blend_steps,
            new_chunk_start_weight=args.new_chunk_start_weight,
            gripper_hysteresis=args.gripper_hysteresis,
            gripper_open_threshold=args.gripper_open_threshold,
            gripper_close_threshold=args.gripper_close_threshold,
            save_metrics_path=args.save_metrics_path,
        )
        logging.info(f"MSE for trajectory {traj_id}: {mse}, MAE: {mae}")
        all_mse.append(mse)
        all_mae.append(mae)

    if all_mse:
        avg_mse = np.mean(np.array(all_mse))
        avg_mae = np.mean(np.array(all_mae))
        logging.info(f"Average MSE across all trajs: {avg_mse}")
        logging.info(f"Average MAE across all trajs: {avg_mae}")
    else:
        logging.info("No valid trajectories were evaluated.")
    logging.info("Done")


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
