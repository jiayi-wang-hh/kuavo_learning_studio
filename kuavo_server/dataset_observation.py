from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


STATE_KEYS = ("left_arm", "left_gripper", "right_arm", "right_gripper")
STATE_DIMS = {"left_arm": 7, "left_gripper": 1, "right_arm": 7, "right_gripper": 1}
VIDEO_KEYS = ("head", "wrist_left", "wrist_right")
VIDEO_OUTPUT_KEYS = {
    "head": "observation.images.head_cam_h",
    "wrist_left": "observation.images.wrist_cam_l",
    "wrist_right": "observation.images.wrist_cam_r",
}
LANGUAGE_KEY = "annotation.human.task_description"


def _default_gr00t_repo_root() -> Path:
    return Path(__file__).resolve().parents[1] / "kuavo_model" / "external_models" / "gr00tn1d7"


def _validate_modality_metadata(metadata: dict[str, Any]) -> None:
    state_meta = metadata.get("state", {})
    video_meta = metadata.get("video", {})
    missing_state = sorted(set(STATE_KEYS).difference(state_meta))
    missing_video = sorted(set(VIDEO_KEYS).difference(video_meta))
    errors: list[str] = []
    if missing_state:
        errors.append(f"missing state groups: {missing_state}")
    if missing_video:
        errors.append(f"missing video groups: {missing_video}")
    for key in STATE_KEYS:
        if key not in state_meta:
            continue
        actual_dim = int(state_meta[key]["end"]) - int(state_meta[key]["start"])
        if actual_dim != STATE_DIMS[key]:
            errors.append(f"state.{key} expected dim {STATE_DIMS[key]}, got {actual_dim}")
    if errors:
        raise ValueError("Dataset is not a dual-arm Kuavo N1.7 dataset: " + "; ".join(errors))


def _row_to_kuavo_observation(row: Any, prompt: str) -> dict[str, Any]:
    state_parts = []
    for key in STATE_KEYS:
        value = np.asarray(row[f"state.{key}"], dtype=np.float32).reshape(-1)
        expected_dim = STATE_DIMS[key]
        if value.shape != (expected_dim,):
            raise ValueError(
                f"Dataset frame state.{key} expected shape ({expected_dim},), got {value.shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"Dataset frame state.{key} contains NaN or Inf")
        state_parts.append(value)

    observation: dict[str, Any] = {
        "observation.state": np.concatenate(state_parts).astype(np.float32),
        "prompt": prompt,
    }
    for key, output_key in VIDEO_OUTPUT_KEYS.items():
        image = np.asarray(row[f"video.{key}"])
        if image.ndim not in (3, 4):
            raise ValueError(f"Dataset frame video.{key} expected an image, got shape {image.shape}")
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError(
                    f"Dataset frame video.{key} expected one frame, got shape {image.shape}"
                )
            image = image[0]
        observation[output_key] = image
    return observation


def load_dataset_observation(
    dataset_path: Path,
    *,
    episode: int,
    frame: int,
    prompt_override: str | None,
    model_repo_root: Path | None = None,
    video_backend: str = "torchcodec",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one dual-arm Kuavo LeRobot frame using the N1.7 dataset loader."""
    dataset_path = dataset_path.expanduser().resolve()
    repo_root = (model_repo_root or _default_gr00t_repo_root()).expanduser().resolve()
    if episode < 0 or frame < 0:
        raise ValueError("--episode and --frame must be non-negative")
    modality_path = dataset_path / "meta" / "modality.json"
    if not modality_path.is_file():
        raise FileNotFoundError(f"Dataset modality file does not exist: {modality_path}")
    metadata = json.loads(modality_path.read_text(encoding="utf-8"))
    _validate_modality_metadata(metadata)
    if not repo_root.is_dir():
        raise FileNotFoundError(f"N1.7 repository does not exist: {repo_root}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.types import ModalityConfig

    modality_configs = {
        "video": ModalityConfig(delta_indices=[0], modality_keys=list(VIDEO_KEYS)),
        "state": ModalityConfig(delta_indices=[0], modality_keys=list(STATE_KEYS)),
    }
    annotation_meta = metadata.get("annotation", {})
    if "human.task_description" in annotation_meta:
        modality_configs["language"] = ModalityConfig(
            delta_indices=[0], modality_keys=[LANGUAGE_KEY]
        )

    loader = LeRobotEpisodeLoader(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        video_backend=video_backend,
        video_backend_kwargs=None,
    )
    if episode >= len(loader):
        raise IndexError(f"Episode {episode} out of range; dataset has {len(loader)} episodes")
    episode_data = loader[episode]
    if frame >= len(episode_data):
        raise IndexError(
            f"Frame {frame} out of range; episode {episode} has {len(episode_data)} frames"
        )
    row = episode_data.iloc[frame]
    if prompt_override is not None:
        prompt = prompt_override
    elif "language" in modality_configs:
        prompt = str(row[f"language.{LANGUAGE_KEY}"])
    else:
        tasks = loader.episodes_metadata[episode].get("tasks", [])
        prompt = str(tasks[0]) if tasks else ""

    observation = _row_to_kuavo_observation(row, prompt)
    source = {
        "type": "lerobot_dataset",
        "dataset_path": str(dataset_path),
        "episode": episode,
        "frame": frame,
        "video_backend": video_backend,
        "model_repo_root": str(repo_root),
    }
    return observation, source
