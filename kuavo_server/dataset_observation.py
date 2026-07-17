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


def _normalize_minmax(values: np.ndarray, min_values: np.ndarray, max_values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    min_values = np.asarray(min_values, dtype=np.float32)
    max_values = np.asarray(max_values, dtype=np.float32)
    normalized = np.zeros_like(values, dtype=np.float32)
    mask = ~np.isclose(max_values, min_values)
    normalized[mask] = (values[mask] - min_values[mask]) / (max_values[mask] - min_values[mask])
    normalized[mask] = 2.0 * normalized[mask] - 1.0
    return np.clip(normalized, -1.0, 1.0)


def _state_normalized_values(
    row: Any,
    state_stats: dict[str, Any],
    *,
    use_percentiles: bool = True,
) -> dict[str, np.ndarray]:
    min_key = "q01" if use_percentiles else "min"
    max_key = "q99" if use_percentiles else "max"
    normalized = {}
    for key in STATE_KEYS:
        values = np.asarray(row[f"state.{key}"], dtype=np.float32).reshape(-1)
        normalized[key] = _normalize_minmax(
            values,
            np.asarray(state_stats[key][min_key], dtype=np.float32).reshape(-1),
            np.asarray(state_stats[key][max_key], dtype=np.float32).reshape(-1),
        )
    return normalized


def _state_safety_score(normalized: dict[str, np.ndarray]) -> float:
    if not normalized:
        return float("inf")
    return float(max(np.abs(values).max(initial=0.0) for values in normalized.values()))


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


def _dataset_loader(
    dataset_path: Path,
    metadata: dict[str, Any],
    repo_root: Path,
    video_backend: str,
) -> Any:
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

    return LeRobotEpisodeLoader(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        video_backend=video_backend,
        video_backend_kwargs=None,
    )


def find_middle_safe_dataset_frame(
    dataset_path: Path,
    *,
    model_repo_root: Path | None = None,
    video_backend: str = "torchcodec",
    max_abs_normalized: float = 0.85,
    use_percentiles: bool = True,
) -> dict[str, Any]:
    """Find a middle-ish dataset frame whose normalized state stays away from +/-1."""
    dataset_path = dataset_path.expanduser().resolve()
    repo_root = (model_repo_root or _default_gr00t_repo_root()).expanduser().resolve()
    modality_path = dataset_path / "meta" / "modality.json"
    if not modality_path.is_file():
        raise FileNotFoundError(f"Dataset modality file does not exist: {modality_path}")
    metadata = json.loads(modality_path.read_text(encoding="utf-8"))
    _validate_modality_metadata(metadata)
    if not repo_root.is_dir():
        raise FileNotFoundError(f"N1.7 repository does not exist: {repo_root}")

    loader = _dataset_loader(dataset_path, metadata, repo_root, video_backend)
    state_stats = loader.get_dataset_statistics()["state"]
    episode_count = len(loader)
    if episode_count == 0:
        raise ValueError(f"Dataset has no episodes: {dataset_path}")

    ranked_episodes = sorted(range(episode_count), key=lambda idx: abs(idx - (episode_count - 1) / 2.0))
    best: dict[str, Any] | None = None
    for episode in ranked_episodes:
        episode_id = loader.episodes_metadata[episode]["episode_index"]
        episode_data = loader._load_parquet_data(episode_id)
        frame_count = min(len(episode_data), int(loader.episodes_metadata[episode]["length"]))
        if frame_count == 0:
            continue
        ranked_frames = sorted(range(frame_count), key=lambda idx: abs(idx - (frame_count - 1) / 2.0))
        for frame in ranked_frames:
            normalized = _state_normalized_values(
                episode_data.iloc[frame], state_stats, use_percentiles=use_percentiles
            )
            score = _state_safety_score(normalized)
            candidate = {
                "episode": int(episode),
                "episode_index": int(episode_id),
                "frame": int(frame),
                "max_abs_normalized_state": score,
                "normalized_state": {key: value.reshape(-1).tolist() for key, value in normalized.items()},
                "threshold": float(max_abs_normalized),
                "use_percentiles": bool(use_percentiles),
            }
            if best is None or score < best["max_abs_normalized_state"]:
                best = candidate
            if score <= max_abs_normalized:
                candidate["selected_by"] = "threshold"
                return candidate

    if best is None:
        raise ValueError(f"Could not inspect any frames in dataset: {dataset_path}")
    best["selected_by"] = "best_available"
    return best


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
    loader = _dataset_loader(dataset_path, metadata, repo_root, video_backend)
    modality_configs = loader.modality_configs
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
