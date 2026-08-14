#!/usr/bin/env python3
"""Diagnose GR00T action-chunk jitter through a running adapter server.

The input is a local LeRobot v3 dataset.  Observations are sent through the
same ZMQ adapter path used by deployment; no robot or ROS environment is
required.  Outputs are intentionally dependency-light: NPZ, CSV and JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ModuleNotFoundError:
        pass
    return np.asarray(value)


def _scalar_int(value: Any) -> int:
    return int(_numpy(value).reshape(-1)[0])


def infer_local_repo_id(root: Path) -> str:
    """Read the repo id stored in a LeRobot v3 dataset, with a local fallback."""
    info_path = root / "meta" / "info.json"
    if info_path.is_file():
        with info_path.open(encoding="utf-8") as stream:
            info = json.load(stream)
        for key in ("repo_id", "dataset_repo_id"):
            if info.get(key):
                return str(info[key])
    return f"local/{root.name}"


def iter_episode_samples(
    dataset: Any,
    episode_index: int,
    *,
    start_frame: int,
    max_frames: int | None,
    stride: int,
) -> Iterator[tuple[int, Any]]:
    """Yield frames from one episode without relying on private LeRobot APIs."""
    emitted = 0
    episode_frame = 0
    for dataset_index in range(len(dataset)):
        sample = dataset[dataset_index]
        if _scalar_int(sample["episode_index"]) != episode_index:
            continue
        if episode_frame >= start_frame and (episode_frame - start_frame) % stride == 0:
            yield dataset_index, sample
            emitted += 1
            if max_frames is not None and emitted >= max_frames:
                return
        episode_frame += 1


def prepare_observation(sample: dict[str, Any], prompt: str) -> dict[str, Any]:
    observation = {
        key: value
        for key, value in sample.items()
        if key == "observation.state" or key.startswith("observation.images.")
    }
    if "observation.state" not in observation:
        raise KeyError("Dataset sample is missing observation.state")
    if not any(key.startswith("observation.images.") for key in observation):
        raise KeyError("Dataset sample has no observation.images.* fields")
    observation["prompt"] = prompt or str(sample.get("task", "robot manipulation"))
    return observation


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def summarize(values: np.ndarray) -> dict[str, float | int | None]:
    values = _finite(values)
    if values.size == 0:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def compute_metrics(
    chunks: np.ndarray,
    *,
    dt: float,
    execution_horizon: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if chunks.ndim != 3:
        raise ValueError(f"Expected chunks [N,T,D], got {chunks.shape}")
    if chunks.shape[1] < 3:
        raise ValueError("Action horizon must be at least 3 for acceleration diagnostics")
    execute_steps = min(execution_horizon, chunks.shape[1])
    if execute_steps < 2:
        raise ValueError("execution_horizon must be at least 2 for boundary diagnostics")

    velocity = np.diff(chunks, axis=1) / dt
    acceleration = np.diff(velocity, axis=1) / dt
    intra_norm = np.linalg.norm(acceleration, axis=-1)

    previous = chunks[:-1, execute_steps - 1, :]
    following = chunks[1:, 0, :]
    boundary_jump = np.linalg.norm(following - previous, axis=-1)
    v_previous = chunks[:-1, execute_steps - 1, :] - chunks[:-1, execute_steps - 2, :]
    v_following = chunks[1:, 1, :] - chunks[1:, 0, :]
    denominator = np.linalg.norm(v_previous, axis=-1) * np.linalg.norm(v_following, axis=-1)
    momentum_cos = np.sum(v_previous * v_following, axis=-1) / (denominator + 1e-8)

    per_chunk = [
        {
            "chunk_id": index,
            **summarize(intra_norm[index]),
        }
        for index in range(chunks.shape[0])
    ]
    per_boundary = [
        {
            "boundary_id": index,
            "previous_chunk": index,
            "next_chunk": index + 1,
            "position_jump_l2": float(boundary_jump[index]),
            "velocity_cosine": float(momentum_cos[index]),
        }
        for index in range(boundary_jump.shape[0])
    ]
    overall = {
        "intra_chunk_acceleration": summarize(intra_norm),
        "boundary_position_jump": summarize(boundary_jump),
        "boundary_velocity_cosine": summarize(momentum_cos),
    }
    return overall, per_chunk, per_boundary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="", help="Defaults to meta/info.json or local/<directory>.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument(
        "--stride",
        type=int,
        default=0,
        help="Dataset frames between requests; 0 uses execution_horizon.",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--api-token", default=None)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--fps", type=float, default=0.0, help="0 uses dataset metadata.")
    parser.add_argument("--execution-horizon", type=int, default=0, help="0 uses server metadata.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.stride < 0 or args.max_frames < 1:
        parser.error("--stride must be non-negative and --max-frames must be positive")
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        parser.error(f"Dataset root does not exist: {dataset_root}")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from kuavo_deploy.kuavo_service.client import BaseInferenceClient
    except ModuleNotFoundError as exc:
        parser.error(f"Missing runtime dependency: {exc}. Run this tool in the Kuavo/LeRobot environment.")

    repo_id = args.repo_id or infer_local_repo_id(dataset_root)
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
    client = BaseInferenceClient(
        host=args.host,
        port=args.port,
        api_token=args.api_token,
    )
    if not client.ping():
        raise RuntimeError(f"Adapter server did not respond at {args.host}:{args.port}")
    metadata = client.call_endpoint("metadata", requires_input=False)
    if isinstance(metadata, dict) and metadata.get("error"):
        raise RuntimeError(metadata["error"])

    fps = args.fps or float(getattr(dataset.meta, "fps", 0) or 0)
    if not math.isfinite(fps) or fps <= 0:
        parser.error("Could not determine dataset FPS; pass --fps explicitly")
    server_horizon = metadata.get("execution_horizon") or metadata.get("model_action_horizon")
    execution_horizon = args.execution_horizon or int(server_horizon or 0)
    if execution_horizon < 2:
        parser.error("Could not determine execution horizon; pass --execution-horizon >= 2")
    stride = args.stride or execution_horizon

    chunks: list[np.ndarray] = []
    states: list[np.ndarray] = []
    references: list[np.ndarray] = []
    dataset_indices: list[int] = []
    latencies_ms: list[float] = []
    raw_actions: list[dict[str, np.ndarray]] = []

    for dataset_index, sample in iter_episode_samples(
        dataset,
        args.episode,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        stride=stride,
    ):
        observation = prepare_observation(sample, args.prompt)
        started = time.perf_counter_ns()
        response = client.call_endpoint("diagnose_action_chunk", observation)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(response["error"])
        chunk = _numpy(response["kuavo_action_chunk"]).astype(np.float64)
        if chunks and chunk.shape != chunks[0].shape:
            raise ValueError(f"Variable chunk shapes: first={chunks[0].shape}, current={chunk.shape}")
        chunks.append(chunk)
        states.append(_numpy(response["observation_state16"]).astype(np.float64))
        references.append(_numpy(sample["action"]).astype(np.float64).reshape(-1))
        dataset_indices.append(dataset_index)
        latencies_ms.append(elapsed_ms)
        raw_actions.append({key: _numpy(value) for key, value in response["raw_action_dict"].items()})

    if len(chunks) < 2:
        raise RuntimeError(
            f"Need at least two sampled frames in episode {args.episode}; got {len(chunks)}"
        )

    chunk_array = np.stack(chunks)
    state_array = np.stack(states)
    reference_array = np.stack(references)
    overall, per_chunk, per_boundary = compute_metrics(
        chunk_array,
        dt=1.0 / fps,
        execution_horizon=execution_horizon,
    )
    overall["request_latency_ms"] = summarize(np.asarray(latencies_ms))

    comparable_dim = min(chunk_array.shape[-1], reference_array.shape[-1])
    one_step_error = chunk_array[:, 0, :comparable_dim] - reference_array[:, :comparable_dim]
    overall["first_action_vs_dataset_l2"] = summarize(np.linalg.norm(one_step_error, axis=-1))
    for row, dataset_index, latency in zip(per_chunk, dataset_indices, latencies_ms):
        row["dataset_index"] = dataset_index
        row["request_latency_ms"] = latency

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_payload = np.asarray(raw_actions, dtype=object)
    np.savez_compressed(
        output_dir / "chunks.npz",
        kuavo_action_chunks=chunk_array,
        observation_state16=state_array,
        dataset_action=reference_array,
        dataset_indices=np.asarray(dataset_indices),
        request_latency_ms=np.asarray(latencies_ms),
        raw_action_dicts=raw_payload,
    )
    write_csv(output_dir / "per_chunk.csv", per_chunk)
    write_csv(output_dir / "per_boundary.csv", per_boundary)
    report = {
        "experiment": {
            "dataset_root": str(dataset_root),
            "repo_id": repo_id,
            "episode": args.episode,
            "start_frame": args.start_frame,
            "sample_count": len(chunks),
            "stride": stride,
            "fps": fps,
            "execution_horizon": execution_horizon,
            "offline_limit": "Cannot diagnose motor, controller, encoder, or mechanical vibration without robot feedback.",
        },
        "server_metadata": metadata,
        "metrics": overall,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote offline jitter diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
