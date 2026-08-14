#!/usr/bin/env python3
"""Run E0 repeatability and E7 adapter-mapping controls via adapter server."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offline_jitter_diagnostic import (  # noqa: E402
    _numpy,
    build_gr00t_episode_loader,
    prepare_observation,
)


def parse_frames(raw: str) -> list[int]:
    try:
        frames = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("frames must be comma-separated integers") from exc
    if not frames or any(frame < 0 for frame in frames):
        raise argparse.ArgumentTypeError("frames must contain non-negative integers")
    return frames


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def expected_kuavo_chunk(
    raw_action_dict: dict[str, Any],
    action_keys: list[str],
    which_arm: str,
) -> np.ndarray:
    """Reconstruct Kuavo ordering for the known four-key NEW_EMBODIMENT layout."""
    expected_keys = ["left_arm", "left_gripper", "right_arm", "right_gripper"]
    if action_keys != expected_keys:
        raise ValueError(
            "E7 exact reconstruction currently requires action_keys "
            f"{expected_keys}, got {action_keys}"
        )
    pieces = []
    for key in action_keys:
        value = _numpy(raw_action_dict[key])
        if value.ndim != 3 or value.shape[0] != 1:
            raise ValueError(f"Expected raw action {key} as [1,T,D], got {value.shape}")
        pieces.append(value[0])
    full = np.concatenate(pieces, axis=-1).astype(np.float64)
    if which_arm == "left":
        return full[:, :8]
    if which_arm == "right":
        return full[:, 8:16]
    return full


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", type=parse_frames, required=True)
    parser.add_argument("--repeat-count", type=int, default=10)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--api-token", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat_count < 2:
        parser.error("--repeat-count must be at least 2")

    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        parser.error(f"Dataset root does not exist: {dataset_root}")

    from kuavo_deploy.kuavo_service.client import BaseInferenceClient

    client = BaseInferenceClient(host=args.host, port=args.port, api_token=args.api_token)
    if not client.ping():
        raise RuntimeError(f"Adapter server did not respond at {args.host}:{args.port}")
    metadata = client.call_endpoint("metadata", requires_input=False)
    if isinstance(metadata, dict) and metadata.get("error"):
        raise RuntimeError(metadata["error"])

    dataset = build_gr00t_episode_loader(dataset_root, metadata)
    if args.episode < 0 or args.episode >= len(dataset):
        parser.error(f"Episode {args.episode} is out of range [0, {len(dataset) - 1}]")
    episode = dataset[args.episode]
    invalid = [frame for frame in args.frames if frame >= len(episode)]
    if invalid:
        parser.error(f"Frames {invalid} exceed episode length {len(episode)}")

    video_keys = list(metadata["video_keys"])
    state_keys = list(metadata["state_keys"])
    action_keys = list(metadata["action_keys"])
    language_key = str(metadata["language_key"])
    which_arm = str(metadata["which_arm"])

    all_chunks: list[np.ndarray] = []
    all_latencies: list[np.ndarray] = []
    frame_reports: list[dict[str, Any]] = []
    mapping_errors: list[float] = []

    for frame in args.frames:
        observation = prepare_observation(
            episode.iloc[frame],
            video_keys=video_keys,
            state_keys=state_keys,
            language_key=language_key,
            prompt=args.prompt,
        )
        chunks: list[np.ndarray] = []
        latencies: list[float] = []
        frame_mapping_errors: list[float] = []
        for _ in range(args.repeat_count):
            started = time.perf_counter_ns()
            response = client.call_endpoint("diagnose_action_chunk", observation)
            latencies.append((time.perf_counter_ns() - started) / 1e6)
            if isinstance(response, dict) and response.get("error"):
                raise RuntimeError(response["error"])
            chunk = _numpy(response["kuavo_action_chunk"]).astype(np.float64)
            expected = expected_kuavo_chunk(
                response["raw_action_dict"], action_keys, which_arm
            )
            if chunk.shape != expected.shape:
                raise ValueError(
                    f"E7 shape mismatch: adapter={chunk.shape}, reconstructed={expected.shape}"
                )
            mapping_error = float(np.max(np.abs(chunk - expected)))
            chunks.append(chunk)
            frame_mapping_errors.append(mapping_error)

        chunk_array = np.stack(chunks)
        repeat_mean = np.mean(chunk_array, axis=0)
        deviations = np.linalg.norm(
            (chunk_array - repeat_mean).reshape(args.repeat_count, -1), axis=-1
        )
        per_element_std = np.std(chunk_array, axis=0)
        frame_report = {
            "frame": frame,
            "repeat_deviation_l2": summarize(deviations),
            "per_element_std": summarize(per_element_std),
            "first_action_per_element_std": summarize(per_element_std[0]),
            "request_latency_ms": summarize(np.asarray(latencies)),
            "adapter_mapping_max_abs_error": max(frame_mapping_errors),
        }
        frame_reports.append(frame_report)
        mapping_errors.extend(frame_mapping_errors)
        all_chunks.append(chunk_array)
        all_latencies.append(np.asarray(latencies))

    report = {
        "experiment": {
            "dataset_root": str(dataset_root),
            "episode": args.episode,
            "frames": args.frames,
            "repeat_count": args.repeat_count,
            "prompt": args.prompt,
        },
        "server_metadata": metadata,
        "e0_repeatability": frame_reports,
        "e7_adapter_mapping": {
            "max_abs_error": max(mapping_errors),
            "pass_at_1e-7": max(mapping_errors) <= 1e-7,
            "method": "Reconstruct Kuavo ordering from raw Gr00tPolicy action dict returned in the same response.",
        },
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "controls.npz",
        frames=np.asarray(args.frames),
        action_chunks=np.stack(all_chunks),
        request_latency_ms=np.stack(all_latencies),
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote E0/E7 controls to {output_dir}")


if __name__ == "__main__":
    main()
