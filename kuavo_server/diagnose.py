from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import numpy as np
import torch

from kuavo_deploy.kuavo_service.client import ExternalRobotInferenceClient
from kuavo_server.dataset_observation import find_middle_safe_dataset_frame, load_dataset_observation


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _parse_server(value: str) -> tuple[str, str, int]:
    try:
        name, address = value.split("=", 1)
        host, port = address.rsplit(":", 1)
        return name, host, int(port)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("server must use NAME=HOST:PORT") from exc


def _raw_observation_report(observation: dict[str, Any]) -> dict[str, Any]:
    state = _to_numpy(observation["observation.state"]).astype(np.float32).reshape(-1)
    images = {}
    for key in sorted(k for k in observation if k.startswith("observation.images.")):
        image = _to_numpy(observation[key])
        images[key] = {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
        }
    return {
        "keys": sorted(observation),
        "state": state.tolist(),
        "state_shape": list(state.shape),
        "prompt": str(observation.get("prompt", "")),
        "images": images,
    }


def _generic_action_probe(
    client: ExternalRobotInferenceClient,
    observation: dict[str, Any],
    repeats: int,
) -> dict[str, Any]:
    chunks = []
    for _ in range(repeats):
        chunk = _to_numpy(client.select_action_chunk(observation)).astype(np.float64)
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        chunks.append(chunk)
    stacked = np.stack(chunks, axis=0)
    state = _to_numpy(observation["observation.state"]).reshape(-1)
    if stacked.shape[-1] == 16 and state.size == 16:
        arm_indices = np.array([*range(7), *range(8, 15)])
        first_delta = stacked[:, 0, arm_indices] - state[arm_indices][None, :]
        max_first_delta = float(np.abs(first_delta).max())
    else:
        first_delta = None
        max_first_delta = None
    return {
        "mode": "generic_select_action_chunk",
        "shape": list(stacked.shape),
        "max_repeat_std": float(stacked.std(axis=0).max()),
        "max_first_action_repeat_std": float(stacked[:, 0, :].std(axis=0).max()),
        "max_abs_first_arm_delta": max_first_delta,
        "first_arm_delta_per_repeat": first_delta.tolist() if first_delta is not None else None,
        "first_chunk": stacked[0].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe N1.5 and N1.7 servers with the exact same Kuavo observation."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--observation", type=Path, help="torch.save() observation file")
    source.add_argument("--dataset-path", type=Path, help="Dual-arm Kuavo LeRobot dataset root")
    parser.add_argument("--episode", type=int, default=0, help="Dataset episode index")
    parser.add_argument("--frame", type=int, default=0, help="Frame index inside the episode")
    parser.add_argument(
        "--auto-middle-safe-frame",
        action="store_true",
        help="For --dataset-path, auto-select a middle-ish frame whose percentile-normalized state is away from +/-1.",
    )
    parser.add_argument(
        "--safe-state-threshold",
        type=float,
        default=0.85,
        help="Maximum allowed abs(percentile-normalized state) for --auto-middle-safe-frame.",
    )
    parser.add_argument(
        "--model-repo-root",
        type=Path,
        default=None,
        help="N1.7 source tree; defaults to kuavo_model/external_models/gr00tn1d7",
    )
    parser.add_argument("--video-backend", type=str, default="torchcodec")
    parser.add_argument(
        "--server",
        action="append",
        type=_parse_server,
        required=True,
        help="Repeatable NAME=HOST:PORT, e.g. n15=localhost:5555",
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", type=str, default=None, help="Override the saved/dataset prompt")
    parser.add_argument("--output", type=Path, default=Path("n15_n17_server_probe.json"))
    args = parser.parse_args()

    if args.dataset_path is not None:
        selected_frame = None
        episode = args.episode
        frame = args.frame
        if args.auto_middle_safe_frame:
            selected_frame = find_middle_safe_dataset_frame(
                args.dataset_path,
                model_repo_root=args.model_repo_root,
                video_backend=args.video_backend,
                max_abs_normalized=args.safe_state_threshold,
                use_percentiles=True,
            )
            episode = int(selected_frame["episode"])
            frame = int(selected_frame["frame"])
        observation, observation_source = load_dataset_observation(
            args.dataset_path,
            episode=episode,
            frame=frame,
            prompt_override=args.prompt,
            model_repo_root=args.model_repo_root,
            video_backend=args.video_backend,
        )
        if selected_frame is not None:
            observation_source["auto_middle_safe_frame"] = selected_frame
    else:
        observation = torch.load(args.observation, map_location="cpu", weights_only=False)
        if not isinstance(observation, dict):
            raise TypeError(f"Expected a saved observation dict, got {type(observation)}")
        if args.prompt is not None:
            observation = dict(observation)
            observation["prompt"] = args.prompt
        observation_source = {
            "type": "torch_file",
            "path": str(args.observation.expanduser().resolve()),
        }

    report: dict[str, Any] = {
        "observation_source": observation_source,
        "raw_observation": _raw_observation_report(observation),
        "repeats": args.repeats,
        "seed": args.seed,
        "servers": {},
    }
    for name, host, port in args.server:
        client = ExternalRobotInferenceClient(host=host, port=port)
        try:
            try:
                server_report = client.diagnose_observation(
                    observation, repeats=args.repeats, seed=args.seed
                )
                server_report["mode"] = "diagnose_observation"
            except RuntimeError as exc:
                if "Unknown endpoint: diagnose_observation" not in str(exc):
                    raise
                server_report = _generic_action_probe(client, observation, args.repeats)
            report["servers"][name] = server_report
        finally:
            client.socket.close(linger=0)
            client.context.term()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved comparison report to {args.output.resolve()}")


if __name__ == "__main__":
    main()
