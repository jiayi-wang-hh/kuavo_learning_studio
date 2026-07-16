from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _to_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _array_summary(value: Any) -> dict[str, Any]:
    arr = _to_numpy(value)
    result: dict[str, Any] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if arr.size and np.issubdtype(arr.dtype, np.number):
        finite = np.isfinite(arr)
        finite_values = arr[finite]
        result.update(
            {
                "nonfinite_count": int(np.count_nonzero(~finite)),
                "finite": bool(finite.all()),
                "min": float(finite_values.min()) if finite_values.size else None,
                "max": float(finite_values.max()) if finite_values.size else None,
                "mean": float(finite_values.mean()) if finite_values.size else None,
            }
        )
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "value"):
        return _json_value(value.value)
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dict__"):
        return {
            str(key): _json_value(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return value


def _processor_diagnostics(adapter: Any, model_obs: dict[str, Any]) -> dict[str, Any]:
    processor = adapter.model.policy.processor.state_action_processor
    embodiment = adapter.model.embodiment_value
    # Mirror Gr00tPolicy.get_action(): remove the transport batch dimension
    # before the checkpoint-native processor normalizes each state group.
    processor_state = {key: value[0] for key, value in model_obs["state"].items()}
    normalized = processor.apply_state(processor_state, embodiment)
    norm_params = processor.norm_params[embodiment]["state"]
    groups: dict[str, Any] = {}
    for key in adapter.model.state_keys:
        norm_arr = _to_numpy(normalized[key]).astype(np.float32)
        groups[key] = {
            "raw": _array_summary(processor_state[key]),
            "normalized": _array_summary(norm_arr),
            "normalized_values": norm_arr.reshape(-1).tolist(),
            "saturated_count": int(np.count_nonzero(np.isclose(np.abs(norm_arr), 1.0))),
            "element_count": int(norm_arr.size),
            "statistics": _json_value(norm_params[key]),
        }
    return {
        "use_percentiles": bool(processor.use_percentiles),
        "clip_outliers": bool(processor.clip_outliers),
        "groups": groups,
    }


def _lora_diagnostics(adapter: Any) -> dict[str, Any]:
    model = adapter.model.policy.model
    lora_parameters = [(name, param) for name, param in model.named_parameters() if "lora_" in name]
    nonzero_count = None
    if torch is not None:
        nonzero_count = int(
            sum(torch.count_nonzero(param.detach()).item() for _, param in lora_parameters)
        )
    return {
        "config_use_lora": bool(getattr(model.config, "use_lora", False)),
        "parameter_tensor_count": len(lora_parameters),
        "parameter_count": int(sum(param.numel() for _, param in lora_parameters)),
        "nonzero_parameter_count": nonzero_count,
        "sample_names": [name for name, _ in lora_parameters[:12]],
    }


def _action_dict_summary(action_dict: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for key, value in action_dict.items():
        arr = _to_numpy(value)
        summary = _array_summary(arr)
        if arr.size and np.issubdtype(arr.dtype, np.number):
            nonfinite = np.argwhere(~np.isfinite(arr))
            summary["nonfinite_indices"] = nonfinite[:20].tolist()
        groups[str(key)] = summary
    return groups


def _collect_action_chunks(
    adapter: Any,
    model_obs: dict[str, Any],
    repeats: int,
    seed: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    chunks: list[np.ndarray] = []
    raw_outputs: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for repeat_idx in range(repeats):
        if torch is not None:
            torch.manual_seed(seed + repeat_idx)
        try:
            action_dict = adapter.model.infer(model_obs)
        except Exception as exc:
            raw_outputs.append({"repeat": repeat_idx, "groups": {}})
            errors.append(
                {
                    "repeat": repeat_idx,
                    "stage": "model_infer",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue
        raw_outputs.append({"repeat": repeat_idx, "groups": _action_dict_summary(action_dict)})
        try:
            chunk = adapter._convert_action_chunk(action_dict)
            if adapter.execution_horizon is not None:
                chunk = chunk[: adapter.execution_horizon]
            chunk_array = np.stack(chunk, axis=0).astype(np.float64)
            if not np.isfinite(chunk_array).all():
                raise ValueError("converted action chunk contains NaN or Inf")
            chunks.append(chunk_array)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(
                {
                    "repeat": repeat_idx,
                    "type": type(exc).__name__,
                    "stage": "action_conversion",
                    "message": str(exc),
                }
            )
    return chunks, raw_outputs, errors


def _action_metrics(
    adapter: Any,
    raw_state16: np.ndarray,
    chunks: list[np.ndarray],
    raw_outputs: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    requested_repeats: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "ok" if not errors else "partial" if chunks else "invalid_model_output",
        "requested_repeats": requested_repeats,
        "valid_chunk_count": len(chunks),
        "invalid_repeat_count": len(errors),
        "errors": errors,
        "raw_model_outputs": raw_outputs,
    }
    if not chunks:
        result.update(
            {
                "shape": None,
                "first_chunk": None,
                "max_repeat_std": None,
                "max_first_action_repeat_std": None,
                "first_arm_delta_per_repeat": None,
                "max_abs_first_arm_delta": None,
                "max_abs_intra_chunk_step": None,
            }
        )
        return result

    stacked = np.stack(chunks, axis=0).astype(np.float64)
    if adapter.which_arm == "both":
        action_arm_indices = np.array([*range(7), *range(8, 15)], dtype=np.int64)
        current_arms = raw_state16[action_arm_indices]
    elif adapter.which_arm == "left":
        action_arm_indices = np.arange(7, dtype=np.int64)
        current_arms = raw_state16[:7]
    else:
        action_arm_indices = np.arange(7, dtype=np.int64)
        current_arms = raw_state16[8:15]
    first_arm_delta = stacked[:, 0, :][:, action_arm_indices] - current_arms[None, :]
    intra_step_delta = np.diff(stacked, axis=1)
    result.update(
        {
            "shape": list(stacked.shape),
            "first_chunk": stacked[0].tolist(),
            "max_repeat_std": float(stacked.std(axis=0).max()),
            "max_first_action_repeat_std": float(stacked[:, 0, :].std(axis=0).max()),
            "first_arm_delta_per_repeat": first_arm_delta.tolist(),
            "max_abs_first_arm_delta": float(np.abs(first_arm_delta).max()),
            "max_abs_intra_chunk_step": float(np.abs(intra_step_delta).max())
            if intra_step_delta.size
            else 0.0,
        }
    )
    return result

def diagnose_n17_observation(adapter: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Inspect a native N1.7 server path without executing robot actions."""
    if "observation" in request:
        obs = request["observation"]
        repeats = int(request.get("repeats", 5))
        seed = int(request.get("seed", 0))
    else:
        obs = request
        repeats = 5
        seed = 0
    repeats = max(1, min(repeats, 50))

    model_obs = adapter._build_model_obs(obs)
    raw_state16 = adapter._canonical_state16(obs["observation.state"])
    image_sources = {
        "head": "observation.images.head_cam_h",
        "wrist_left": "observation.images.wrist_cam_l",
        "wrist_right": "observation.images.wrist_cam_r",
    }
    images: dict[str, Any] = {}
    for name, source in image_sources.items():
        if source not in obs:
            images[name] = {"present": False}
            continue
        arr = adapter._canonical_image(obs[source])
        images[name] = {
            "present": True,
            **_array_summary(arr),
            "sha256": hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest(),
        }

    chunks, raw_outputs, action_errors = _collect_action_chunks(
        adapter, model_obs, repeats, seed
    )

    return {
        "adapter": adapter.name,
        "checkpoint": str(adapter.checkpoint),
        "embodiment_tag": adapter.model.embodiment_value,
        "repeats": repeats,
        "seed_start": seed,
        "raw_observation": {
            "keys": sorted(obs.keys()),
            "state16": raw_state16.tolist(),
            "state16_summary": _array_summary(raw_state16),
            "prompt": str(obs.get("prompt", "")),
            "images": images,
        },
        "model_observation": {
            "state_keys": adapter.model.state_keys,
            "action_keys": adapter.model.action_keys,
            "video_keys": adapter.model.video_keys,
            "state": {
                key: _to_numpy(model_obs["state"][key]).reshape(-1).tolist()
                for key in adapter.model.state_keys
            },
        },
        "processor": _processor_diagnostics(adapter, model_obs),
        "action_config": _json_value(adapter.model.modality["action"].action_configs),
        "lora": _lora_diagnostics(adapter),
        "actions": _action_metrics(
            adapter,
            raw_state16,
            chunks,
            raw_outputs,
            action_errors,
            repeats,
        ),
    }
