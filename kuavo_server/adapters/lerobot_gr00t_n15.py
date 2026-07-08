from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from ..runtime import register_adapter
from .base import ModelServerAdapter, resolve_model_repo_root


def _resolve_repo_root(model_repo_root: str | None = None) -> Path:
    return resolve_model_repo_root("lerobot_gr00t_n15", model_repo_root)


def _ensure_repo_import_paths(repo_root: Path) -> None:
    candidates = [
        repo_root,
        repo_root / "src",
        repo_root / "third_party" / "try",
        repo_root / "third_party" / "try" / "lerobot",
        repo_root / "third_party" / "lerobot" / "src",
    ]
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _to_numpy(x: Any) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _as_hwc_uint8(img: Any) -> np.ndarray:
    arr = _to_numpy(img)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.max(initial=0) <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected 3-channel image, got {arr.shape}")
    return arr


def _image_to_chw_tensor(img: Any) -> Any:
    if torch is None:
        raise ModuleNotFoundError("torch is required for LeRobot GR00T N1.5 inference")
    arr = _as_hwc_uint8(img)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float() / 255.0


def _fit_dim(vec: np.ndarray, dim: int, fill: float = 0.0) -> np.ndarray:
    arr = vec.astype(np.float32).reshape(-1)
    if arr.shape[0] == dim:
        return arr
    if arr.shape[0] > dim:
        return arr[:dim]
    pad = np.full((dim - arr.shape[0],), fill, dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


def _kuavo_state16(raw_state: Any, which_arm: str) -> np.ndarray:
    state = _to_numpy(raw_state).astype(np.float32).reshape(-1)
    if state.shape[0] == 16:
        return state
    if state.shape[0] == 14:
        return np.concatenate([state[:7], np.zeros(1, dtype=np.float32), state[7:14], np.zeros(1, dtype=np.float32)])
    if state.shape[0] == 8:
        if which_arm == "left":
            return np.concatenate([state[:7], state[7:8], np.zeros(7, dtype=np.float32), np.zeros(1, dtype=np.float32)])
        if which_arm == "right":
            return np.concatenate([np.zeros(7, dtype=np.float32), np.zeros(1, dtype=np.float32), state[:7], state[7:8]])
    if state.shape[0] > 16:
        return state[:16]
    return _fit_dim(state, 16)


def _normalize_key(key: str) -> str:
    return key.lower().replace(".", "_").replace("-", "_")


def _is_left_key(name: str) -> bool:
    return any(token in name for token in ("left", "l_", "_l", "arm_l", "zarm_l", "larm"))


def _is_right_key(name: str) -> bool:
    return any(token in name for token in ("right", "r_", "_r", "arm_r", "zarm_r", "rarm"))


def _is_gripper_key(name: str) -> bool:
    return any(token in name for token in ("gripper", "claw", "hand", "effector"))


def _is_arm_like(name: str) -> bool:
    return any(token in name for token in ("arm", "joint", "qpos", "zarm", "link", "state", "position"))


def _to_numpy_action(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        if "action" not in value:
            raise KeyError(f"Postprocessor returned a dict without 'action': {value.keys()}")
        value = value["action"]
    arr = _to_numpy(value).astype(np.float32)
    while arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.reshape(-1)


class _LeRobotGr00tN15Runtime:
    def __init__(
        self,
        *,
        repo_root: Path,
        checkpoint: Path,
        device: str,
    ) -> None:
        if torch is None:
            raise ModuleNotFoundError("torch is required for LeRobot GR00T N1.5 inference")

        print(f"[lerobot-gr00t-n15] repo_root={repo_root}", flush=True)
        print(f"[lerobot-gr00t-n15] checkpoint={checkpoint}", flush=True)
        _ensure_repo_import_paths(repo_root)

        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        from lerobot.processor import DeviceProcessorStep

        self.policy = GrootPolicy.from_pretrained(checkpoint)
        self.policy = self.policy.to(device).eval()
        self.device = device

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint,
            dataset_stats=None,
        )
        for step in getattr(self.preprocessor, "steps", []):
            if isinstance(step, DeviceProcessorStep):
                step.device = device
                step.__post_init__()

        action_feature = self.policy.config.output_features.get("action")
        self.action_dim = int(action_feature.shape[0]) if action_feature and action_feature.shape else None
        state_feature = self.policy.config.input_features.get("observation.state")
        self.state_dim = int(state_feature.shape[0]) if state_feature and state_feature.shape else None
        self.action_horizon = int(min(self.policy.config.chunk_size, 16))
        print(
            f"[lerobot-gr00t-n15] policy loaded state_dim={self.state_dim} action_dim={self.action_dim} "
            f"action_horizon={self.action_horizon} device={device}",
            flush=True,
        )

    def reset(self) -> None:
        self.policy.reset()

    def infer_chunk(self, policy_input: dict[str, Any], horizon: int | None) -> np.ndarray:
        with torch.inference_mode():
            processed = self.preprocessor(policy_input)
            raw_chunk = self.policy.predict_action_chunk(processed)
            raw_horizon = int(raw_chunk.shape[1])
            limit = raw_horizon if horizon is None else min(int(horizon), raw_horizon)
            actions = [_to_numpy_action(self.postprocessor(raw_chunk[:, idx])) for idx in range(limit)]
        if not actions:
            raise ValueError("LeRobot GR00T N1.5 returned empty action chunk.")
        return np.stack(actions, axis=0)


@register_adapter
class LeRobotGr00tN15Adapter(ModelServerAdapter):
    """Serve LeRobot GR00T N1.5 checkpoints through the Kuavo ZMQ runtime."""

    name = "lerobot_gr00t_n15"

    def __init__(
        self,
        *,
        model_repo_root: str,
        checkpoint: str,
        which_arm: str,
        execution_horizon: int | None,
        device: str,
    ) -> None:
        self.model_repo_root = _resolve_repo_root(model_repo_root)
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Model checkpoint dir does not exist: {self.checkpoint}")

        self.which_arm = which_arm
        self.execution_horizon = execution_horizon if execution_horizon and execution_horizon > 0 else None
        self._pending_actions: list[np.ndarray] = []
        self._last_state16: np.ndarray = np.zeros(16, dtype=np.float32)
        self._warned_missing_depth: set[str] = set()

        print(f"[lerobot-gr00t-n15] initializing adapter={self.name}", flush=True)
        self.model = _LeRobotGr00tN15Runtime(
            repo_root=self.model_repo_root,
            checkpoint=self.checkpoint,
            device=device,
        )

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--model_repo_root",
            type=str,
            default="",
            help="Optional path to openpi repo root that contains third_party/try/lerobot. Defaults to external_models/openpi.",
        )
        parser.add_argument("--checkpoint", type=str, required=True, help="Path to LeRobot GR00T N1.5 pretrained_model dir")
        parser.add_argument("--which_arm", type=str, default="both", choices=["left", "right", "both"])
        parser.add_argument(
            "--execution_horizon",
            type=int,
            default=16,
            help="Number of actions to execute per chunk. <=0 means use full model chunk.",
        )
        parser.add_argument("--device", type=str, default="cuda", help="Torch device, e.g. cuda, cuda:0, or cpu")

    @classmethod
    def from_args(cls, args: Namespace) -> "LeRobotGr00tN15Adapter":
        return cls(
            model_repo_root=args.model_repo_root,
            checkpoint=args.checkpoint,
            which_arm=args.which_arm,
            execution_horizon=args.execution_horizon,
            device=args.device,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "adapter": self.name,
            "model_repo_root": str(self.model_repo_root),
            "checkpoint": str(self.checkpoint),
            "which_arm": self.which_arm,
            "execution_horizon": self.execution_horizon or self.model.action_horizon,
            "state_dim": self.model.state_dim,
            "action_dim": self.model.action_dim,
        }

    def reset(self) -> dict[str, Any]:
        self._pending_actions.clear()
        self.model.reset()
        return {"status": "ok", "message": "adapter state cleared"}

    def _state_for_policy(self, obs: dict[str, Any]) -> np.ndarray:
        kuavo_state16 = _kuavo_state16(obs["observation.state"], self.which_arm)
        self._last_state16 = kuavo_state16
        state_dim = self.model.state_dim

        if state_dim == 8 and self.which_arm == "left":
            return kuavo_state16[:8]
        if state_dim == 8 and self.which_arm == "right":
            return kuavo_state16[8:16]
        if state_dim == 8 and self.which_arm == "both":
            raise ValueError(
                "Checkpoint expects an 8-D single-arm state, but adapter was started with which_arm=both. "
                "Restart with --which_arm left or --which_arm right."
            )
        if state_dim == 14:
            return np.concatenate([kuavo_state16[:7], kuavo_state16[8:15]], axis=0)
        if state_dim == 16:
            return kuavo_state16
        if state_dim is not None:
            return _fit_dim(kuavo_state16, state_dim)
        return kuavo_state16

    def _image_for_feature(self, key: str, obs: dict[str, Any]) -> Any:
        name = _normalize_key(key)
        if "left" in name or "_l" in name:
            source = "observation.images.wrist_cam_l"
        elif "right" in name or "_r" in name:
            source = "observation.images.wrist_cam_r"
        elif "wrist" in name:
            source = "observation.images.wrist_cam_r"
        else:
            source = "observation.images.head_cam_h"
        return _image_to_chw_tensor(obs.get(source, obs["observation.images.head_cam_h"]))

    def _depth_for_feature(self, key: str, obs: dict[str, Any]) -> Any:
        if key in obs:
            return _image_to_chw_tensor(obs[key])

        if key not in self._warned_missing_depth:
            print(
                f"[lerobot-gr00t-n15] Missing depth feature '{key}', filling with zeros. "
                "For best performance, provide the depth observation used during training.",
                flush=True,
            )
            self._warned_missing_depth.add(key)

        return torch.zeros_like(self._image_for_feature(key, obs))

    def _build_policy_input(self, obs: dict[str, Any]) -> dict[str, Any]:
        if "observation.state" not in obs:
            raise KeyError("Missing required observation.state")
        if "observation.images.head_cam_h" not in obs:
            raise KeyError("Missing required observation.images.head_cam_h")

        policy_input: dict[str, Any] = {"task": str(obs.get("prompt", "")) or "Perform the task."}
        input_features = self.model.policy.config.input_features
        for key in input_features:
            if key == "observation.state":
                policy_input[key] = torch.as_tensor(self._state_for_policy(obs), dtype=torch.float32)
            elif key in obs and key.startswith("observation.images."):
                policy_input[key] = _image_to_chw_tensor(obs[key])
            elif key.startswith("observation.depth"):
                policy_input[key] = self._depth_for_feature(key, obs)
            elif key.startswith("observation.images.") or key == "observation.image":
                policy_input[key] = self._image_for_feature(key, obs)
            elif key in obs:
                policy_input[key] = obs[key]
            else:
                raise KeyError(
                    f"Cannot build LeRobot GR00T N1.5 input feature '{key}'. "
                    f"Available observation keys: {sorted(obs.keys())}"
                )
        return policy_input

    def _convert_action(self, action: Any) -> np.ndarray:
        action_np = _to_numpy(action).reshape(-1).astype(np.float64)

        if action_np.shape[0] == 16:
            if self.which_arm == "both":
                return action_np
            if self.which_arm == "left":
                return action_np[:8]
            if self.which_arm == "right":
                return action_np[8:16]

        if action_np.shape[0] == 14:
            full = np.concatenate(
                [action_np[:7], self._last_state16[7:8], action_np[7:14], self._last_state16[15:16]],
                axis=0,
            )
            if self.which_arm == "both":
                return full
            if self.which_arm == "left":
                return np.concatenate([action_np[:7], self._last_state16[7:8]], axis=0)
            if self.which_arm == "right":
                return np.concatenate([action_np[7:14], self._last_state16[15:16]], axis=0)

        if action_np.shape[0] == 8:
            if self.which_arm in ("left", "right"):
                return action_np
            raise ValueError(
                "LeRobot GR00T N1.5 returned an 8-D single-arm action, but adapter was started "
                "with which_arm=both. Restart with --which_arm left or --which_arm right."
            )

        if self.model.action_dim is not None and action_np.shape[0] == self.model.action_dim:
            return action_np

        raise ValueError(
            f"Unsupported LeRobot GR00T N1.5 action shape {action_np.shape} "
            f"for which_arm={self.which_arm} action_dim={self.model.action_dim}"
        )

    def _predict_action_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        policy_input = self._build_policy_input(obs)
        action_chunk = self.model.infer_chunk(policy_input, self.execution_horizon)
        return np.stack([self._convert_action(step) for step in action_chunk], axis=0)

    def select_action(self, obs: dict[str, Any]) -> np.ndarray:
        if self._pending_actions:
            return self._pending_actions.pop(0)

        action_chunk = self._predict_action_chunk(obs)
        self._pending_actions = [np.asarray(step) for step in action_chunk[1:]]
        return np.asarray(action_chunk[0])

    def select_action_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        self._pending_actions.clear()
        return self._predict_action_chunk(obs)
