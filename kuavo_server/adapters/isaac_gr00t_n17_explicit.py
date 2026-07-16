from __future__ import annotations

from argparse import ArgumentParser
from typing import Any

import numpy as np

from ..runtime import register_adapter
from .isaac_gr00t_n17 import IsaacGr00tN17Adapter, _as_hwc_uint8, _to_numpy


_EXPECTED_VIDEO_KEYS = ("head", "wrist_left", "wrist_right")
_EXPECTED_STATE_KEYS = ("left_arm", "left_gripper", "right_arm", "right_gripper")
_EXPECTED_ACTION_KEYS = _EXPECTED_STATE_KEYS
_VIDEO_SOURCES = {
    "head": "observation.images.head_cam_h",
    "wrist_left": "observation.images.wrist_cam_l",
    "wrist_right": "observation.images.wrist_cam_r",
}
_STATE_SLICES = {
    "left_arm": slice(0, 7),
    "left_gripper": slice(7, 8),
    "right_arm": slice(8, 15),
    "right_gripper": slice(15, 16),
}
_EXPECTED_DIMS = {
    "left_arm": 7,
    "left_gripper": 1,
    "right_arm": 7,
    "right_gripper": 1,
}


@register_adapter
class IsaacGr00tN17ExplicitAdapter(IsaacGr00tN17Adapter):
    """Native N1.7 Kuavo path with exact IO contracts and no heuristic fallback."""

    name = "isaac_gr00t_n17_explicit"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._validate_checkpoint_contract()

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        super().add_cli_args(parser)

    def _validate_checkpoint_contract(self) -> None:
        actual = {
            "video": tuple(self.model.video_keys),
            "state": tuple(self.model.state_keys),
            "action": tuple(self.model.action_keys),
        }
        expected = {
            "video": _EXPECTED_VIDEO_KEYS,
            "state": _EXPECTED_STATE_KEYS,
            "action": _EXPECTED_ACTION_KEYS,
        }
        errors = [
            f"{kind}: expected={expected[kind]}, actual={actual[kind]}"
            for kind in expected
            if actual[kind] != expected[kind]
        ]
        for key, expected_dim in _EXPECTED_DIMS.items():
            state_dim = self.model.state_dims.get(key)
            action_dim = self.model.action_dims.get(key)
            if state_dim != expected_dim:
                errors.append(f"state.{key}: expected dim={expected_dim}, actual={state_dim}")
            if action_dim != expected_dim:
                errors.append(f"action.{key}: expected dim={expected_dim}, actual={action_dim}")
        if errors:
            raise ValueError(
                "Checkpoint does not satisfy the explicit Kuavo N1.7 contract:\n  "
                + "\n  ".join(errors)
            )

    def _canonical_state16(self, raw_state: Any) -> np.ndarray:
        state = _to_numpy(raw_state).astype(np.float32).reshape(-1)
        if state.shape != (16,):
            raise ValueError(f"Expected exact Kuavo state shape (16,), got {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("Kuavo state contains NaN or Inf")
        return state

    @staticmethod
    def _canonical_image(image: Any) -> np.ndarray:
        return _as_hwc_uint8(image)

    def _build_model_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        required = {"observation.state", *_VIDEO_SOURCES.values()}
        missing = sorted(required.difference(obs))
        if missing:
            raise KeyError(f"Missing explicit Kuavo N1.7 observation keys: {missing}")

        state16 = self._canonical_state16(obs["observation.state"])
        self._last_state16 = state16
        state = {
            key: state16[state_slice][None, None, :].astype(np.float32)
            for key, state_slice in _STATE_SLICES.items()
        }
        video = {
            key: self._canonical_image(obs[source])[None, None, ...]
            for key, source in _VIDEO_SOURCES.items()
        }
        language = {self.model.language_key: [[str(obs.get("prompt", ""))]]}
        return {"video": video, "state": state, "language": language}

    def _compose_kuavo_action(self, action_step: dict[str, np.ndarray]) -> np.ndarray:
        missing = sorted(set(_EXPECTED_ACTION_KEYS).difference(action_step))
        extra = sorted(set(action_step).difference(_EXPECTED_ACTION_KEYS))
        if missing or extra:
            raise ValueError(f"Explicit action key mismatch: missing={missing}, extra={extra}")

        pieces = []
        for key in _EXPECTED_ACTION_KEYS:
            piece = _to_numpy(action_step[key]).astype(np.float32).reshape(-1)
            expected_dim = _EXPECTED_DIMS[key]
            if piece.shape != (expected_dim,):
                raise ValueError(
                    f"Expected action.{key} shape ({expected_dim},), got {piece.shape}"
                )
            if not np.isfinite(piece).all():
                raise ValueError(f"action.{key} contains NaN or Inf")
            pieces.append(piece)

        full = np.concatenate(pieces, axis=0).astype(np.float64)
        if self.which_arm == "left":
            return full[:8]
        if self.which_arm == "right":
            return full[8:16]
        return full
