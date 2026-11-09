"""Task-specific joint command and tracking safety checks."""

from dataclasses import dataclass
from typing import Any

import numpy as np


ARM_JOINT_NAMES = tuple(
    f"zarm_{side}{joint}_joint"
    for side in ("l", "r")
    for joint in range(1, 8)
)

APPLE_PICK_DELTA_Q_LIMIT = np.array(
    [0.07, 0.03, 0.07, 0.18, 0.18, 0.12, 0.08,
     0.07, 0.04, 0.09, 0.14, 0.17, 0.09, 0.12],
    dtype=np.float64,
)

BOTTLE_PICK_DELTA_Q_LIMIT = np.array(
    [0.12, 0.08, 0.16, 0.25, 0.19, 0.14, 0.14,
     0.12, 0.18, 0.16, 0.25, 0.19, 0.14, 0.14],
    dtype=np.float64,
)

PACKAGE_RECOVERY_DELTA_Q_LIMIT = np.array(
    [0.26, 0.14, 0.32, 0.40, 0.29, 0.24, 0.27,
     0.30, 0.36, 0.60, 0.55, 0.69, 0.32, 0.24],
    dtype=np.float64,
)
FLOWER_ARRANGEMENT_DELTA_Q_LIMIT = np.array(
    [
        0.13, 0.16, 0.19, 0.27, 0.27, 0.29, 0.24,
        0.12, 0.18, 0.16, 0.25, 0.19, 0.14, 0.14,
    ],
    dtype=np.float64,
)

@dataclass(frozen=True)
class SafetyResult:
    safe: bool
    reason: str
    details: dict[str, Any]


class SafetySupervisor:

    PROFILES = {
        "apple_pick": APPLE_PICK_DELTA_Q_LIMIT,
        "bottle_pick": BOTTLE_PICK_DELTA_Q_LIMIT,
        "package_recovery": PACKAGE_RECOVERY_DELTA_Q_LIMIT,
        "flower_arrangement": FLOWER_ARRANGEMENT_DELTA_Q_LIMIT,
    }

    def __init__(self, task_name: str):
        if task_name not in self.PROFILES:
            raise ValueError(
                f"Unknown safety task: {task_name}"
            )

        self.task_name = task_name
        self.delta_q_limit = self.PROFILES[task_name].copy()

    @staticmethod
    def parse_action(action: Any) -> np.ndarray:
        values = np.asarray(action, dtype=np.float64)

        if values.shape != (16,) or not np.isfinite(values).all():
            raise ValueError(
                f"Expected finite action (16,), got {values.shape}"
            )

        return np.concatenate(
            (values[:7], values[8:15])
        )

    @staticmethod
    def _arm14(q: Any, name: str) -> np.ndarray:
        values = np.asarray(q, dtype=np.float64)

        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError(
                f"{name} must have finite shape (14,)"
            )

        return values

    def _compare(
        self,
        q_cmd: Any,
        q_ref: Any,
        reason: str,
    ) -> SafetyResult:

        try:
            q_cmd = self._arm14(q_cmd, "q_cmd")
            q_ref = self._arm14(q_ref, "q_ref")
        except (TypeError, ValueError) as exc:
            return SafetyResult(
                False,
                "invalid_safety_input",
                {"error": str(exc)},
            )

        delta = np.abs(q_cmd - q_ref)

        indices = np.flatnonzero(
            delta > self.delta_q_limit
        ).tolist()

        details = {
            "safety_task": self.task_name,
            "violating_joint_indices": indices,
            "violating_joint_names": [
                ARM_JOINT_NAMES[i] for i in indices
            ],
            "q_ref": q_ref.copy(),
            "q_cmd": q_cmd.copy(),
            "delta_q": delta,
            "delta_q_limit": self.delta_q_limit.copy(),
        }

        return SafetyResult(
            safe=not indices,
            reason=reason if indices else "ok",
            details=details,
        )

    def check_command_continuity(
        self,
        action: Any,
        previous_command: Any,
    ) -> SafetyResult:

        try:
            q_cmd = self.parse_action(action)
        except (TypeError, ValueError) as exc:
            return SafetyResult(
                False,
                "invalid_safety_input",
                {"error": str(exc)},
            )

        return self._compare(
            q_cmd,
            previous_command,
            "command_continuity_limit_exceeded",
        )

    def check_tracking(
        self,
        last_command: Any,
        actual: Any,
    ) -> SafetyResult:

        # Temporary conservative guard.
        # These demonstration-derived limits are not
        # calibrated tracking-error thresholds.
        return self._compare(
            last_command,
            actual,
            "tracking_error_limit_exceeded",
        )

    def check(
        self,
        action: Any,
        q_ref: Any,
    ) -> SafetyResult:

        # Retained for first-step validation.
        try:
            q_cmd = self.parse_action(action)
        except (TypeError, ValueError) as exc:
            return SafetyResult(
                False,
                "invalid_safety_input",
                {"error": str(exc)},
            )

        return self._compare(
            q_cmd,
            q_ref,
            "initial_target_deviation_limit_exceeded",
        )