"""Task-specific action continuity checks for real-robot inference."""

from dataclasses import dataclass
from typing import Any

import numpy as np


ARM_JOINT_NAMES = (
    "zarm_l1_joint",
    "zarm_l2_joint",
    "zarm_l3_joint",
    "zarm_l4_joint",
    "zarm_l5_joint",
    "zarm_l6_joint",
    "zarm_l7_joint",
    "zarm_r1_joint",
    "zarm_r2_joint",
    "zarm_r3_joint",
    "zarm_r4_joint",
    "zarm_r5_joint",
    "zarm_r6_joint",
    "zarm_r7_joint",
)

# Demonstration-derived task-specific joint command continuity limits for
# apple_pick. The demonstration dataset was recorded at approximately 30 Hz;
# within-episode 3-frame differences approximate one approximately 0.1 s
# inference step. Each value is observed max(abs(delta_q)) * 1.2, rounded
# upward to 0.01 rad.
#
# These are NOT hardware safety limits, torque limits, velocity limits, or
# URDF joint limits.
APPLE_PICK_DELTA_Q_LIMIT = np.array(
    [
        0.07,  # L1
        0.03,  # L2
        0.07,  # L3
        0.18,  # L4
        0.18,  # L5
        0.12,  # L6
        0.08,  # L7
        0.07,  # R1
        0.04,  # R2
        0.09,  # R3
        0.14,  # R4
        0.17,  # R5
        0.09,  # R6
        0.12,  # R7
    ],
    dtype=np.float64,
)


# Demonstration-derived task-specific joint command continuity limits for
# bottle_pick. The dataset contains 95 episodes and 13,965 frames sampled at
# 0.1 s intervals. Each value is the within-episode adjacent-frame 99.9th
# percentile of abs(delta_q), multiplied by 1.2 and rounded upward to 0.01 rad.
#
# These are NOT hardware safety limits, torque limits, velocity limits, or
# URDF joint limits.
BOTTLE_PICK_DELTA_Q_LIMIT = np.array(
    [
        0.11,  # L1
        0.04,  # L2
        0.12,  # L3
        0.20,  # L4
        0.19,  # L5
        0.13,  # L6
        0.14,  # L7
        0.08,  # R1
        0.05,  # R2
        0.11,  # R3
        0.25,  # R4
        0.14,  # R5
        0.13,  # R6
        0.12,  # R7
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class SafetyProfile:
    delta_q_limit: np.ndarray


SAFETY_PROFILES = {
    "apple_pick": SafetyProfile(delta_q_limit=APPLE_PICK_DELTA_Q_LIMIT),
    "bottle_pick": SafetyProfile(delta_q_limit=BOTTLE_PICK_DELTA_Q_LIMIT),
}


@dataclass(frozen=True)
class SafetyResult:
    safe: bool
    reason: str
    details: dict[str, Any]


class SafetySupervisor:
    """Validate a decoded 16D dual-arm action without modifying it."""

    def __init__(self, task_name: str):
        if task_name not in SAFETY_PROFILES:
            supported = ", ".join(sorted(SAFETY_PROFILES))
            raise ValueError(
                f"Unknown safety task '{task_name}'. Supported tasks: {supported}"
            )

        self.task_name = task_name
        self.delta_q_limit = np.asarray(
            SAFETY_PROFILES[task_name].delta_q_limit, dtype=np.float64
        ).copy()
        if self.delta_q_limit.shape != (14,):
            raise ValueError(
                f"Safety profile '{task_name}' must contain 14 joint limits"
            )

    @staticmethod
    def parse_action(action: Any) -> np.ndarray:
        """Return q14 as [left q7, right q7] from a decoded 16D action."""
        try:
            action_array = np.asarray(action, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Action must contain numeric values") from exc

        if action_array.shape != (16,):
            raise ValueError(
                f"Decoded both-arm action must have shape (16,), got {action_array.shape}"
            )
        if not np.isfinite(action_array).all():
            raise ValueError("Decoded action contains NaN or Inf")

        left_q = action_array[0:7]
        right_q = action_array[8:15]
        return np.concatenate((left_q, right_q))

    def check(self, action: Any, q_ref: Any) -> SafetyResult:
        """Check per-joint command continuity relative to the supplied q14."""
        try:
            q_cmd = self.parse_action(action)
            q_ref_array = np.asarray(q_ref, dtype=np.float64)
            if q_ref_array.shape != (14,):
                raise ValueError(
                    f"Reference arm state must have shape (14,), got {q_ref_array.shape}"
                )
            if not np.isfinite(q_ref_array).all():
                raise ValueError("Reference arm state contains NaN or Inf")
        except (TypeError, ValueError) as exc:
            return SafetyResult(
                safe=False,
                reason="invalid_safety_input",
                details={"error": str(exc)},
            )

        delta_q = np.abs(q_cmd - q_ref_array)
        violating_indices = np.flatnonzero(delta_q > self.delta_q_limit).tolist()
        details = {
            "safety_task": self.task_name,
            "violating_joint_indices": violating_indices,
            "violating_joint_names": [ARM_JOINT_NAMES[i] for i in violating_indices],
            "q_ref": q_ref_array.copy(),
            "q_cmd": q_cmd.copy(),
            "delta_q": delta_q.copy(),
            "delta_q_limit": self.delta_q_limit.copy(),
        }

        if violating_indices:
            return SafetyResult(
                safe=False,
                reason="joint_command_continuity_limit_exceeded",
                details=details,
            )

        return SafetyResult(safe=True, reason="ok", details=details)
