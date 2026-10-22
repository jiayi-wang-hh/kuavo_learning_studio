#!/usr/bin/env python3
"""Mock policy server for testing SafetySupervisor through normal eval.py flow.

The server speaks the repository's PolicyClient protocol over ZeroMQ. It does
not connect to ROS or MuJoCo directly. Instead, eval.py sends it the current
MuJoCo observation and receives a decoded 16D dual-arm action:

    [left_q7, left_gripper, right_q7, right_gripper]

For normal requests, one selected arm joint follows a sine target using a
per-request change below the selected SafetySupervisor profile limit. At
``--fail-after``, the returned command deliberately exceeds that limit. This
lets eval.py exercise its real path:

    PolicyClient -> postprocessor -> SafetySupervisor
      PASS -> env.step(action)
      FAIL -> reject action -> /kuavo/stop_state=True -> terminate rollout

Port 5555 is ZeroMQ TCP (tcp://localhost:5555), not an HTTP endpoint.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kuavo_deploy.kuavo_service.client import BaseInferenceServer
from kuavo_deploy.safety.safety_supervisor import (
    ARM_JOINT_NAMES,
    SAFETY_PROFILES,
    SafetySupervisor,
)


class MujocoSafetyTestPolicy:
    """Generate state-relative arm commands and one intentional violation."""

    def __init__(
        self,
        safety_task: str,
        joint_index: int,
        safe_fraction: float,
        amplitude: float,
        period_steps: int,
        fail_after: int,
        unsafe_excess: float,
        log_every: int,
    ) -> None:
        self.supervisor = SafetySupervisor(task_name=safety_task)
        self.safety_task = safety_task
        self.joint_index = joint_index
        self.safe_fraction = safe_fraction
        self.amplitude = amplitude
        self.period_steps = period_steps
        self.fail_after = fail_after
        self.unsafe_excess = unsafe_excess
        self.log_every = log_every

        self.request_count = 0
        self.initial_q: np.ndarray | None = None
        self.failure_latched = False

    @staticmethod
    def _observation_state16(observation: dict[str, Any]) -> np.ndarray:
        if "observation.state" not in observation:
            raise KeyError(
                "Policy request does not contain 'observation.state'; got keys "
                f"{sorted(observation)}"
            )

        state = observation["observation.state"]
        if torch.is_tensor(state):
            state = state.detach().cpu().numpy()
        state = np.asarray(state, dtype=np.float64)
        if state.shape == (1, 16):
            state = state[0]
        if state.shape != (16,):
            raise ValueError(
                "Expected observation.state shape (1, 16) or (16,), "
                f"got {state.shape}"
            )
        if not np.isfinite(state).all():
            raise ValueError("observation.state contains NaN or Inf")
        return state.copy()

    @staticmethod
    def _replace_q14(state16: np.ndarray, q14: np.ndarray) -> np.ndarray:
        """Preserve both grippers and replace only [left q7, right q7]."""
        action = state16.copy()
        action[0:7] = q14[0:7]
        action[8:15] = q14[7:14]
        return action.astype(np.float32)

    def reset(self) -> dict[str, Any]:
        self.request_count = 0
        self.initial_q = None
        self.failure_latched = False
        print("[RESET] mock policy state cleared", flush=True)
        return {"status": "ok"}

    def _make_safe_action(
        self, state16: np.ndarray, q_measured: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        assert self.initial_q is not None
        joint = self.joint_index
        limit = float(self.supervisor.delta_q_limit[joint])
        max_safe_step = self.safe_fraction * limit

        phase = 2.0 * math.pi * self.request_count / self.period_steps
        target = float(self.initial_q[joint] + self.amplitude * math.sin(phase))
        error = target - float(q_measured[joint])
        delta = float(np.clip(error, -max_safe_step, max_safe_step))

        q_cmd = q_measured.copy()
        q_cmd[joint] += delta
        action = self._replace_q14(state16, q_cmd)

        result = self.supervisor.check(action=action, q_ref=q_measured)
        if not result.safe:
            raise RuntimeError(
                "Mock server generated an unexpectedly unsafe normal action: "
                f"{result.reason}, details={result.details}"
            )
        return action, target, delta

    def _make_unsafe_action(
        self, state16: np.ndarray, q_measured: np.ndarray
    ) -> tuple[np.ndarray, float]:
        joint = self.joint_index
        limit = float(self.supervisor.delta_q_limit[joint])
        unsafe_delta = limit + self.unsafe_excess

        # This command must be rejected by eval.py and must not reach env.step().
        q_cmd = q_measured.copy()
        q_cmd[joint] += unsafe_delta
        action = self._replace_q14(state16, q_cmd)

        result = self.supervisor.check(action=action, q_ref=q_measured)
        if result.safe:
            raise RuntimeError("Failed to construct the requested unsafe action")
        return action, unsafe_delta

    def select_action(self, observation: dict[str, Any]) -> np.ndarray:
        state16 = self._observation_state16(observation)
        q_measured = self.supervisor.parse_action(state16)
        if self.initial_q is None:
            self.initial_q = q_measured.copy()

        self.request_count += 1
        if self.fail_after > 0 and self.request_count >= self.fail_after:
            self.failure_latched = True

        joint = self.joint_index
        joint_name = ARM_JOINT_NAMES[joint]
        limit = float(self.supervisor.delta_q_limit[joint])

        if self.failure_latched:
            action, delta = self._make_unsafe_action(state16, q_measured)
            print(
                f"[{self.request_count:04d}] INJECT FAIL "
                f"joint={joint}:{joint_name} measured={q_measured[joint]:+.5f} "
                f"delta={delta:+.5f} limit={limit:.5f}",
                flush=True,
            )
            return action

        action, target, delta = self._make_safe_action(state16, q_measured)
        if self.request_count == 1 or self.request_count % self.log_every == 0:
            print(
                f"[{self.request_count:04d}] SAFE "
                f"joint={joint}:{joint_name} measured={q_measured[joint]:+.5f} "
                f"target={target:+.5f} delta={delta:+.5f} limit={limit:.5f}",
                flush=True,
            )
        return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serve synthetic 16D actions on the same ZeroMQ endpoint used by "
            "LingBot PolicyClient."
        )
    )
    parser.add_argument("--host", default="*", help="ZeroMQ bind host.")
    parser.add_argument("--port", type=int, default=5555, help="ZeroMQ bind port.")
    parser.add_argument(
        "--safety-task",
        choices=sorted(SAFETY_PROFILES),
        default="apple_pick",
        help="Must match inference.safety_task in the eval config.",
    )
    parser.add_argument(
        "--joint-index",
        type=int,
        choices=range(14),
        default=0,
        metavar="0..13",
        help="q14 joint to move and eventually violate.",
    )
    parser.add_argument(
        "--safe-fraction",
        type=float,
        default=0.5,
        help="Maximum normal command delta as a fraction of its profile limit.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.12,
        help="Sine target amplitude in radians around the initial measured pose.",
    )
    parser.add_argument(
        "--period-steps",
        type=int,
        default=40,
        help="Number of inference requests per sine cycle.",
    )
    parser.add_argument(
        "--fail-after",
        type=int,
        default=20,
        help="Inject unsafe action on this 1-based request; 0 disables failure.",
    )
    parser.add_argument(
        "--unsafe-excess",
        type=float,
        default=0.01,
        help="Radians above the profile limit for the injected failure.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print every Nth safe action.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.safe_fraction < 1.0:
        raise ValueError("--safe-fraction must be strictly between 0 and 1")
    if args.amplitude <= 0.0:
        raise ValueError("--amplitude must be positive")
    if args.period_steps < 2:
        raise ValueError("--period-steps must be at least 2")
    if args.fail_after < 0:
        raise ValueError("--fail-after must be >= 0")
    if args.unsafe_excess <= 0.0:
        raise ValueError("--unsafe-excess must be positive")
    if args.log_every < 1:
        raise ValueError("--log-every must be at least 1")


def raise_keyboard_interrupt() -> None:
    raise KeyboardInterrupt


def main() -> int:
    args = parse_args()
    validate_args(args)

    policy = MujocoSafetyTestPolicy(
        safety_task=args.safety_task,
        joint_index=args.joint_index,
        safe_fraction=args.safe_fraction,
        amplitude=args.amplitude,
        period_steps=args.period_steps,
        fail_after=args.fail_after,
        unsafe_excess=args.unsafe_excess,
        log_every=args.log_every,
    )
    server = BaseInferenceServer(host=args.host, port=args.port)
    server.register_endpoint("select_action", policy.select_action)
    server.register_endpoint("reset", policy.reset, requires_input=False)

    joint_name = ARM_JOINT_NAMES[args.joint_index]
    limit = policy.supervisor.delta_q_limit[args.joint_index]
    print("=== MuJoCo SafetySupervisor mock policy server ===")
    print(f"endpoint: tcp://{args.host}:{args.port} (ZeroMQ, not HTTP)")
    print(f"profile: {args.safety_task}")
    print(f"controlled_joint: {args.joint_index} ({joint_name})")
    print(f"profile_limit_rad: {limit:.5f}")
    print(f"normal_max_delta_rad: {args.safe_fraction * limit:.5f}")
    print(f"fail_after_request: {args.fail_after or 'disabled'}")
    print("Press Ctrl+C to stop the mock server.\n", flush=True)

    signal.signal(signal.SIGTERM, lambda _signum, _frame: raise_keyboard_interrupt())
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nMock policy server stopped.")
    finally:
        server.socket.close(linger=0)
        server.context.term()
    return 0


if __name__ == "__main__":
    sys.exit(main())
