"""Test 14-joint SafetySupervisor thresholds offline or send one guarded target."""

import argparse
import sys

import numpy as np

from kuavo_deploy.safety.safety_supervisor import ARM_JOINT_NAMES, SafetySupervisor


def make_action(q_cmd: np.ndarray) -> np.ndarray:
    """Pack 14 arm joints into the supervisor's decoded 16D action format."""
    return np.concatenate((q_cmd[:7], [0.0], q_cmd[7:], [0.0]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-test per-joint action continuity limits. No robot or ROS "
            "commands are sent."
        )
    )
    parser.add_argument(
        "--task",
        choices=("apple_pick", "bottle_pick"),
        default="apple_pick",
        help="Safety profile to test (default: apple_pick).",
    )
    parser.add_argument(
        "--reference",
        type=float,
        nargs=14,
        metavar="Q",
        default=None,
        help="Offline reference joint positions in radians (default: all zero).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs=14,
        metavar="LIMIT",
        help=(
            "Override the instance delta_q_limit attribute with 14 positive "
            "radian limits, in left-joints then right-joints order."
        ),
    )
    parser.add_argument(
        "--joint-delta",
        type=float,
        nargs=14,
        metavar="DELTA",
        help=(
            "Also check one custom 14-joint displacement from --reference "
            "(radians), in left-joints then right-joints order."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.001,
        help="Amount beyond a threshold used by the rejection test (default: 0.001 rad).",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Enable one real-robot joint command; requires --config and --joint-delta.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Real Kuavo deploy YAML used to initialize Kuavo-Real.",
    )
    parser.add_argument(
        "--max-step-rad",
        type=float,
        help="Required with --send; rejects any requested joint delta above this magnitude.",
    )
    return parser.parse_args()


def send_one_hardware_target(
    args: argparse.Namespace,
    supervisor: SafetySupervisor,
    delta: np.ndarray,
) -> int:
    from kuavo_deploy.config import load_kuavo_config

    config_path = args.config
    config = load_kuavo_config(config_path)
    if config.env.inference_env != "real":
        print("ERROR: --send requires env.inference_env=real.", file=sys.stderr)
        return 2
    if config.env.which_arm != "both" or not config.env.only_arm:
        print(
            "ERROR: --send requires which_arm=both and only_arm=true; "
            "the supervisor expects a dual-arm joint action.",
            file=sys.stderr,
        )
        return 2
    if config.env.control_mode != "joint":
        print("ERROR: --send requires env.control_mode=joint.", file=sys.stderr)
        return 2

    print("REAL ROBOT MODE: this command can move both arms.")
    print(f"Config: {config_path}")
    print(f"Safety profile: {args.task}")
    print(f"Maximum requested delta: {args.max_step_rad:.4f} rad per joint")
    if input("To initialize the real Kuavo environment, type ARM TEST ENABLED: ").strip() != "ARM TEST ENABLED":
        print("Confirmation did not match; no environment was initialized.")
        return 2

    import gymnasium as gym
    import kuavo_deploy.kuavo_env  # noqa: F401  # Register Kuavo-Real.
    import rospy
    from std_msgs.msg import Bool
    from threading import Event

    rospy.init_node("kuavo_safety_threshold_test", anonymous=True, disable_signals=True)
    stop_requested = Event()
    stop_subscriber = rospy.Subscriber(
        "/kuavo/stop_state",
        Bool,
        lambda message: stop_requested.set() if message.data else None,
        queue_size=1,
    )
    env = None
    try:
        env = gym.make(
            config.env.env_name,
            max_episode_steps=config.inference.max_episode_steps,
            config=config,
        )
        robot_env = env.unwrapped
        robot_env._enter_external_control_mode()

        q_ref = np.asarray(
            robot_env.robot_state.arm_joint_state().position,
            dtype=np.float64,
        )
        if q_ref.shape != (14,) or not np.isfinite(q_ref).all():
            raise RuntimeError(
                f"Live arm state must contain 14 finite values, got shape {q_ref.shape}"
            )

        q_cmd = q_ref + delta
        action_low = np.asarray(robot_env.action_space.low, dtype=np.float64)
        action_high = np.asarray(robot_env.action_space.high, dtype=np.float64)
        if action_low.shape != (16,) or action_high.shape != (16,):
            raise RuntimeError(
                "Expected a 16D both-arm joint action space; "
                f"got low={action_low.shape}, high={action_high.shape}"
            )
        joint_low = np.concatenate((action_low[:7], action_low[8:15]))
        joint_high = np.concatenate((action_high[:7], action_high[8:15]))
        if np.any(q_cmd < joint_low) or np.any(q_cmd > joint_high):
            print("ERROR: target exceeds configured joint position limits; no command sent.")
            return 1

        safety_result = supervisor.check(make_action(q_cmd), q_ref)
        print("\nLive joint target preview (radians):")
        print(f"{'Joint':<18} {'measured':>11} {'delta':>11} {'target':>11} {'limit':>11}")
        print("-" * 68)
        for index, joint_name in enumerate(ARM_JOINT_NAMES):
            print(
                f"{joint_name:<18} {q_ref[index]:>11.4f} {delta[index]:>11.4f} "
                f"{q_cmd[index]:>11.4f} {supervisor.delta_q_limit[index]:>11.4f}"
            )

        if not safety_result.safe:
            print(
                "\nSAFETY REJECTED: "
                f"{safety_result.reason}; "
                f"violating joints={safety_result.details.get('violating_joint_names', [])}"
            )
            print("No arm command was sent.")
            return 1
        if stop_requested.is_set():
            print("\nStop signal already active; no arm command was sent.")
            return 1

        print(
            "\nSafety check passed. Confirm the robot workspace is clear and "
            "you can reach the physical E-stop."
        )
        if input("To send this one joint target, type MOVE ROBOT ONCE: ").strip() != "MOVE ROBOT ONCE":
            print("Confirmation did not match; no arm command was sent.")
            return 2
        if stop_requested.is_set():
            print("Stop signal received before dispatch; no arm command was sent.")
            return 1

        robot_env.safe_control_arm(q_cmd)
        print("One arm joint target was sent. No further targets will be issued by this script.")
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Use the physical E-stop if the robot is still moving.")
        return 130
    finally:
        stop_subscriber.unregister()
        if env is not None:
            env.close()


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.epsilon) or args.epsilon <= 0:
        print("ERROR: --epsilon must be finite and greater than zero.", file=sys.stderr)
        return 2

    supervisor = SafetySupervisor(task_name=args.task)
    if args.thresholds is not None:
        thresholds = np.asarray(args.thresholds, dtype=np.float64)
        if not np.isfinite(thresholds).all() or np.any(thresholds <= 0):
            print("ERROR: --thresholds must contain 14 positive finite values.", file=sys.stderr)
            return 2
        supervisor.delta_q_limit = thresholds.copy()

    limits = supervisor.delta_q_limit
    if limits.shape != (14,) or not np.isfinite(limits).all() or np.any(limits <= 0):
        print("ERROR: delta_q_limit must contain 14 positive finite values.", file=sys.stderr)
        return 2

    if args.send:
        if args.config is None or args.joint_delta is None:
            print("ERROR: --send requires both --config and --joint-delta.", file=sys.stderr)
            return 2
        if args.reference is not None:
            print("ERROR: --reference is offline-only; live measured joints are used with --send.", file=sys.stderr)
            return 2
        if args.thresholds is not None:
            print("ERROR: --thresholds cannot override safety limits in hardware mode.", file=sys.stderr)
            return 2
        if args.max_step_rad is None or not np.isfinite(args.max_step_rad) or args.max_step_rad <= 0:
            print("ERROR: --send requires a positive finite --max-step-rad.", file=sys.stderr)
            return 2
        delta = np.asarray(args.joint_delta, dtype=np.float64)
        if not np.isfinite(delta).all():
            print("ERROR: --joint-delta values must be finite.", file=sys.stderr)
            return 2
        if np.any(np.abs(delta) > args.max_step_rad):
            print(
                "ERROR: requested joint delta exceeds --max-step-rad; "
                "no command was sent.",
                file=sys.stderr,
            )
            return 2
        return send_one_hardware_target(args, supervisor, delta)

    if args.config is not None or args.max_step_rad is not None:
        print("ERROR: --config and --max-step-rad are only valid with --send.", file=sys.stderr)
        return 2

    reference = np.asarray(
        args.reference if args.reference is not None else [0.0] * 14,
        dtype=np.float64,
    )
    if not np.isfinite(reference).all():
        print("ERROR: --reference values must be finite.", file=sys.stderr)
        return 2

    print(f"Safety profile: {args.task}")
    print("Mode: offline only; no robot, ROS, or env.step() calls")
    print("Threshold source: SafetySupervisor.delta_q_limit (radians)")
    print("Joint order: left L1-L7, then right R1-R7")
    print()
    print(f"{'Joint':<18} {'delta_q_limit':>14} {'inside test':>14} {'outside test':>15}")
    print("-" * 65)

    failures = 0
    for index, (joint_name, limit) in enumerate(zip(ARM_JOINT_NAMES, limits)):
        # Stay a small amount inside the boundary to avoid floating-point
        # rounding at q_ref + limit; the existing unit test covers equality.
        margin = min(args.epsilon * 0.1, limit * 0.1)
        inside_delta = limit - margin
        outside_delta = limit + args.epsilon

        inside_q = reference.copy()
        inside_q[index] += inside_delta
        inside_result = supervisor.check(make_action(inside_q), reference)

        outside_q = reference.copy()
        outside_q[index] += outside_delta
        outside_result = supervisor.check(make_action(outside_q), reference)

        inside_ok = inside_result.safe
        outside_ok = (
            not outside_result.safe
            and index in outside_result.details.get("violating_joint_indices", [])
        )
        failures += int(not inside_ok) + int(not outside_ok)

        print(
            f"{joint_name:<18} {limit:>14.4f} "
            f"{'PASS' if inside_ok else 'FAIL':>14} "
            f"{'REJECT' if outside_ok else 'FAIL':>15}"
        )

    if args.joint_delta is not None:
        delta = np.asarray(args.joint_delta, dtype=np.float64)
        if not np.isfinite(delta).all():
            print("ERROR: --joint-delta values must be finite.", file=sys.stderr)
            return 2
        custom_result = supervisor.check(make_action(reference + delta), reference)
        if custom_result.safe:
            print("\nCustom displacement: PASS (all 14 joint deltas within threshold)")
        else:
            details = custom_result.details
            names = details.get("violating_joint_names", [])
            print(
                "\nCustom displacement: REJECT "
                f"(reason={custom_result.reason}, violating_joints={names})"
            )

    print()
    if failures:
        print(f"RESULT: FAIL ({failures} threshold checks did not match expectations)")
        return 1
    print("RESULT: PASS (all 14 joints passed inside and rejected outside their limits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())