"""CLI entry for checkpoint simulation evaluation with YOLO + robot state trigger."""

from __future__ import annotations

import argparse
from pathlib import Path

from kuavo_deploy.config import load_kuavo_config
from kuavo_deploy.src.eval.sim_auto_test_yolo_robot_state import evaluate
from kuavo_deploy.src.scripts.script_auto_test import ArmMove, log_robot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kuavo simulation auto-test with online YOLO robot-state PAUSE trigger"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=["auto_test_yolo_robot_state"],
        default="auto_test_yolo_robot_state",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class YoloRobotStateArmMove(ArmMove):
    def auto_test_yolo_robot_state(self) -> None:
        evaluate(self.config)


def main() -> None:
    args = parse_args()
    config = load_kuavo_config(args.config)
    if args.dry_run:
        log_robot.info("Config: %s", args.config)
        log_robot.info("Checkpoint: %s", config.inference.pretrained_path)
        log_robot.info("YOLO trigger enabled: %s", config.inference.failure_trigger_enabled)
        return
    arm = YoloRobotStateArmMove(config)
    arm.auto_test_yolo_robot_state()


if __name__ == "__main__":
    main()
