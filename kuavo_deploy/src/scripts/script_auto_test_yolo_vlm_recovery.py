"""CLI entry for YOLO stage-1 plus Qwen3.5 stage-2 recovery evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

from kuavo_deploy.config import load_kuavo_config
from kuavo_deploy.src.eval.sim_auto_test_vlm_agentic import VLMVerifierConfig
from kuavo_deploy.src.eval.sim_auto_test_yolo_vlm_recovery import evaluate
from kuavo_deploy.src.scripts.script_auto_test import ArmMove, log_robot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=["auto_test_yolo_vlm_recovery"], default="auto_test_yolo_vlm_recovery")
    parser.add_argument("--stage2-python", default=VLMVerifierConfig.python_path)
    parser.add_argument("--stage2-model-path", default=VLMVerifierConfig.model_path)
    parser.add_argument("--stage2-verifier-script", default=VLMVerifierConfig.verifier_script)
    parser.add_argument("--stage2-device-map", default="auto")
    parser.add_argument("--stage2-max-retries-per-episode", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_kuavo_config(args.config)
    verifier = VLMVerifierConfig(python_path=args.stage2_python, model_path=args.stage2_model_path, verifier_script=args.stage2_verifier_script, device_map=args.stage2_device_map, max_retries_per_episode=args.stage2_max_retries_per_episode)
    if args.dry_run:
        log_robot.info("Config: %s", args.config); log_robot.info("Stage-2 verifier: %s", verifier); return
    class Runner(ArmMove):
        def run(self): evaluate(self.config, verifier)
    Runner(config).run()


if __name__ == "__main__":
    main()
