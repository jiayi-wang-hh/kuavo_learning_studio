"""CLI entry for closed-loop simulation evaluation with a VLM pause trigger."""

from __future__ import annotations

import argparse
from pathlib import Path

from kuavo_deploy.config import load_kuavo_config
from kuavo_deploy.src.eval.sim_auto_test_vlm_agentic import (
    VLMTriggerConfig,
    VLMVerifierConfig,
    kuavo_eval_autotest_vlm_agentic,
)
from kuavo_deploy.src.scripts.script_auto_test import ArmMove, log_robot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kuavo simulation auto-test with asynchronous VLM PAUSE trigger"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vlm-mode", default="qwen25_vl_7b")
    parser.add_argument("--vlm-model-path")
    parser.add_argument(
        "--vlm-camera-key", default="observation.images.head_cam_h"
    )
    parser.add_argument("--vlm-window-seconds", type=float, default=3.0)
    parser.add_argument("--vlm-fps", type=float, default=4.0)
    parser.add_argument("--vlm-check-interval-steps", type=int, default=10)
    parser.add_argument("--vlm-max-new-tokens", type=int, default=96)
    parser.add_argument("--vlm-temperature", type=float, default=0.0)
    parser.add_argument(
        "--vlm-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--vlm-attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument(
        "--vlm-device-map",
        default="auto",
        help="Transformers device_map (default: auto; no fixed GPU ordinal)",
    )
    parser.add_argument("--vlm-pause-confirmations", type=int, default=1)
    parser.add_argument("--disable-stage2-verifier", action="store_true")
    parser.add_argument("--stage2-vlm-mode", default="qwen35_9b")
    parser.add_argument("--stage2-vlm-model-path")
    parser.add_argument(
        "--stage2-vlm-device-map",
        default="auto",
        help="Transformers device_map for stage 2 (default: auto)",
    )
    parser.add_argument("--stage2-vlm-max-new-tokens", type=int, default=192)
    parser.add_argument("--stage2-max-retries-per-episode", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task",
                        choices=["auto_test_vlm_agentic"],
                        default="auto_test_vlm_agentic",
    ),
    return parser.parse_args()


class AgenticArmMove(ArmMove):
    def __init__(self, config, trigger_config: VLMTriggerConfig):
        super().__init__(config)
        self.trigger_config = trigger_config

    def auto_test_vlm_agentic(self, verifier_config: VLMVerifierConfig) -> None:
        kuavo_eval_autotest_vlm_agentic(
            self.config, self.trigger_config, verifier_config
        )


def main() -> None:
    args = parse_args()
    config = load_kuavo_config(args.config)
    trigger_config = VLMTriggerConfig(
        mode=args.vlm_mode,
        model_path=args.vlm_model_path,
        camera_key=args.vlm_camera_key,
        window_seconds=args.vlm_window_seconds,
        sample_fps=args.vlm_fps,
        check_interval_steps=args.vlm_check_interval_steps,
        max_new_tokens=args.vlm_max_new_tokens,
        temperature=args.vlm_temperature,
        dtype=args.vlm_dtype,
        attn_implementation=args.vlm_attn_implementation,
        device_map=args.vlm_device_map,
        pause_confirmations=args.vlm_pause_confirmations,
    )
    verifier_config = VLMVerifierConfig(
        enabled=not args.disable_stage2_verifier,
        mode=args.stage2_vlm_mode,
        model_path=args.stage2_vlm_model_path,
        device_map=args.stage2_vlm_device_map,
        max_new_tokens=args.stage2_vlm_max_new_tokens,
        max_retries_per_episode=args.stage2_max_retries_per_episode,
    )
    arm = AgenticArmMove(config, trigger_config)
    if args.dry_run:
        log_robot.info("Config: %s", args.config)
        log_robot.info("VLM trigger config: %s", trigger_config)
        log_robot.info("Stage-2 verifier config: %s", verifier_config)
        return
    arm.auto_test_vlm_agentic(verifier_config)


if __name__ == "__main__":
    main()
