"""CLI for simulation evaluation with V2 stage-1 and existing Qwen stage-2."""

from __future__ import annotations

import argparse
from pathlib import Path

from kuavo_deploy.config import load_kuavo_config
from kuavo_deploy.src.eval.sim_auto_test_visual_critic import (
    VisualCriticSimulationConfig,
    kuavo_eval_autotest_visual_critic,
)
from kuavo_deploy.src.eval.sim_auto_test_vlm_agentic import VLMVerifierConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kuavo simulation with V2 visual critic")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--critic-model", default="microsoft/Florence-2-base")
    parser.add_argument("--critic-device", default="cuda:0")
    parser.add_argument("--critic-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--critic-frequency-hz", type=float, default=5.0)
    parser.add_argument("--critic-max-result-age-s", type=float, default=1.5)
    parser.add_argument("--critic-stall-confirm-count", type=int, default=3)
    parser.add_argument("--critic-camera-key", default="observation.images.head_cam_h")
    parser.add_argument("--critic-max-new-tokens", type=int, default=96)
    parser.add_argument("--critic-reset-timeout-s", type=float, default=2.0)
    parser.add_argument("--stage2-vlm-model-path")
    parser.add_argument("--stage2-python-path", default=VLMVerifierConfig.python_path)
    parser.add_argument("--stage2-verifier-script", default=VLMVerifierConfig.verifier_script)
    parser.add_argument("--stage2-max-retries-per-episode", type=int, default=1)
    parser.add_argument("--disable-stage2-verifier", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_kuavo_config(args.config)
    critic = VisualCriticSimulationConfig(
        model_name_or_path=args.critic_model,
        device=args.critic_device,
        dtype=args.critic_dtype,
        frequency_hz=args.critic_frequency_hz,
        max_result_age_s=args.critic_max_result_age_s,
        stall_confirm_count=args.critic_stall_confirm_count,
        camera_key=args.critic_camera_key,
        max_new_tokens=args.critic_max_new_tokens,
        reset_timeout_s=args.critic_reset_timeout_s,
    )
    verifier = VLMVerifierConfig(
        enabled=not args.disable_stage2_verifier,
        model_path=args.stage2_vlm_model_path,
        python_path=args.stage2_python_path,
        verifier_script=args.stage2_verifier_script,
        max_retries_per_episode=args.stage2_max_retries_per_episode,
    )
    if args.dry_run:
        print(critic)
        print(verifier)
        return
    kuavo_eval_autotest_visual_critic(config, critic, verifier)


if __name__ == "__main__":
    main()
