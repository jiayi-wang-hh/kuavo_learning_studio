"""V2 single-frame visual critic wired into the existing agentic simulation."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import imageio

from kuavo_deploy.src.eval import sim_auto_test_vlm_agentic as agentic
from kuavo_deploy.src.eval.failure_trigger.visual_critic_trigger import (
    FlorenceVisualCriticModel,
    TriggerDecision,
    VisualCriticConfig,
    VisualCriticTrigger,
)


@dataclass(frozen=True)
class VisualCriticSimulationConfig(VisualCriticConfig):
    camera_key: str = "observation.images.head_cam_h"
    save_trigger_context: bool = True
    pre_trigger_seconds: float = 1.5
    post_trigger_seconds: float = 0.0


class VisualCriticAgentAdapter:
    """Makes the V2 trigger usable by the unchanged stage-2/episode runner."""

    def __init__(self, config, task_prompt: str, ros_rate: float, output_directory: Path):
        self.config = config
        self.task_prompt = task_prompt
        self.ros_rate = float(ros_rate)
        self._period_steps = max(1, round(self.ros_rate / config.frequency_hz))
        self._latest_frame = None
        self._episode = 0
        self._decision = TriggerDecision(config)
        self._backend = FlorenceVisualCriticModel(config)
        self._trigger = VisualCriticTrigger(config, self._backend)
        self._context = deque(maxlen=max(2, round(config.pre_trigger_seconds * self.ros_rate)))
        self._clip_directory = output_directory / "v2_visual_critic_context"
        self._clip_directory.mkdir(parents=True, exist_ok=True)
        agentic.log_robot.info(
            "V2 critic initialized camera=%s frequency=%.2fHz stale_after=%.2fs stall=%d",
            config.camera_key, config.frequency_hz, config.max_result_age_s,
            config.stall_confirm_count,
        )

    def append_observation(self, observation) -> None:
        frame = agentic.observation_to_rgb(observation, self.config.camera_key)
        self._latest_frame = frame
        self._context.append(frame.copy())

    def ready(self, step: int) -> bool:
        # Submission is allowed while inference runs: the trigger overwrites its
        # one pending slot, which is the core anti-backlog behavior.
        return self._latest_frame is not None and step % self._period_steps == 0

    def submit(self, episode: int, step: int) -> None:
        self._episode = episode
        self._trigger.submit(
            self._latest_frame, self.task_prompt, step, time.perf_counter()
        )

    def poll(self):
        result = self._trigger.get_latest_result()
        if result is None:
            return None
        final_trigger, trigger_type = self._decision.evaluate(result)
        clip_path = ""
        if final_trigger and self.config.save_trigger_context:
            path = self._clip_directory / (
                f"episode_{self._episode:03d}_step_{result.source_step:05d}.mp4"
            )
            imageio.mimsave(
                str(path), list(self._context), fps=self.ros_rate, codec="libx264"
            )
            clip_path = str(path)
        progress = "None" if result.progress_score is None else f"{result.progress_score:.3f}"
        agentic.log_robot.info(
            "[V2_CRITIC] source_step=%d state=%s confidence=%s progress=%s "
            "inference=%.1fms age=%.1fms stall_count=%d/%d stale=%s trigger=%s type=%s",
            result.source_step, result.state, result.confidence, progress,
            result.inference_ms, result.result_age_ms or 0.0,
            self._decision.stall_counter, self.config.stall_confirm_count,
            result.stale_discarded, final_trigger, trigger_type or "NONE",
        )
        # Legacy-shaped boundary object keeps stage 2 and recovery untouched.
        return agentic.TriggerResult(
            episode=self._episode,
            source_step=result.source_step,
            start_s=result.source_step / self.ros_rate,
            end_s=result.source_step / self.ros_rate,
            completed_at=time.perf_counter(),
            raw_decision=result.state,
            final_decision="PAUSE" if final_trigger else "CONTINUE",
            confidence=result.confidence,
            evidence=result.reason,
            guard_reason=("STALE_RESULT_DISCARDED" if result.stale_discarded else (trigger_type or "NONE")),
            schema_valid=result.state != "UNKNOWN",
            raw_output="",
            inference_seconds=result.inference_ms / 1000.0,
            clip_path=clip_path,
            current_phase=result.state,
            expected_effect=f"Visible progress toward subtask: {self.task_prompt}",
            observed_effect=result.reason,
        )

    def confirmed_pause(self, result) -> bool:
        return result.final_decision == "PAUSE"

    def reset_episode(self) -> None:
        self._trigger.reset()
        self._decision.reset()
        self._context.clear()
        self._latest_frame = None

    def close(self) -> None:
        self._trigger.shutdown()


def kuavo_eval_autotest_visual_critic(
    config,
    trigger_config: VisualCriticSimulationConfig,
    verifier_config: agentic.VLMVerifierConfig | None = None,
) -> None:
    """Reuse the stable episode/stage-2 path with a scoped V2 agent factory."""
    original = agentic.VLMTriggerAgent
    agentic.VLMTriggerAgent = VisualCriticAgentAdapter
    try:
        agentic.kuavo_eval_autotest_vlm_agentic(config, trigger_config, verifier_config)
    finally:
        agentic.VLMTriggerAgent = original
