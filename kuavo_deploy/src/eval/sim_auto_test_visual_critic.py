"""V2 single-frame visual critic wired into the existing agentic simulation."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
from pathlib import Path

import imageio

from kuavo_deploy.src.eval import sim_auto_test_vlm_agentic as agentic
from kuavo_deploy.src.eval.failure_trigger.visual_critic_trigger import (
    FlorenceVisualCriticModel,
    TriggerDecision,
    VisualCriticConfig,
    VisualCriticTrigger,
    observation_camera_timestamp,
)


@dataclass(frozen=True)
class VisualCriticSimulationConfig(VisualCriticConfig):
    camera_key: str = "observation.images.head_cam_h"
    save_trigger_context: bool = True
    pre_trigger_seconds: float = 1.5
    post_trigger_seconds: float = 0.0


@dataclass(frozen=True)
class VisualCriticStage2Result(agentic.TriggerResult):
    critic_source_timestamp: float
    pause_step: int
    context_start_step: int
    context_end_step: int


class VisualCriticAgentAdapter:
    """Makes the V2 trigger usable by the unchanged stage-2/episode runner."""

    def __init__(self, config, task_prompt: str, ros_rate: float, output_directory: Path):
        self.config = config
        self.task_prompt = task_prompt
        self.ros_rate = float(ros_rate)
        self._period_steps = max(1, round(self.ros_rate / config.frequency_hz))
        self._latest_frame = None
        self._latest_source_timestamp = None
        self._latest_monotonic_timestamp = None
        self._observation_step = -1
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
        camera_value = observation[self.config.camera_key]
        image_observation = observation
        if isinstance(camera_value, Mapping) and "data" in camera_value:
            image_observation = dict(observation)
            image_observation[self.config.camera_key] = camera_value["data"]
        frame = agentic.observation_to_rgb(image_observation, self.config.camera_key)
        now = time.perf_counter()
        raw_timestamp = observation_camera_timestamp(observation, self.config.camera_key)
        self._latest_source_timestamp = now if raw_timestamp is None else raw_timestamp
        self._latest_monotonic_timestamp = now
        self._observation_step += 1
        self._latest_frame = frame
        self._context.append(
            (self._observation_step, self._latest_source_timestamp, frame.copy())
        )

    def ready(self, step: int) -> bool:
        # Submission is allowed while inference runs: the trigger overwrites its
        # one pending slot, which is the core anti-backlog behavior.
        return self._latest_frame is not None and step % self._period_steps == 0

    def submit(self, episode: int, step: int) -> None:
        self._episode = episode
        submitted_at = time.perf_counter()
        self._trigger.submit(
            self._latest_frame,
            self.task_prompt,
            step,
            (
                submitted_at
                if self._latest_source_timestamp is None
                else self._latest_source_timestamp
            ),
            self._latest_monotonic_timestamp,
        )

    def poll(self):
        result = self._trigger.get_latest_result()
        if result is None:
            return None
        final_trigger, trigger_type = self._decision.evaluate(result)
        clip_path = ""
        pause_step = self._observation_step
        context = list(self._context)
        context_start_step = context[0][0] if context else pause_step
        context_end_step = context[-1][0] if context else pause_step
        if final_trigger and self.config.save_trigger_context:
            path = self._clip_directory / (
                f"episode_{self._episode:03d}_pause_step_{pause_step:05d}.mp4"
            )
            imageio.mimsave(
                str(path), [item[2] for item in context], fps=self.ros_rate, codec="libx264"
            )
            clip_path = str(path)
            metadata = {
                "context_semantics": "recent execution context immediately before formal pause",
                "critic_source_step": result.source_step,
                "critic_source_timestamp": result.source_timestamp,
                "pause_step": pause_step,
                "context_start_step": context_start_step,
                "context_end_step": context_end_step,
            }
            path.with_suffix(".json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        progress = "None" if result.progress_score is None else f"{result.progress_score:.3f}"
        agentic.log_robot.info(
            "[V2_CRITIC] source_step=%d state=%s confidence=%s progress=%s "
            "inference=%.1fms age=%.1fms stall_count=%d/%d stale=%s trigger=%s type=%s",
            result.source_step, result.state, result.confidence, progress,
            result.inference_ms, result.result_age_ms or 0.0,
            self._decision.stall_counter, self.config.stall_confirm_count,
            result.stale_discarded, final_trigger, trigger_type or "NONE",
        )
        if result.state == "SUCCESS":
            agentic.log_robot.info(
                "[V2_CRITIC] SUCCESS informational_only=true subtask_transition=false"
            )
        # Legacy-shaped boundary object keeps stage 2 and recovery untouched.
        return VisualCriticStage2Result(
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
            critic_source_timestamp=result.source_timestamp,
            pause_step=pause_step,
            context_start_step=context_start_step,
            context_end_step=context_end_step,
        )

    def confirmed_pause(self, result) -> bool:
        return result.final_decision == "PAUSE"

    def reset_episode(self) -> None:
        self._trigger.reset()
        self._decision.reset()
        self._context.clear()
        self._latest_frame = None
        self._latest_source_timestamp = None
        self._latest_monotonic_timestamp = None
        self._observation_step = -1

    def close(self) -> None:
        self._trigger.shutdown()


class VisualCriticFailureVerifier(agentic.VLMFailureVerifier):
    """Adds explicit V2 timing/context metadata to the existing Stage-2 prompt."""

    def verify(self, trigger):
        metadata = {
            "critic_source_step": trigger.source_step,
            "critic_source_timestamp": trigger.critic_source_timestamp,
            "pause_step": trigger.pause_step,
            "context_start_step": trigger.context_start_step,
            "context_end_step": trigger.context_end_step,
            "context_semantics": "recent execution context immediately before formal pause",
        }
        enriched = replace(
            trigger,
            evidence=(
                f"{trigger.evidence}\nStage-2 context metadata: "
                f"{json.dumps(metadata, ensure_ascii=False)}"
            ),
        )
        return super().verify(enriched)


def kuavo_eval_autotest_visual_critic(
    config,
    trigger_config: VisualCriticSimulationConfig,
    verifier_config: agentic.VLMVerifierConfig | None = None,
) -> None:
    """Reuse the stable episode/stage-2 path with a scoped V2 agent factory."""
    original = agentic.VLMTriggerAgent
    original_verifier = agentic.VLMFailureVerifier
    agentic.VLMTriggerAgent = VisualCriticAgentAdapter
    agentic.VLMFailureVerifier = VisualCriticFailureVerifier
    try:
        agentic.kuavo_eval_autotest_vlm_agentic(config, trigger_config, verifier_config)
    finally:
        agentic.VLMTriggerAgent = original
        agentic.VLMFailureVerifier = original_verifier
