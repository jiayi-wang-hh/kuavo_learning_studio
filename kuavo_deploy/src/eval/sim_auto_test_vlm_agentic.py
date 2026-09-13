"""Closed-loop Kuavo simulation evaluation with an asynchronous VLM pause agent."""

from __future__ import annotations

import csv
import datetime
import gc
import json
import time
import subprocess
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import imageio
import numpy as np
import rospy
import torch
from std_msgs.msg import Bool
from std_srvs.srv import Trigger, TriggerRequest

from kuavo_deploy.kuavo_env.KuavoSimEnv import KuavoSimEnv  # noqa: F401
from kuavo_deploy.src.eval import sim_auto_test as base
from kuavo_deploy.src.scripts import script_auto_test as control
from kuavo_deploy.utils.policy_loader import (
    inject_task_prompt,
    resolve_eval_output_dir,
)
from kuavo_deploy.utils.ros_manager import ROSManager

from vlm_test import test_vlm_video_joy_pause_trigger as pause_trigger
from vlm_test import test_vlm_video_joy_trigger as vlm_common


log_model = base.log_model
log_robot = base.log_robot


# ============================================================
# VLM configuration
# ============================================================


@dataclass(frozen=True)
class VLMTriggerConfig:
    mode: str = "qwen25_vl_7b"
    model_path: str | None = None

    camera_key: str = "observation.images.head_cam_h"

    # Sliding window
    window_seconds: float = 3.0
    sample_fps: float = 4.0

    # Run one VLM check every N robot steps
    check_interval_steps: int = 10

    # Generation settings
    max_new_tokens: int = 96
    temperature: float = 0.0

    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    # Let Transformers select from the devices visible to this process.
    device_map: str = "auto"

    # Number of consecutive PAUSE decisions required
    pause_confirmations: int = 1


@dataclass(frozen=True)
class VLMVerifierConfig:
    """Stage-2 verifier executed in a separate Qwen3.5 environment."""

    enabled: bool = True

    mode: str = "qwen35_9b"

    model_path: str = (
        "/media/data/jiayi/hf_model/Qwen3.5-9B"
    )

    # Python executable belonging to the separate qwen35 env
    python_path: str = (
        "/home/kuavo/miniforge3/envs/qwen35/bin/python"
    )

    # External verifier script
    verifier_script: str = (
        "/home/kuavo/jiayi/kuavo_learning_studio/"
        "vlm_test/qwen35_failure_verifier.py"
    )

    max_new_tokens: int = 192
    temperature: float = 0.0
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    # The external verifier resolves its visible devices independently.
    device_map: str = "auto"

    # Maximum time to wait for stage-2
    timeout_seconds: float = 120.0

    max_retries_per_episode: int = 1


# ============================================================
# VLM request / result structures
# ============================================================


@dataclass(frozen=True)
class TriggerRequestData:
    episode: int
    step: int

    start_s: float
    end_s: float

    submitted_at: float

    frames: list[np.ndarray]


@dataclass(frozen=True)
class TriggerResult:
    episode: int
    source_step: int

    start_s: float
    end_s: float

    completed_at: float

    raw_decision: str
    final_decision: str

    confidence: str
    evidence: str
    guard_reason: str

    schema_valid: bool
    raw_output: str

    inference_seconds: float
    clip_path: str

    ####phase parameters####
    current_phase: str
    expected_effect: str
    observed_effect: str


@dataclass(frozen=True)
class VerificationResult:
    failure_type: str
    recovery_action: str
    confidence: str
    evidence: str
    raw_output: str
    inference_seconds: float



# ============================================================
# Observation utilities
# ============================================================


def observation_to_rgb(
    observation,
    camera_key: str,
) -> np.ndarray:
    """
    Convert a Kuavo image observation tensor to uint8 RGB image.

    Expected input shape:
        [1, C, H, W]

    Output:
        [H, W, C]
    """

    if camera_key not in observation:
        raise KeyError(
            f"Camera key '{camera_key}' is not present in observation. "
            f"Available keys: {list(observation.keys())}"
        )

    image = (
        observation[camera_key]
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    image = (
        image.transpose(1, 2, 0) * 255
    ).clip(0, 255).astype(np.uint8)

    if image.shape[-1] == 1:
        return image.squeeze(-1)

    return image


# ============================================================
# Asynchronous VLM trigger agent
# ============================================================


class VLMTriggerAgent:
    """
    Runs one local VLM inference request at a time.

    The robot policy loop does not wait for the VLM.
    While the VLM processes the previous window, the robot
    continues executing the policy.

    When the inference finishes, poll() returns the result.
    """

    VIEW_MAP = {
        "observation.images.head_cam_h": "head",
        "observation.images.wrist_cam_l": "left",
        "observation.images.wrist_cam_r": "right",
    }

    def __init__(
        self,
        config: VLMTriggerConfig,
        task_prompt: str,
        ros_rate: float,
        output_directory: Path,
    ) -> None:
        self.config = config
        self.task_prompt = task_prompt
        self.ros_rate = float(ros_rate)
        self.current_command = task_prompt
        self.expected_effect = "The robot should make normal progress toward the task goal without any unexpected interruptions or failures."

        # --------------------------------------------------------
        # Validate camera
        # --------------------------------------------------------

        if config.camera_key not in self.VIEW_MAP:
            raise ValueError(
                f"Unsupported VLM camera key: {config.camera_key}. "
                f"Supported camera keys: {list(self.VIEW_MAP.keys())}"
            )

        self.view = self.VIEW_MAP[config.camera_key]

        # --------------------------------------------------------
        # Sliding video window
        # --------------------------------------------------------

        self.window_frames = max(
            2,
            round(config.window_seconds * self.ros_rate),
        )

        self.frames: deque[np.ndarray] = deque(
            maxlen=self.window_frames
        )

        # --------------------------------------------------------
        # Async inference worker
        # --------------------------------------------------------

        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="vlm-trigger",
        )

        self.future: Future[TriggerResult] | None = None

        # Number of consecutive PAUSE results
        self.pause_streak = 0

        # --------------------------------------------------------
        # Save VLM input clips
        # --------------------------------------------------------

        self.clip_directory = (
            output_directory / "vlm_trigger_clips"
        )

        self.clip_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------------
        # Load VLM
        # --------------------------------------------------------

        model_path = Path(
            config.model_path
            or vlm_common.MODE_INFO[config.mode][1]
        ).expanduser().resolve()

        log_robot.info(
            "Loading VLM mode=%s path=%s device=%s",
            config.mode,
            model_path,
            config.device_map,
        )

        model_args = SimpleNamespace(
            mode=config.mode,
            dtype=config.dtype,
            attn_implementation=config.attn_implementation,
            device_map=config.device_map,
        )

        self.generate_args = SimpleNamespace(
            mode=config.mode,
            fps=config.sample_fps,
            temperature=config.temperature,

            # Disable thinking for trigger inference
            qwen38_thinking=False,
            qwen35_thinking=False,
        )

        (
            self.model,
            self.processor,
            self.provenance,
        ) = vlm_common.load_model(
            model_path,
            model_args,
        )

        log_robot.info(
            "VLM trigger agent initialized: "
            "camera=%s window=%.2fs (%d frames) "
            "check_interval=%d steps",
            config.camera_key,
            config.window_seconds,
            self.window_frames,
            config.check_interval_steps,
        )

    # ========================================================
    # Sliding window handling
    # ========================================================

    def append_observation(
        self,
        observation,
    ) -> None:
        frame = observation_to_rgb(
            observation,
            self.config.camera_key,
        )

        self.frames.append(frame)

    def ready(
        self,
        step: int,
    ) -> bool:
        """
        Ready when:

        1. Sliding window is full.
        2. Current step matches check interval.
        3. No previous VLM inference is running.
        """

        return (
            len(self.frames) == self.window_frames
            and step % self.config.check_interval_steps == 0
            and self.future is None
        )

    # ========================================================
    # Async request
    # ========================================================

    def submit(
        self,
        episode: int,
        step: int,
    ) -> None:
        """
        Submit one VLM inference without blocking robot loop.
        """

        if self.future is not None:
            return

        request = TriggerRequestData(
            episode=episode,
            step=step,
            start_s=(
                step - self.window_frames + 1
            ) / self.ros_rate,
            end_s=step / self.ros_rate,
            submitted_at=time.perf_counter(),

            # Copy frames so the deque can continue changing
            frames=list(self.frames),
        )

        log_model.debug(
            "Submitting VLM trigger: "
            "episode=%d step=%d window=[%.2f, %.2f]",
            episode,
            step,
            request.start_s,
            request.end_s,
        )

        self.future = self.executor.submit(
            self._infer,
            request,
        )

    # ========================================================
    # Poll result
    # ========================================================

    def poll(
        self,
    ) -> TriggerResult | None:
        """
        Non-blocking result check.
        """

        if self.future is None:
            return None

        if not self.future.done():
            return None

        try:
            result = self.future.result()
        finally:
            self.future = None

        if result.final_decision == "PAUSE":
            self.pause_streak += 1
        else:
            self.pause_streak = 0

        return result

    def confirmed_pause(
        self,
        result: TriggerResult,
    ) -> bool:
        """
        Return True when PAUSE has been confirmed enough times.
        """

        return (
            result.final_decision == "PAUSE"
            and self.pause_streak
            >= self.config.pause_confirmations
        )

    # ========================================================
    # Episode reset
    # ========================================================

    def reset_episode(
        self,
    ) -> None:
        """
        Reset temporal state.

        If an asynchronous request is still running, wait for
        it before resetting. This prevents an old episode result
        from leaking into a new episode.
        """

        if self.future is not None:
            try:
                self.future.result()
            except Exception:
                log_robot.exception(
                    "Pending VLM request failed during reset"
                )

            self.future = None

        self.frames.clear()
        self.pause_streak = 0

    # ========================================================
    # Shutdown
    # ========================================================

    def close(
        self,
    ) -> None:
        self.executor.shutdown(wait=True)

    # ========================================================
    # Actual VLM inference
    # ========================================================

    def _infer(
        self,
        request: TriggerRequestData,
    ) -> TriggerResult:

        clip_path = self.clip_directory / (
            f"episode_{request.episode:03d}_"
            f"step_{request.step:05d}.mp4"
        )

        # ----------------------------------------------------
        # Save temporal window as video
        # ----------------------------------------------------

        imageio.mimsave(
            str(clip_path),
            request.frames,
            fps=self.ros_rate,
            codec="libx264",
        )

        # ----------------------------------------------------
        # Build trigger prompt
        # ----------------------------------------------------

        prompt = pause_trigger.build_pause_prompt(
            self.task_prompt,
            [self.view],
            request.start_s,
            request.end_s,
            self.current_command,
            self.expected_effect,
        )

        # ----------------------------------------------------
        # VLM inference
        # ----------------------------------------------------

        raw, timing = vlm_common.generate(
            self.model,
            self.processor,
            self.view,
            [clip_path],
            prompt,
            self.generate_args,
            self.config.max_new_tokens,
        )

        # ----------------------------------------------------
        # Parse VLM JSON
        # ----------------------------------------------------

        parsed, _ = vlm_common.extract_json(raw)

        raw_decision = vlm_common.norm(
            parsed.get("trigger_decision")
        )

        (
            final_decision,
            guard_reason,
            schema_valid,
            current_phase, # phase triggered by the VLM decision
            confidence,
            expected_effect, # phase effect expected by the VLM decision
            observed_effect, # phase effect observed by the VLM decision
        ) = pause_trigger.final_trigger_decision(
            parsed
        )

        # Preserve ordinary PREGRASP execution, but do not erase a PAUSE for a
        # directly observed PREGRASP stall.  The stage-1 module applies the
        # same semantic guard; this duplicate check protects against importing
        # an older stage-1 module in a long-running ROS process.
        failure_text = " ".join(
            (
                observed_effect,
                str(parsed.get("evidence") or ""),
            )
        ).lower()
        immediate_safety_signals = (
            "unsafe",
            "collision",
            "collided",
            "self-collision",
            "crash",
            "emergency",
        )
        pregrasp_stall_signals = (
            "no visible approach progress",
            "no approach progress",
            "no visible progress",
            "no task progress",
            "no progress",
            "grasp miss",
            "missed the target",
            "target remained on",
            "stalled",
            "stuck",
            "unchanged",
            "same position",
            "remains in the same",
            "not getting closer",
            "does not get closer",
            "did not get closer",
            "moves away",
            "moved away",
            "repeatedly moves",
            "repeated motion",
        )
        if (
            current_phase == "PREGRASP"
            and final_decision == "PAUSE"
            and not any(
                signal in failure_text
                for signal in pregrasp_stall_signals
            )
            and not any(
                signal in failure_text
                for signal in immediate_safety_signals
            )
        ):
            final_decision = "CONTINUE"
            guard_reason = "SIM_PREGRASP_NO_COMPLETED_ATTEMPT"

        return TriggerResult(
            episode=request.episode,
            source_step=request.step,

            start_s=request.start_s,
            end_s=request.end_s,

            completed_at=time.perf_counter(),

            ####phase parameters####
            current_phase=current_phase,
            expected_effect=expected_effect,
            observed_effect=observed_effect,
            ########################


            raw_decision=raw_decision,
            final_decision=final_decision,

            confidence=confidence,
            evidence=str(
                parsed.get("evidence") or ""
            ),
            guard_reason=guard_reason,

            schema_valid=schema_valid,
            raw_output=raw,

            inference_seconds=timing[
                "end_to_end_seconds"
            ],

            clip_path=str(clip_path),
        )


class VLMFailureVerifier:
    """
    Stage-2 verifier executed in an external Python environment.

    The Kuavo simulation continues running inside the lingbotvla
    environment. Qwen3.5 is launched using the Python executable
    from the dedicated qwen35 environment.
    """

    ALLOWED_FAILURES = {
        "FALSE_ALARM",
        "GRASP_MISS",
        "OBJECT_DROP",
        "PLACE_MISS",
        "WRONG_OBJECT",
        "WRONG_DESTINATION",
        "COLLISION",
        "UNKNOWN",
    }

    ALLOWED_ACTIONS = {
        "RESUME",
        "RESET_AND_RETRY",
        "STOP",
    }

    def __init__(
        self,
        config: VLMVerifierConfig,
        task_prompt: str,
    ) -> None:

        self.config = config
        self.task_prompt = task_prompt

        # ----------------------------------------------------
        # Validate Qwen3.5 Python environment
        # ----------------------------------------------------

        self.python_path = Path(
            config.python_path
        ).expanduser().resolve()

        if not self.python_path.is_file():
            raise FileNotFoundError(
                "Stage-2 Python executable does not exist: "
                f"{self.python_path}"
            )

        # ----------------------------------------------------
        # Validate external verifier script
        # ----------------------------------------------------

        self.verifier_script = Path(
            config.verifier_script
        ).expanduser().resolve()

        if not self.verifier_script.is_file():
            raise FileNotFoundError(
                "Stage-2 verifier script does not exist: "
                f"{self.verifier_script}"
            )

        # ----------------------------------------------------
        # Validate model path
        # ----------------------------------------------------

        self.model_path = Path(
            config.model_path
            or "/media/data/jiayi/hf_model/Qwen3.5-9B"
        ).expanduser().resolve()

        if not self.model_path.is_dir():
            raise NotADirectoryError(
                "Stage-2 VLM model path does not exist: "
                f"{self.model_path}"
            )

        log_robot.info(
            "Stage-2 verifier configured as external process: "
            "python=%s model=%s script=%s device=%s",
            self.python_path,
            self.model_path,
            self.verifier_script,
            config.device_map,
        )

    def verify(
        self,
        trigger: TriggerResult,
    ) -> VerificationResult:

        # ----------------------------------------------------
        # Build Stage-2 prompt
        # ----------------------------------------------------

        prompt = f"""
You are the stage-2 safety verifier for robot manipulation.

Task goal:
{self.task_prompt}

Stage-1 information:

Current phase:
{trigger.current_phase}

Expected effect:
{trigger.expected_effect}

Observed effect:
{trigger.observed_effect}

Stage-1 evidence:
{trigger.evidence}

Stage 1 paused robot execution because it suspected a failure.

Review the chronological video carefully.

Important:
- Do not assume arm motion means task progress.
- Check whether the action was actually completed before declaring failure.
- Visual x-y alignment in the head camera does NOT prove that the gripper
  has reached the correct vertical depth.
- A gripper that is still descending toward the object is not a grasp miss.
- A grasp miss should only be declared when a grasp/lift attempt has
  actually completed and the target was not acquired.
- Choose FALSE_ALARM when the Stage-1 pause occurred while the robot was
  still normally approaching, aligning, descending, or executing an
  incomplete action.

Failure types:
- FALSE_ALARM
- GRASP_MISS
- OBJECT_DROP
- PLACE_MISS
- WRONG_OBJECT
- WRONG_DESTINATION
- COLLISION
- UNKNOWN

Recovery policy:
- FALSE_ALARM -> RESUME
- GRASP_MISS -> RESET_AND_RETRY
- OBJECT_DROP -> RESET_AND_RETRY
- PLACE_MISS -> RESET_AND_RETRY
- WRONG_OBJECT -> STOP
- WRONG_DESTINATION -> STOP
- COLLISION -> STOP
- UNKNOWN -> STOP

Return exactly one JSON object:

{{
  "failure_type": "FALSE_ALARM | GRASP_MISS | OBJECT_DROP | PLACE_MISS | WRONG_OBJECT | WRONG_DESTINATION | COLLISION | UNKNOWN",
  "recovery_action": "RESUME | RESET_AND_RETRY | STOP",
  "confidence": "LOW | MEDIUM | HIGH",
  "evidence": "short directly visible evidence"
}}
""".strip()

        # ----------------------------------------------------
        # Run Qwen3.5 using external conda environment
        # ----------------------------------------------------

        command = [
            str(self.python_path),
            str(self.verifier_script),

            "--video",
            str(trigger.clip_path),

            "--model-path",
            str(self.model_path),

            "--prompt",
            prompt,

            "--device-map",
            self.config.device_map,

            "--dtype",
            self.config.dtype,

            "--attn-implementation",
            self.config.attn_implementation,

            "--max-new-tokens",
            str(self.config.max_new_tokens),

            "--temperature",
            str(self.config.temperature),
        ]

        log_robot.info(
            "Starting external stage-2 verifier: "
            "episode=%d source_step=%d clip=%s",
            trigger.episode,
            trigger.source_step,
            trigger.clip_path,
        )

        start = time.perf_counter()

        try:
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )

        except subprocess.TimeoutExpired:

            inference_seconds = (
                time.perf_counter() - start
            )

            log_robot.error(
                "Stage-2 verifier timed out after %.1fs",
                inference_seconds,
            )

            return VerificationResult(
                failure_type="UNKNOWN",
                recovery_action="STOP",
                confidence="LOW",
                evidence="Stage-2 verifier timed out.",
                raw_output="",
                inference_seconds=inference_seconds,
            )

        inference_seconds = (
            time.perf_counter() - start
        )

        # ----------------------------------------------------
        # Handle external process failure
        # ----------------------------------------------------

        if process.returncode != 0:

            log_robot.error(
                "Stage-2 verifier failed with return code %d\n"
                "stderr:\n%s",
                process.returncode,
                process.stderr,
            )

            return VerificationResult(
                failure_type="UNKNOWN",
                recovery_action="STOP",
                confidence="LOW",
                evidence=(
                    "Stage-2 external verifier process failed."
                ),
                raw_output=process.stdout,
                inference_seconds=inference_seconds,
            )

        # ----------------------------------------------------
        # Parse JSON returned by qwen35_failure_verifier.py
        # ----------------------------------------------------

        raw = process.stdout.strip()

        try:
            parsed = json.loads(raw)

        except json.JSONDecodeError:

            log_robot.error(
                "Stage-2 verifier returned invalid JSON:\n%s\n"
                "stderr:\n%s",
                raw,
                process.stderr,
            )

            return VerificationResult(
                failure_type="UNKNOWN",
                recovery_action="STOP",
                confidence="LOW",
                evidence=(
                    "Stage-2 verifier returned invalid JSON."
                ),
                raw_output=raw,
                inference_seconds=inference_seconds,
            )

        failure_type = str(
            parsed.get("failure_type") or ""
        ).strip().upper()

        action = str(
            parsed.get("recovery_action") or ""
        ).strip().upper()

        confidence = str(
            parsed.get("confidence") or ""
        ).strip().upper()

        evidence = str(
            parsed.get("evidence") or ""
        ).strip()

        # ----------------------------------------------------
        # Fail-safe validation
        # ----------------------------------------------------

        if (
            failure_type not in self.ALLOWED_FAILURES
            or action not in self.ALLOWED_ACTIONS
            or confidence not in {
                "LOW",
                "MEDIUM",
                "HIGH",
            }
            or not evidence
        ):
            log_robot.error(
                "Invalid stage-2 response: %s",
                raw,
            )

            failure_type = "UNKNOWN"
            action = "STOP"
            confidence = "LOW"

            if not evidence:
                evidence = (
                    "Stage-2 verifier returned invalid output."
                )

        # ----------------------------------------------------
        # Enforce deterministic recovery policy
        # ----------------------------------------------------

        expected_action = {
            "FALSE_ALARM": "RESUME",

            "GRASP_MISS": "RESET_AND_RETRY",
            "OBJECT_DROP": "RESET_AND_RETRY",
            "PLACE_MISS": "RESET_AND_RETRY",

        }.get(
            failure_type,
            "STOP",
        )

        if action != expected_action:

            log_robot.warning(
                "Overriding stage-2 recovery action: "
                "model=%s expected=%s failure=%s",
                action,
                expected_action,
                failure_type,
            )

            action = expected_action

        return VerificationResult(
            failure_type=failure_type,
            recovery_action=action,
            confidence=confidence,
            evidence=evidence,
            raw_output=raw,
            inference_seconds=inference_seconds,
        )


# ============================================================
# Trigger logging
# ============================================================


def append_trigger_record(
    path: Path,
    result: TriggerResult,
    observed_step: int,
) -> None:

    record = asdict(result)

    # Step when robot loop actually received the result
    record["observed_step"] = observed_step

    # Difference between video-window endpoint
    # and when the inference result became available.
    record["step_lag"] = (
        observed_step - result.source_step
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:

        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# Single episode
# ============================================================


def run_single_episode_agentic(
    config,
    policy,
    preprocessor,
    postprocessor,
    episode: int,
    output_directory: Path,
    trigger_agent: VLMTriggerAgent,
    verifier: VLMFailureVerifier | None,
    trigger_log_path: Path,
    pause_publisher,
) -> tuple[int, int]:

    cfg = config.inference
    task_prompt = cfg.task_prompt
    recovery_count = 0

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = gym.make(
        config.env.env_name,
        max_episode_steps=cfg.max_episode_steps,
        config=config,
    )

    ros_manager = ROSManager()

    ros_manager.register_subscriber(
        "/simulator/success",
        Bool,
        base.env_success_callback,
    )

    start_service = rospy.ServiceProxy(
        "/simulator/start",
        Trigger,
    )

    # --------------------------------------------------------
    # Reset episode
    # --------------------------------------------------------

    policy.reset()
    trigger_agent.reset_episode()

    observation, info = env.reset(
        seed=cfg.seed
    )

    # First image enters VLM buffer
    trigger_agent.append_observation(
        observation
    )

    start_service(
        TriggerRequest()
    )

    # --------------------------------------------------------
    # Rollout recording
    # --------------------------------------------------------

    cam_keys = [
        key
        for key in observation
        if "images" in key or "depth" in key
    ]

    rollout_frames: dict[
        str,
        list[np.ndarray],
    ] = {
        key: []
        for key in cam_keys
    }

    rewards = []

    step = 0
    done = False
    pause_count = 0

    log_robot.info(
        "Starting VLM-agentic episode %d",
        episode,
    )

    # ========================================================
    # Closed-loop episode
    # ========================================================

    while not done:

        # ----------------------------------------------------
        # 1. Check asynchronous VLM result
        # ----------------------------------------------------

        result = trigger_agent.poll()

        if result is not None:

            append_trigger_record(
                trigger_log_path,
                result,
                step,
            )

            log_robot.info(
                "VLM trigger episode=%d source_step=%d observed_step=%d "
                "phase=%s raw=%s final=%s confidence=%s guard=%s inference=%.3fs "
                "expected=%s observed=%s evidence=%s",
                episode,
                result.source_step,
                step,
                result.current_phase,
                result.raw_decision,
                result.final_decision,
                result.confidence,
                result.guard_reason,
                result.inference_seconds,
                result.expected_effect,
                result.observed_effect,
                result.evidence,
            )

            # ------------------------------------------------
            # Confirmed VLM failure trigger
            # ------------------------------------------------

            if trigger_agent.confirmed_pause(
                result
            ):
                pause_count += 1

                log_robot.warning(
                    "VLM PAUSE triggered: "
                    "episode=%d "
                    "step=%d "
                    "source_step=%d "
                    "evidence=%s",
                    episode,
                    step,
                    result.source_step,
                    result.evidence,
                )

                # Send pause command to robot controller
                control.arm_controller.pause()

                # Publish pause state
                pause_publisher.publish(True)

                # Tell original evaluation control logic
                # that the evaluation is currently paused.
                base.pause_flag.set()

                if verifier is not None:
                    verification = verifier.verify(result)
                    log_robot.warning(
                        "Stage-2 verdict: episode=%d type=%s action=%s confidence=%s "
                        "inference=%.3fs evidence=%s",
                        episode, verification.failure_type,
                        verification.recovery_action, verification.confidence,
                        verification.inference_seconds, verification.evidence,
                    )
                    if verification.recovery_action == "RESUME":
                        control.arm_controller.resume()
                        base.pause_flag.clear()
                        pause_publisher.publish(False)
                        trigger_agent.reset_episode()
                        continue
                    if (
                        verification.recovery_action == "RESET_AND_RETRY"
                        and recovery_count < verifier.config.max_retries_per_episode
                    ):
                        recovery_count += 1
                        log_robot.warning(
                            "Stage-2 recovery: resetting episode %d (retry %d/%d)",
                            episode, recovery_count, verifier.config.max_retries_per_episode,
                        )
                        policy.reset()
                        observation, info = env.reset(seed=cfg.seed)
                        trigger_agent.reset_episode()
                        trigger_agent.append_observation(observation)
                        control.arm_controller.resume()
                        base.pause_flag.clear()
                        pause_publisher.publish(False)
                        continue
                    log_robot.error("Stage-2 recovery stopped execution for episode %d", episode)
                    return 0, pause_count

                # Clear old video history.
                #
                # After resume, the VLM must collect a new
                # complete temporal window rather than
                # immediately re-triggering using pre-pause
                # frames.
                trigger_agent.reset_episode()

                # IMPORTANT:
                #
                # Do NOT execute policy.select_action()
                # or env.step() in this iteration.
                #
                # Go directly to the next loop iteration.
                # base.check_control_signals() below will then
                # handle the existing pause/resume mechanism.
                continue

        # ----------------------------------------------------
        # 2. Existing pause / resume / stop handling
        # ----------------------------------------------------

        if not base.check_control_signals():

            log_robot.warning(
                "Episode %d stopped by control signal",
                episode,
            )

            env.close()
            ros_manager.close()

            return 0, pause_count

        # If check_control_signals() returned after a resume,
        # report that the control loop is active again.
        if not base.pause_flag.is_set():
            pause_publisher.publish(False)

        # ----------------------------------------------------
        # 3. Policy inference
        # ----------------------------------------------------

        start_time = time.perf_counter()

        policy_observation = observation

        if cfg.policy_type != "client":
            policy_observation = inject_task_prompt(
                policy_observation,
                task_prompt,
            )

            if preprocessor is not None:
                policy_observation = preprocessor(
                    policy_observation
                )

        # ----------------------------------------------------
        # 4. Select action
        # ----------------------------------------------------

        with torch.inference_mode():
            action = policy.select_action(
                policy_observation
            )

        # Local policies use action postprocessing.
        if (
            cfg.policy_type != "client"
            and postprocessor is not None
        ):
            action = postprocessor(action)

        # ----------------------------------------------------
        # 5. Convert action to numpy
        # ----------------------------------------------------

        if torch.is_tensor(action):
            numpy_action = (
                action
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
        else:
            numpy_action = np.asarray(action)

            if (
                numpy_action.ndim > 1
                and numpy_action.shape[0] == 1
            ):
                numpy_action = numpy_action.squeeze(0)

        # ----------------------------------------------------
        # 6. Execute robot action
        # ----------------------------------------------------

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            numpy_action
        )

        rewards.append(reward)

        # ----------------------------------------------------
        # 7. Add newest frame to VLM sliding window
        # ----------------------------------------------------

        trigger_agent.append_observation(
            observation
        )

        # ----------------------------------------------------
        # 8. Record rollout video
        # ----------------------------------------------------

        for key in cam_keys:
            rollout_frames[key].append(
                observation_to_rgb(
                    observation,
                    key,
                )
            )

        step += 1

        # ----------------------------------------------------
        # 9. Submit asynchronous VLM trigger request
        # ----------------------------------------------------

        if trigger_agent.ready(step):

            trigger_agent.submit(
                episode,
                step,
            )

        # ----------------------------------------------------
        # 10. Episode termination
        # ----------------------------------------------------

        done = (
            terminated
            or truncated
            or base.success_evt.is_set()
        )

        log_model.debug(
            "episode %d "
            "step %d "
            "time %.3fs",
            episode,
            step,
            time.perf_counter()
            - start_time,
        )

    # ========================================================
    # Episode completed
    # ========================================================

    fps = env.unwrapped.ros_rate

    # --------------------------------------------------------
    # Save rollout videos
    # --------------------------------------------------------

    for camera_key, frames in rollout_frames.items():

        if not frames:
            continue

        output_path = (
            output_directory
            / f"rollout_{episode}_{camera_key}.mp4"
        )

        base.save_rollout_video(
            output_path,
            frames,
            fps,
        )

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    success = base.success_evt.is_set()

    log_robot.info(
        "Episode %d finished: success=%s pauses=%d steps=%d",
        episode,
        success,
        pause_count,
        step,
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    env.close()
    ros_manager.close()

    del (
        rewards,
        observation,
        env,
        ros_manager,
        rollout_frames,
    )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        1 if success else 0,
        pause_count,
    )


# ============================================================
# Full evaluation
# ============================================================


def kuavo_eval_autotest_vlm_agentic(
    config,
    trigger_config: VLMTriggerConfig,
    verifier_config: VLMVerifierConfig | None = None,
) -> None:

    cfg = config.inference

    # ========================================================
    # Resolve policy checkpoint
    # ========================================================

    if cfg.pretrained_path:

        pretrained_path = Path(
            cfg.pretrained_path
        )

    else:

        pretrained_path = Path(
            f"outputs/train/"
            f"{cfg.task}/"
            f"{cfg.method}/"
            f"{cfg.timestamp}/"
            f"epoch{cfg.epoch}"
        )

    # ========================================================
    # Output directory
    # ========================================================

    base_output_directory = (
        resolve_eval_output_dir(
            pretrained_path,
            Path("outputs/eval"),
        )
    )

    run_name = datetime.datetime.now().strftime(
        "vlm_agentic_%Y%m%d_%H%M%S"
    )

    output_directory = (
        base_output_directory / run_name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_log_path = (
        output_directory
        / "evaluation_autotest_vlm_agentic.log"
    )

    trigger_log_path = (
        output_directory
        / "vlm_trigger_events.jsonl"
    )

    trigger_log_path.write_text(
        "",
        encoding="utf-8",
    )

    # ========================================================
    # Initial evaluation log
    # ========================================================

    with evaluation_log_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            f"Evaluation Timestamp: "
            f"{datetime.datetime.now()}\n"
        )

        handle.write(
            f"Total Episodes: "
            f"{cfg.eval_episodes}\n"
        )

        handle.write(
            "VLM Trigger: "
            f"{json.dumps(asdict(trigger_config))}\n"
        )

    # ========================================================
    # Seed / device
    # ========================================================

    torch.manual_seed(cfg.seed)

    device = torch.device(
        cfg.device
    )

    task_prompt = getattr(
        cfg,
        "task_prompt",
        "robot manipulation",
    )

    # ========================================================
    # Policy
    # ========================================================

    (
        policy,
        preprocessor,
        postprocessor,
        _,
    ) = base.setup_policy(
        pretrained_path,
        cfg.policy_type,
        device,
        task_prompt=task_prompt,
    )

    # ========================================================
    # ROS services
    # ========================================================

    reset_service = rospy.ServiceProxy(
        "/simulator/reset",
        Trigger,
    )

    init_service = rospy.Service(
        "/simulator/init",
        Trigger,
        base.env_init_service,
    )

    pause_publisher = rospy.Publisher(
        "/kuavo/pause_state",
        Bool,
        queue_size=1,
    )

    # ========================================================
    # VLM agent
    # ========================================================

    trigger_agent = VLMTriggerAgent(
        trigger_config,
        task_prompt,
        config.env.ros_rate,
        output_directory,
    )
    verifier = (
        VLMFailureVerifier(verifier_config, task_prompt)
        if verifier_config is not None and verifier_config.enabled
        else None
    )

    # ========================================================
    # Wait for simulator init
    # ========================================================

    wait_times = 8

    while (
        not base.init_evt.is_set()
        and wait_times > 0
    ):

        log_robot.info(
            "Waiting for first env init..."
        )

        time.sleep(1)

        wait_times -= 1

    base.safe_reset_service(
        reset_service
    )

    base.init_evt.clear()

    # ========================================================
    # Episode loop
    # ========================================================

    rows = []

    success_count = 0

    try:

        for episode in range(
            cfg.eval_episodes
        ):

            # ------------------------------------------------
            # Wait for simulator
            # ------------------------------------------------

            while not base.init_evt.is_set():

                log_robot.info(
                    "Waiting for env init..."
                )

                if not base.check_control_signals():

                    log_robot.warning(
                        "Evaluation stopped "
                        "while waiting for env init"
                    )

                    return

                time.sleep(1)

            # ------------------------------------------------
            # Run one episode
            # ------------------------------------------------

            result, pause_count = (
                run_single_episode_agentic(
                    config=config,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    episode=episode,
                    output_directory=output_directory,
                    trigger_agent=trigger_agent,
                    verifier=verifier,
                    trigger_log_path=trigger_log_path,
                    pause_publisher=pause_publisher,
                )
            )

            success_count += result

            rows.append(
                {
                    "episode": episode,
                    "success": bool(result),
                    "vlm_pause_count": pause_count,
                }
            )

            # ------------------------------------------------
            # Episode log
            # ------------------------------------------------

            with evaluation_log_path.open(
                "a",
                encoding="utf-8",
            ) as handle:

                handle.write(
                    f"Episode {episode}: "
                    f"success={bool(result)}, "
                    f"vlm_pause_count={pause_count}\n"
                )

            # ------------------------------------------------
            # Prepare next episode
            # ------------------------------------------------

            policy.reset()

            trigger_agent.reset_episode()

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            base.safe_reset_service(
                reset_service
            )

            base.init_evt.clear()
            base.success_evt.clear()
            base.pause_flag.clear()

            pause_publisher.publish(False)

    finally:

        # Always shut down asynchronous thread
        trigger_agent.close()

    # ========================================================
    # Episode summary CSV
    # ========================================================

    summary_path = (
        output_directory
        / "vlm_agentic_episode_summary.csv"
    )

    if rows:

        with summary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "episode",
                    "success",
                    "vlm_pause_count",
                ],
            )

            writer.writeheader()
            writer.writerows(rows)

    else:

        # Avoid rows[0] IndexError when eval_episodes == 0
        with summary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "episode",
                    "success",
                    "vlm_pause_count",
                ],
            )

            writer.writeheader()

    # ========================================================
    # Final evaluation summary
    # ========================================================

    total_episodes = len(rows)

    success_rate = (
        success_count / total_episodes
        if total_episodes > 0
        else 0.0
    )

    with evaluation_log_path.open(
        "a",
        encoding="utf-8",
    ) as handle:

        handle.write(
            f"Success Count: "
            f"{success_count}/{total_episodes}\n"
        )

        handle.write(
            f"Success Rate: "
            f"{success_rate:.2%}\n"
        )

    log_model.info(
        "VLM-agentic evaluation completed"
    )

    log_model.info(
        "Success count: %d/%d",
        success_count,
        total_episodes,
    )

    log_model.info(
        "Success rate: %.2f%%",
        success_rate * 100.0,
    )

    log_model.info(
        "Outputs: %s",
        output_directory,
    )

    # ========================================================
    # ROS cleanup
    # ========================================================

    init_service.shutdown()

    base.pause_sub.unregister()
    base.stop_sub.unregister()
