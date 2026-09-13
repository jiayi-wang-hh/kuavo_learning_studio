"""Single-frame, asynchronous V2 visual critic trigger."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .base_trigger import CriticInput, CriticResult, FailureTrigger

STATES = {"PROGRESSING", "STALLED", "FAILURE", "SUCCESS", "UNKNOWN"}
CONFIDENCES = {"LOW", "MEDIUM", "HIGH"}
CONFIDENCE_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


@dataclass(frozen=True)
class VisualCriticConfig:
    model_name_or_path: str = "microsoft/Florence-2-base"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    frequency_hz: float = 5.0
    max_result_age_s: float = 1.5
    stall_confirm_count: int = 3
    failure_min_confidence: str = "MEDIUM"
    stall_min_confidence: str = "MEDIUM"
    progress_enabled: bool = False
    progress_epsilon: float = 0.03
    no_progress_confirm_count: int = 3
    max_new_tokens: int = 96
    reset_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        if self.frequency_hz <= 0 or self.max_result_age_s <= 0 or self.reset_timeout_s < 0:
            raise ValueError("frequency_hz/max_result_age_s must be positive and reset_timeout_s non-negative")
        if self.stall_confirm_count < 1 or self.no_progress_confirm_count < 1:
            raise ValueError("confirmation counts must be >= 1")
        if self.failure_min_confidence not in CONFIDENCES:
            raise ValueError("invalid failure_min_confidence")
        if self.stall_min_confidence not in CONFIDENCES:
            raise ValueError("invalid stall_min_confidence")


class VisualCriticModel(Protocol):
    def infer(self, frame: np.ndarray, subtask: str) -> Mapping[str, Any]: ...


def _timestamp_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "to_sec"):
        return float(value.to_sec())
    if hasattr(value, "sec"):
        nanosec = getattr(value, "nanosec", getattr(value, "nsec", 0))
        return float(value.sec) + float(nanosec) * 1e-9
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def observation_camera_timestamp(
    observation: Mapping[str, Any], camera_key: str
) -> float | None:
    """Read a camera/ROS timestamp when an environment exposes one."""
    camera_value = observation.get(camera_key)
    if isinstance(camera_value, Mapping):
        timestamp = _timestamp_seconds(camera_value.get("timestamp"))
        if timestamp is not None:
            return timestamp
    for key in (
        f"{camera_key}.timestamp",
        f"{camera_key}_timestamp",
        "observation.camera_timestamp",
        "observation.timestamp",
        "timestamp",
    ):
        if key in observation:
            timestamp = _timestamp_seconds(observation[key])
            if timestamp is not None:
                return timestamp
    return None


def build_critic_prompt(subtask: str) -> str:
    return f"""You are a lightweight robot execution critic.
Current subtask: {subtask}
Classify only the state visible in the current image.
PROGRESSING: visible progress toward this subtask.
STALLED: no meaningful visible progress and no explicit failure.
FAILURE: visible missed grasp, drop, wrong-object interaction, or incorrect physical interaction.
SUCCESS: this subtask is visibly complete.
UNKNOWN: cannot determine reliably.
Return JSON only:
{{"state":"PROGRESSING|STALLED|FAILURE|SUCCESS|UNKNOWN","progress_score":null,"confidence":"LOW|MEDIUM|HIGH","reason":"short phrase"}}"""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```")
        text = text.removesuffix("```").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def parse_critic_output(payload: Mapping[str, Any]) -> tuple[str, float | None, str, str]:
    """Strict schema parsing. It deliberately has no prose/keyword fallback."""
    state = str(payload.get("state") or "").strip().upper()
    confidence = str(payload.get("confidence") or "").strip().upper()
    reason = str(payload.get("reason") or "").strip()
    score = payload.get("progress_score")
    valid_score = score is None or (
        isinstance(score, (int, float)) and not isinstance(score, bool) and 0 <= score <= 1
    )
    if state not in STATES or confidence not in CONFIDENCES or not reason or not valid_score:
        return "UNKNOWN", None, "LOW", "malformed visual critic output"
    return state, (float(score) if score is not None else None), confidence, reason


class FlorenceVisualCriticModel:
    """Isolated Transformers backend; a fine-tuned checkpoint is a config change."""

    def __init__(self, config: VisualCriticConfig) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._torch = torch
        self._device = torch.device(config.device)
        dtype = getattr(torch, config.dtype)
        self._processor = AutoProcessor.from_pretrained(
            config.model_name_or_path, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, trust_remote_code=True, torch_dtype=dtype
        ).to(self._device).eval()
        self._max_new_tokens = config.max_new_tokens

    def infer(self, frame: np.ndarray, subtask: str) -> Mapping[str, Any]:
        image = Image.fromarray(frame.astype(np.uint8, copy=False)).convert("RGB")
        prompt = build_critic_prompt(subtask)
        inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            ids = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)
        if not getattr(self._model.config, "is_encoder_decoder", False):
            ids = ids[:, inputs["input_ids"].shape[1] :]
        raw = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
        return _extract_json(raw)


class VisualCriticTrigger(FailureTrigger):
    """One in-flight inference plus a single newest pending input slot."""

    def __init__(self, config: VisualCriticConfig, model: VisualCriticModel) -> None:
        self.config = config
        self.model = model
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._running = False
        self._closed = False
        self._pending: tuple[CriticInput, int] | None = None
        self._latest_result: CriticResult | None = None
        self._generation = 0
        self._worker: threading.Thread | None = None

    def submit(
        self,
        frame: np.ndarray,
        subtask: str,
        source_step: int,
        timestamp: float,
        monotonic_timestamp: float | None = None,
    ) -> None:
        item = CriticInput(
            np.array(frame, copy=True), subtask, source_step, timestamp,
            time.perf_counter() if monotonic_timestamp is None else monotonic_timestamp,
        )
        with self._condition:
            if self._closed:
                raise RuntimeError("visual critic is shut down")
            if self._running:
                self._pending = (item, self._generation)
                return
            self._running = True
            generation = self._generation
            self._worker = threading.Thread(
                target=self._run_chain, args=(item, generation), daemon=True,
                name="v2-visual-critic",
            )
            self._worker.start()

    def _run_chain(self, item: CriticInput, generation: int) -> None:
        while True:
            started = time.perf_counter()
            try:
                state, score, confidence, reason = parse_critic_output(
                    self.model.infer(item.frame, item.subtask)
                )
            except Exception as exc:  # model faults are UNKNOWN, not task failures
                state, score, confidence = "UNKNOWN", None, "LOW"
                reason = f"critic inference error: {type(exc).__name__}"
            result = CriticResult(
                state=state, progress_score=score, confidence=confidence, reason=reason,
                source_step=item.source_step, source_timestamp=item.source_timestamp,
                source_monotonic_timestamp=item.source_monotonic_timestamp,
                inference_ms=(time.perf_counter() - started) * 1000,
            )
            with self._condition:
                if generation == self._generation:
                    self._latest_result = result
                pending = self._pending
                self._pending = None
                if pending is None or self._closed:
                    self._running = False
                    self._condition.notify_all()
                    return
                item, generation = pending

    def get_latest_result(self) -> CriticResult | None:
        with self._lock:
            if self._latest_result is None:
                return None
            result = self._latest_result
            self._latest_result = None
        age_ms = max(
            0.0,
            (time.perf_counter() - result.source_monotonic_timestamp) * 1000,
        )
        return result.with_age(age_ms, age_ms > self.config.max_result_age_s * 1000)

    def reset(self) -> None:
        with self._condition:
            # Invalidate before waiting. A late result from this generation is
            # discarded, even if model.generate cannot be cancelled.
            self._generation += 1
            self._pending = None
            self._latest_result = None
            deadline = time.monotonic() + self.config.reset_timeout_s
            while self._running:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)

    def shutdown(self) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            while self._running:
                self._condition.wait(timeout=0.1)


class TriggerDecision:
    """Stateful policy kept separate from model inference and text."""

    def __init__(self, config: VisualCriticConfig) -> None:
        self.config = config
        self.stall_counter = 0
        self.max_progress: float | None = None
        self.no_progress_counter = 0

    def reset(self) -> None:
        self.stall_counter = 0
        self.max_progress = None
        self.no_progress_counter = 0

    def evaluate(self, result: CriticResult) -> tuple[bool, str | None]:
        if (
            result.stale_discarded
            or result.state == "UNKNOWN"
            or result.confidence == "LOW"
        ):
            self.stall_counter = 0
            self.no_progress_counter = 0
            return False, None
        enough_failure = CONFIDENCE_RANK[result.confidence] >= CONFIDENCE_RANK[self.config.failure_min_confidence]
        enough_stall = CONFIDENCE_RANK[result.confidence] >= CONFIDENCE_RANK[self.config.stall_min_confidence]
        if result.state == "FAILURE":
            self.stall_counter = 0
            return (enough_failure, "FAILURE" if enough_failure else None)
        if result.state == "STALLED" and enough_stall:
            self.stall_counter += 1
        else:
            self.stall_counter = 0
        if self.stall_counter >= self.config.stall_confirm_count:
            return True, "STALLED"
        if self.config.progress_enabled and result.progress_score is not None:
            if self.max_progress is None or result.progress_score > self.max_progress + self.config.progress_epsilon:
                self.max_progress = result.progress_score
                self.no_progress_counter = 0
            else:
                self.no_progress_counter += 1
            if self.no_progress_counter >= self.config.no_progress_confirm_count:
                return True, "STALLED"
        return False, None
