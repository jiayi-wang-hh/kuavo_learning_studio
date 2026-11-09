"""Single-worker VLM scheduler; sensor/control callers only enqueue and poll."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import logging
import time

log = logging.getLogger("kuavo.recovery.scheduler")


def _frame_ages(observation, at: float) -> dict[str, float]:
    return {name: max(0.0, at - frame.received_at)
            for name, frame in observation.images.items() if frame is not None}


class VLMInvocationScheduler:
    def __init__(self, reasoner, *, periodic_interval: float, max_observation_age: float,
                 max_result_age: float | None = None):
        self.reasoner = reasoner
        self.periodic_interval = max(0.0, periodic_interval)
        self.max_observation_age = max_observation_age
        self.max_result_age = max_observation_age if max_result_age is None else max_result_age
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recovery-vlm")
        self._future: Future | None = None
        self._queued_event = None
        self._paused = False
        self._discard_next_result = False
        self._last_start: float | None = None
        self._created_at = time.monotonic()
        self._last_events: dict[str, float] = {}
        self._closed = False

    def _invoke_reasoner(self, observation):
        result = self.reasoner.reason(observation)
        return result, time.monotonic()

    def tick(self, observation, events=(), *, now: float | None = None):
        now = time.monotonic() if now is None else now
        completed = self.poll(now=now)
        if self._closed or self._paused:
            return completed
        for event in events:
            key = f"{event.source}:{event.event_type}:{event.evidence.get('track_id', event.evidence.get('hand', ''))}"
            if now - self._last_events.get(key, float("-inf")) >= self.max_observation_age:
                self._last_events[key] = now
                self._queued_event = event  # newest event; event work takes priority over periodic work
        reason = None
        if self._future is None:
            if self._queued_event is not None:
                reason, self._queued_event = {"type": "event", "event": self._queued_event}, None
            elif now - (self._last_start if self._last_start is not None else self._created_at) >= self.periodic_interval:
                reason = {"type": "periodic", "event": None}
            if reason is not None:
                input_frame_ages = _frame_ages(observation, now)
                input_age = max(input_frame_ages.values(), default=float("inf"))
                if input_age > self.max_observation_age:
                    log.info("Skipping VLM invocation: input observation too old (%.3fs)", input_age)
                    return completed
                self._last_start = now
                submitted_at = now
                log.info("vlm_trigger %s", reason)
                self._future = self._worker.submit(self._invoke_reasoner, observation)
                self._metadata = (submitted_at, observation, reason, input_age, input_frame_ages)
        return completed

    def poll(self, *, now: float | None = None):
        now = time.monotonic() if now is None else now
        if self._future is None or not self._future.done():
            return None
        future, self._future = self._future, None
        started, observation, trigger, input_age, input_frame_ages = self._metadata
        try:
            result, inference_completed_at = future.result()
            latency = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            completion_frame_ages = _frame_ages(observation, inference_completed_at)
            consumption_frame_ages = _frame_ages(observation, now)
            result_age_at_completion = max(completion_frame_ages.values(), default=float("inf"))
            result_age_at_consumption = max(consumption_frame_ages.values(), default=float("inf"))
            wall_time = max(0.0, inference_completed_at - started)
            polling_delay = max(0.0, now - inference_completed_at)
            valid = (input_age <= self.max_observation_age
                     and result_age_at_completion <= self.max_result_age
                     and result_age_at_consumption <= self.max_result_age
                     and not self._discard_next_result)
            discard_reason = ("pause_signal" if self._discard_next_result else
                              "stale_input" if input_age > self.max_observation_age else
                              "stale_result_at_completion" if result_age_at_completion > self.max_result_age else
                              "stale_before_consumption" if result_age_at_consumption > self.max_result_age else None)
            record = {"result": result, "trigger": trigger, "inference_started_at": started,
                      "inference_latency_seconds": latency, "inference_wall_time_seconds": wall_time,
                      "inference_completed_at": inference_completed_at,
                      "polling_delay_seconds": polling_delay,
                      "observation_age_seconds": result_age_at_consumption,
                      "input_observation_age_seconds": input_age,
                      "input_frame_ages_seconds": input_frame_ages,
                      "completion_frame_ages_seconds": completion_frame_ages,
                      "consumption_frame_ages_seconds": consumption_frame_ages,
                      "result_age_seconds": result_age_at_consumption,
                      "result_age_at_completion_seconds": result_age_at_completion,
                      "result_age_at_consumption_seconds": result_age_at_consumption,
                      "valid": valid, "discard_reason": discard_reason,
                      "completed_at": now, "observation": observation}
            self._discard_next_result = False
            log.info("vlm_result trigger=%s model_latency=%s worker_wall_time=%s input_age=%s completion_age=%s consumption_age=%s poll_delay=%s valid=%s",
                     trigger, latency, wall_time, input_age, result_age_at_completion, result_age_at_consumption, polling_delay, valid)
            return record
        except Exception:
            log.exception("scheduled VLM inference failed")
            return {"result": None, "trigger": trigger, "valid": False, "error": "inference_failed", "completed_at": now}

    def set_paused(self, paused: bool) -> None:
        paused = bool(paused)
        if paused and not self._paused:
            self._queued_event = None
            if self._future is not None:
                self._discard_next_result = True
        self._paused = paused

    def close(self):
        self._closed = True
        # `cancel_futures` was added in Python 3.9; the repository also runs on 3.8.
        self._worker.shutdown(wait=False)
