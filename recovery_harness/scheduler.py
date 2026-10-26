"""Single-worker VLM scheduler; sensor/control callers only enqueue and poll."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import logging
import time

log = logging.getLogger("kuavo.recovery.scheduler")


class VLMInvocationScheduler:
    def __init__(self, reasoner, *, periodic_interval: float, max_observation_age: float):
        self.reasoner = reasoner
        self.periodic_interval = max(0.0, periodic_interval)
        self.max_observation_age = max_observation_age
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recovery-vlm")
        self._future: Future | None = None
        self._queued_event = None
        self._paused = False
        self._discard_next_result = False
        self._last_start: float | None = None
        self._created_at = time.monotonic()
        self._last_events: dict[str, float] = {}
        self._closed = False

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
                self._last_start = now
                submitted_at = now
                log.info("vlm_trigger %s", reason)
                self._future = self._worker.submit(self.reasoner.reason, observation)
                self._metadata = (submitted_at, observation, reason)
        return completed

    def poll(self, *, now: float | None = None):
        now = time.monotonic() if now is None else now
        if self._future is None or not self._future.done():
            return None
        future, self._future = self._future, None
        started, observation, trigger = self._metadata
        try:
            result = future.result()
            latency = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            age = max((now - frame.received_at for frame in observation.images.values() if frame is not None), default=float("inf"))
            valid = age <= self.max_observation_age and not self._discard_next_result
            record = {"result": result, "trigger": trigger, "inference_started_at": started,
                      "inference_latency_seconds": latency, "observation_age_seconds": age,
                      "valid": valid, "discard_reason": "pause_signal" if self._discard_next_result else None,
                      "completed_at": now, "observation": observation}
            self._discard_next_result = False
            log.info("vlm_result trigger=%s latency=%s age=%s valid=%s", trigger, latency, age, valid)
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
