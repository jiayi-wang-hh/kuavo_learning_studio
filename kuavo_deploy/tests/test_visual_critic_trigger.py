"""Unit tests for V2 scheduling, parsing, staleness, and debounce."""

from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from kuavo_deploy.src.eval.failure_trigger.base_trigger import CriticResult
from kuavo_deploy.src.eval.failure_trigger.visual_critic_trigger import (
    TriggerDecision,
    VisualCriticConfig,
    VisualCriticTrigger,
    parse_critic_output,
)


class BlockingModel:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.steps: list[int] = []

    def infer(self, frame, subtask):
        self.steps.append(int(frame[0, 0, 0]))
        if len(self.steps) == 1:
            self.started.set()
            self.release.wait(timeout=2)
        return {"state": "PROGRESSING", "progress_score": None, "confidence": "HIGH", "reason": "advancing"}


def result(state="STALLED", confidence="HIGH", stale=False):
    return CriticResult(
        state=state, progress_score=None, confidence=confidence, reason="visible state",
        source_step=1, source_timestamp=time.perf_counter(), inference_ms=1,
        result_age_ms=1, stale_discarded=stale,
    )


class VisualCriticTests(unittest.TestCase):
    def test_malformed_output_becomes_unknown_without_keyword_guessing(self):
        parsed = parse_critic_output({"text": "object dropped and stationary"})
        self.assertEqual(parsed[:3], ("UNKNOWN", None, "LOW"))

    def test_failure_is_immediate_and_stall_is_debounced(self):
        decision = TriggerDecision(VisualCriticConfig(stall_confirm_count=3))
        self.assertEqual(decision.evaluate(result()), (False, None))
        self.assertEqual(decision.evaluate(result()), (False, None))
        self.assertEqual(decision.evaluate(result()), (True, "STALLED"))
        decision.reset()
        self.assertEqual(decision.evaluate(result("FAILURE", "MEDIUM")), (True, "FAILURE"))

    def test_stale_failure_never_triggers(self):
        decision = TriggerDecision(VisualCriticConfig())
        self.assertEqual(decision.evaluate(result("FAILURE", stale=True)), (False, None))

    def test_only_newest_pending_frame_is_processed(self):
        model = BlockingModel()
        trigger = VisualCriticTrigger(VisualCriticConfig(), model)
        frame = lambda value: np.full((2, 2, 3), value, dtype=np.uint8)
        trigger.submit(frame(1), "task", 1, time.perf_counter())
        self.assertTrue(model.started.wait(timeout=1))
        trigger.submit(frame(2), "task", 2, time.perf_counter())
        trigger.submit(frame(3), "task", 3, time.perf_counter())
        model.release.set()
        deadline = time.time() + 2
        while len(model.steps) < 2 and time.time() < deadline:
            time.sleep(0.01)
        trigger.shutdown()
        self.assertEqual(model.steps, [1, 3])


if __name__ == "__main__":
    unittest.main()
