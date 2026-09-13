"""Focused tests for stage-1 pause guards; no model weights are loaded."""

from __future__ import annotations

import unittest

from vlm_test.test_vlm_video_joy_pause_trigger import final_trigger_decision


def payload(
    *,
    phase: str = "PREGRASP",
    decision: str = "PAUSE",
    observed: str = "The gripper is near the object and is still aligning.",
    evidence: str = "The gripper remains near the target.",
) -> dict[str, str]:
    return {
        "current_phase": phase,
        "expected_effect": "The gripper should move closer to grasp readiness.",
        "observed_effect": observed,
        "trigger_decision": decision,
        "confidence": "HIGH",
        "evidence": evidence,
    }


class PauseTriggerGuardTests(unittest.TestCase):
    def test_normal_pregrasp_pause_is_suppressed(self) -> None:
        decision, reason, *_ = final_trigger_decision(payload())
        self.assertEqual(decision, "CONTINUE")
        self.assertEqual(reason, "PREGRASP_NO_COMPLETED_ATTEMPT")

    def test_pregrasp_stall_pause_is_preserved(self) -> None:
        decision, reason, *_ = final_trigger_decision(
            payload(
                observed="No visible approach progress; the gripper stays at the same position.",
                evidence="Repeated motion leaves the gripper no closer to the object.",
            )
        )
        self.assertEqual(decision, "PAUSE")
        self.assertEqual(reason, "NONE")

    def test_pregrasp_stall_overrides_model_continue(self) -> None:
        decision, reason, *_ = final_trigger_decision(
            payload(
                decision="CONTINUE",
                observed="The gripper is stuck with no task progress.",
                evidence="The same position persists through the window.",
            )
        )
        self.assertEqual(decision, "PAUSE")
        self.assertEqual(reason, "OBSERVED_EFFECT_CONTRADICTION")

    def test_grasp_miss_overrides_model_continue(self) -> None:
        decision, reason, *_ = final_trigger_decision(
            payload(
                phase="GRASP",
                decision="CONTINUE",
                observed="The gripper missed its attempted target; the target remained on the table.",
                evidence="The gripper moved away while the attempted target remained at its original position.",
            )
        )
        self.assertEqual(decision, "PAUSE")
        self.assertEqual(reason, "OBSERVED_EFFECT_CONTRADICTION")

    def test_non_pregrasp_pause_is_unchanged(self) -> None:
        decision, reason, *_ = final_trigger_decision(
            payload(
                phase="GRASP",
                observed="The object remained on the table after the gripper closed.",
                evidence="The grasp attempt did not acquire the object.",
            )
        )
        self.assertEqual(decision, "PAUSE")
        self.assertEqual(reason, "NONE")


if __name__ == "__main__":
    unittest.main()
