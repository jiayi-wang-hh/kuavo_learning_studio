import unittest

import numpy as np

from kuavo_deploy.safety.safety_supervisor import (
    APPLE_PICK_DELTA_Q_LIMIT,
    SafetySupervisor,
)


def make_action(q14):
    q14 = np.asarray(q14, dtype=np.float64)
    return np.concatenate((q14[:7], [0.25], q14[7:], [0.75]))


class SafetySupervisorTest(unittest.TestCase):
    def setUp(self):
        self.supervisor = SafetySupervisor(task_name="apple_pick")
        self.q_ref = np.zeros(14, dtype=np.float64)

    def test_apple_pick_profile_matches_expected_values(self):
        np.testing.assert_array_equal(
            APPLE_PICK_DELTA_Q_LIMIT,
            np.array(
                [
                    0.07,
                    0.03,
                    0.07,
                    0.18,
                    0.18,
                    0.12,
                    0.08,
                    0.07,
                    0.04,
                    0.09,
                    0.14,
                    0.17,
                    0.09,
                    0.12,
                ],
                dtype=np.float64,
            ),
        )

    def test_normal_action_passes_and_grippers_are_ignored(self):
        q_cmd = APPLE_PICK_DELTA_Q_LIMIT * 0.5
        result = self.supervisor.check(make_action(q_cmd), self.q_ref)

        self.assertTrue(result.safe)
        np.testing.assert_array_equal(result.details["q_cmd"], q_cmd)

    def test_value_equal_to_limit_passes(self):
        result = self.supervisor.check(
            make_action(APPLE_PICK_DELTA_Q_LIMIT), self.q_ref
        )

        self.assertTrue(result.safe)

    def test_jump_over_limit_fails_with_joint_details(self):
        q_cmd = np.zeros(14, dtype=np.float64)
        q_cmd[1] = APPLE_PICK_DELTA_Q_LIMIT[1] + 0.001
        result = self.supervisor.check(make_action(q_cmd), self.q_ref)

        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "joint_command_continuity_limit_exceeded")
        self.assertEqual(result.details["violating_joint_indices"], [1])
        self.assertEqual(result.details["violating_joint_names"], ["zarm_l2_joint"])

    def test_nan_action_fails(self):
        action = make_action(np.zeros(14))
        action[4] = np.nan
        result = self.supervisor.check(action, self.q_ref)

        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "invalid_safety_input")

    def test_wrong_action_shape_fails(self):
        result = self.supervisor.check(np.zeros(14), self.q_ref)

        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "invalid_safety_input")

    def test_invalid_reference_fails(self):
        result = self.supervisor.check(make_action(np.zeros(14)), np.zeros(13))

        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "invalid_safety_input")

    def test_unknown_task_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown safety task"):
            SafetySupervisor(task_name="unknown_task")


if __name__ == "__main__":
    unittest.main()
