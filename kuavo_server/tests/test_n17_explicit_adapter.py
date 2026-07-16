from types import SimpleNamespace
import unittest

import numpy as np

from kuavo_server.adapters.isaac_gr00t_n17_explicit import IsaacGr00tN17ExplicitAdapter


class TestIsaacGr00tN17ExplicitAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = IsaacGr00tN17ExplicitAdapter.__new__(IsaacGr00tN17ExplicitAdapter)
        self.adapter.which_arm = "both"

    def test_exact_state16_is_preserved(self):
        state = np.arange(16, dtype=np.float32)
        np.testing.assert_array_equal(self.adapter._canonical_state16(state[None, :]), state)

    def test_wrong_state_dimension_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exact Kuavo state shape"):
            self.adapter._canonical_state16(np.zeros(14, dtype=np.float32))

    def test_action_groups_are_composed_in_kuavo_order(self):
        action = {
            "left_arm": np.arange(7, dtype=np.float32),
            "left_gripper": np.array([7], dtype=np.float32),
            "right_arm": np.arange(8, 15, dtype=np.float32),
            "right_gripper": np.array([15], dtype=np.float32),
        }
        np.testing.assert_array_equal(
            self.adapter._compose_kuavo_action(action), np.arange(16, dtype=np.float64)
        )

    def test_missing_action_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            self.adapter._compose_kuavo_action(
                {
                    "left_arm": np.zeros(7),
                    "left_gripper": np.zeros(1),
                    "right_arm": np.zeros(7),
                }
            )

    def test_checkpoint_contract_is_exact(self):
        self.adapter.model = SimpleNamespace(
            video_keys=["head", "wrist_left", "wrist_right"],
            state_keys=["left_arm", "left_gripper", "right_arm", "right_gripper"],
            action_keys=["left_arm", "left_gripper", "right_arm", "right_gripper"],
            state_dims={"left_arm": 7, "left_gripper": 1, "right_arm": 7, "right_gripper": 1},
            action_dims={"left_arm": 7, "left_gripper": 1, "right_arm": 7, "right_gripper": 1},
        )
        self.adapter._validate_checkpoint_contract()


if __name__ == "__main__":
    unittest.main()
