import unittest

import numpy as np

from kuavo_server.dataset_observation import (
    _row_to_kuavo_observation,
    _validate_modality_metadata,
)


class TestDatasetObservation(unittest.TestCase):
    def test_dual_arm_row_is_converted_in_exact_kuavo_order(self):
        row = {
            "state.left_arm": np.arange(7, dtype=np.float32),
            "state.left_gripper": np.array([7], dtype=np.float32),
            "state.right_arm": np.arange(8, 15, dtype=np.float32),
            "state.right_gripper": np.array([15], dtype=np.float32),
            "video.head": np.zeros((8, 9, 3), dtype=np.uint8),
            "video.wrist_left": np.ones((8, 9, 3), dtype=np.uint8),
            "video.wrist_right": np.full((8, 9, 3), 2, dtype=np.uint8),
        }
        obs = _row_to_kuavo_observation(row, "pick")
        np.testing.assert_array_equal(obs["observation.state"], np.arange(16))
        self.assertEqual(obs["prompt"], "pick")
        self.assertEqual(obs["observation.images.wrist_cam_r"].shape, (8, 9, 3))

    def test_modality_metadata_rejects_single_arm_dataset(self):
        metadata = {
            "state": {
                "left_arm": {"start": 0, "end": 7},
                "left_gripper": {"start": 7, "end": 8},
            },
            "video": {"head": {}, "wrist_left": {}},
        }
        with self.assertRaisesRegex(ValueError, "dual-arm Kuavo"):
            _validate_modality_metadata(metadata)

    def test_state_group_dimension_is_checked(self):
        metadata = {
            "state": {
                "left_arm": {"start": 0, "end": 6},
                "left_gripper": {"start": 6, "end": 7},
                "right_arm": {"start": 7, "end": 14},
                "right_gripper": {"start": 14, "end": 15},
            },
            "video": {"head": {}, "wrist_left": {}, "wrist_right": {}},
        }
        with self.assertRaisesRegex(ValueError, "left_arm expected dim 7"):
            _validate_modality_metadata(metadata)


if __name__ == "__main__":
    unittest.main()
