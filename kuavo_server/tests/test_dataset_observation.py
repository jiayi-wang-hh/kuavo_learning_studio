import unittest

import numpy as np

from types import SimpleNamespace

from kuavo_server.dataset_observation import (
    _normalize_minmax,
    _row_to_kuavo_observation,
    _state_normalized_values,
    _state_safety_score,
    _state_statistics_from_loader,
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

    def test_minmax_normalization_clips_to_unit_range(self):
        values = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
        normalized = _normalize_minmax(
            values,
            np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(normalized, np.array([-1.0, 0.0, 1.0], dtype=np.float32))

    def test_state_safety_score_uses_worst_normalized_joint(self):
        row = {
            "state.left_arm": np.full(7, 0.25, dtype=np.float32),
            "state.left_gripper": np.array([0.0], dtype=np.float32),
            "state.right_arm": np.full(7, 0.75, dtype=np.float32),
            "state.right_gripper": np.array([-0.5], dtype=np.float32),
        }
        stats = {
            key: {
                "q01": np.full(dim, -1.0, dtype=np.float32),
                "q99": np.full(dim, 1.0, dtype=np.float32),
            }
            for key, dim in {
                "left_arm": 7,
                "left_gripper": 1,
                "right_arm": 7,
                "right_gripper": 1,
            }.items()
        }
        normalized = _state_normalized_values(row, stats)
        self.assertAlmostEqual(_state_safety_score(normalized), 0.75)

    def test_state_statistics_do_not_require_action_modality_config(self):
        loader = SimpleNamespace(
            modality_meta={
                "state": {
                    "left_arm": {"start": 0, "end": 7},
                    "left_gripper": {"start": 7, "end": 8},
                    "right_arm": {"start": 8, "end": 15},
                    "right_gripper": {"start": 15, "end": 16},
                }
            },
            stats={
                "observation.state": {
                    "q01": list(range(16)),
                    "q99": list(range(100, 116)),
                    "min": list(range(200, 216)),
                    "max": list(range(300, 316)),
                }
            },
        )
        stats = _state_statistics_from_loader(loader)
        self.assertEqual(stats["left_arm"]["q01"], list(range(7)))
        self.assertEqual(stats["right_gripper"]["q99"], [115])


if __name__ == "__main__":
    unittest.main()
