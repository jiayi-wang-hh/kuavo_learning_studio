from types import SimpleNamespace
import unittest

import numpy as np

from kuavo_server.adapters.n17_diagnostics import (
    _action_metrics,
    _collect_action_chunks,
)


class _NaNAdapter:
    which_arm = "both"
    execution_horizon = 16

    def __init__(self):
        self.model = SimpleNamespace(infer=self._infer)

    @staticmethod
    def _infer(_model_obs):
        return {"left_arm": np.full((1, 2, 7), np.nan, dtype=np.float32)}

    @staticmethod
    def _convert_action_chunk(_action_dict):
        raise ValueError("action.left_arm contains NaN or Inf")


class TestN17Diagnostics(unittest.TestCase):
    def test_nan_model_output_is_reported_without_raising(self):
        adapter = _NaNAdapter()
        chunks, raw_outputs, errors = _collect_action_chunks(adapter, {}, 2, 0)
        self.assertEqual(chunks, [])
        self.assertEqual(len(errors), 2)
        self.assertEqual(
            raw_outputs[0]["groups"]["left_arm"]["nonfinite_count"], 14
        )

        metrics = _action_metrics(
            adapter,
            np.zeros(16, dtype=np.float32),
            chunks,
            raw_outputs,
            errors,
            2,
        )
        self.assertEqual(metrics["status"], "invalid_model_output")
        self.assertEqual(metrics["valid_chunk_count"], 0)
        self.assertIsNone(metrics["first_chunk"])


if __name__ == "__main__":
    unittest.main()
