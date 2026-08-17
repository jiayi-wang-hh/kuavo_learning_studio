from types import SimpleNamespace

import pytest
import torch

from gr00t.policy.gr00t_policy import Gr00tPolicy


def make_policy(action_horizon: int = 16) -> Gr00tPolicy:
    policy = object.__new__(Gr00tPolicy)
    policy.modality_configs = {
        "action": SimpleNamespace(delta_indices=list(range(action_horizon)))
    }
    policy._rtc_previous_normalized_action = torch.ones(1, action_horizon, 29)
    return policy


def test_rtc_disabled_clears_feedback_state() -> None:
    policy = make_policy()

    assert policy._rtc_options(None) is None
    assert policy._rtc_previous_normalized_action is None


def test_rtc_options_add_model_action_horizon() -> None:
    policy = make_policy()

    options = policy._rtc_options(
        {
            "enable_rtc": True,
            "rtc_overlap_steps": 8,
            "rtc_frozen_steps": 4,
            "rtc_ramp_rate": 2.0,
        }
    )

    assert options == {
        "action_horizon": 16,
        "rtc_overlap_steps": 8,
        "rtc_frozen_steps": 4,
        "rtc_ramp_rate": 2.0,
    }


def test_rtc_options_accept_time_aligned_previous_offset() -> None:
    policy = make_policy(action_horizon=32)

    options = policy._rtc_options(
        {
            "enable_rtc": True,
            "rtc_overlap_steps": 8,
            "rtc_frozen_steps": 3,
            "rtc_ramp_rate": 2.0,
            "rtc_previous_offset": 13,
        }
    )

    assert options["rtc_previous_offset"] == 13


def test_rtc_previous_action_is_aligned_by_physical_offset() -> None:
    previous = torch.arange(16, dtype=torch.float32).view(1, 16, 1)
    options = {
        "action_horizon": 16,
        "rtc_overlap_steps": 4,
        "rtc_frozen_steps": 2,
        "rtc_ramp_rate": 2.0,
        "rtc_previous_offset": 5,
    }

    aligned, source_start = Gr00tPolicy._align_rtc_previous_action(previous, options)

    assert source_start == 5
    assert aligned[0, 12:16, 0].tolist() == [5.0, 6.0, 7.0, 8.0]
    assert previous[0, 12:16, 0].tolist() == [12.0, 13.0, 14.0, 15.0]


@pytest.mark.parametrize(
    "override, message",
    [
        ({"rtc_overlap_steps": 0}, "rtc_overlap_steps"),
        ({"rtc_overlap_steps": 17}, "rtc_overlap_steps"),
        ({"rtc_frozen_steps": 9}, "rtc_frozen_steps"),
        ({"rtc_ramp_rate": 0.0}, "rtc_ramp_rate"),
        ({"rtc_previous_offset": 9}, "rtc_previous_offset"),
    ],
)
def test_rtc_options_reject_invalid_values(override: dict, message: str) -> None:
    policy = make_policy()
    options = {
        "enable_rtc": True,
        "rtc_overlap_steps": 8,
        "rtc_frozen_steps": 4,
        "rtc_ramp_rate": 2.0,
        **override,
    }

    with pytest.raises(ValueError, match=message):
        policy._rtc_options(options)


def test_reset_clears_rtc_feedback_state() -> None:
    policy = make_policy()

    result = policy.reset()

    assert result == {"rtc_state_cleared": True}
    assert policy._rtc_previous_normalized_action is None
