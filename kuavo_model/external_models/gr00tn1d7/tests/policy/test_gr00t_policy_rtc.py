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


@pytest.mark.parametrize(
    "override, message",
    [
        ({"rtc_overlap_steps": 0}, "rtc_overlap_steps"),
        ({"rtc_overlap_steps": 17}, "rtc_overlap_steps"),
        ({"rtc_frozen_steps": 9}, "rtc_frozen_steps"),
        ({"rtc_ramp_rate": 0.0}, "rtc_ramp_rate"),
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
