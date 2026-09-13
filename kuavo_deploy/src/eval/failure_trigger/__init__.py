"""Task-state failure triggers used by closed-loop evaluation."""

from .base_trigger import CriticInput, CriticResult, FailureTrigger
from .visual_critic_trigger import (
    FlorenceVisualCriticModel,
    VisualCriticConfig,
    VisualCriticTrigger,
)

__all__ = [
    "CriticInput",
    "CriticResult",
    "FailureTrigger",
    "FlorenceVisualCriticModel",
    "VisualCriticConfig",
    "VisualCriticTrigger",
]
