"""Model-independent interface for asynchronous task-failure triggers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

CriticState = Literal["PROGRESSING", "STALLED", "FAILURE", "SUCCESS", "UNKNOWN"]
CriticConfidence = Literal["LOW", "MEDIUM", "HIGH"]


@dataclass(frozen=True)
class CriticInput:
    frame: np.ndarray
    subtask: str
    source_step: int
    source_timestamp: float
    source_monotonic_timestamp: float


@dataclass(frozen=True)
class CriticResult:
    state: CriticState
    progress_score: float | None
    confidence: CriticConfidence
    reason: str
    source_step: int
    source_timestamp: float
    source_monotonic_timestamp: float
    inference_ms: float
    result_age_ms: float | None = None
    stale_discarded: bool = False

    def with_age(self, age_ms: float, stale: bool) -> "CriticResult":
        return replace(self, result_age_ms=age_ms, stale_discarded=stale)


class FailureTrigger(ABC):
    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def submit(
        self,
        frame: np.ndarray,
        subtask: str,
        source_step: int,
        timestamp: float,
        monotonic_timestamp: float | None = None,
    ) -> None: ...

    @abstractmethod
    def get_latest_result(self) -> CriticResult | None: ...

    @abstractmethod
    def shutdown(self) -> None: ...
