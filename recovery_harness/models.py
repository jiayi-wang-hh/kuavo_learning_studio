from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CameraFrame:
    name: str
    image: Any
    timestamp: float | None
    received_at: float
    timestamp_source: str = "header"


@dataclass
class MultiViewObservation:
    images: dict[str, CameraFrame | None]
    tactile: dict[str, Any] = field(default_factory=dict)
    robot_state: dict[str, Any] = field(default_factory=dict)
    perception: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def available_views(self) -> list[str]:
        return [name for name, frame in self.images.items() if frame is not None]


@dataclass
class SkillRecommendation:
    scene_state: str
    task_progress: str
    recommended_skill: str
    target_object: str | None
    reason: str
    requires_verification: bool
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float | None = None
    label: str = "package_candidate"
    source: str = "external"
    partial: bool = False


@dataclass
class TrackEvidence:
    track_id: str
    bbox_xyxy: tuple[float, float, float, float]
    age_seconds: float
    valid: bool
    source: str
    partial: bool = False

