from __future__ import annotations

import time
from collections import deque

from .models import CameraFrame, MultiViewObservation


class TimestampedBuffer:
    def __init__(self, capacity: int = 16, max_age_seconds: float = 1.0):
        self.frames = deque(maxlen=capacity)
        self.max_age_seconds = max_age_seconds

    def append(self, frame: CameraFrame) -> None:
        self.frames.append(frame)

    def latest(self, *, now: float | None = None) -> CameraFrame | None:
        now = time.monotonic() if now is None else now
        if not self.frames:
            return None
        frame = self.frames[-1]
        return frame if now - frame.received_at <= self.max_age_seconds else None

    def recent(self, count: int = 4, *, now: float | None = None) -> list[CameraFrame]:
        now = time.monotonic() if now is None else now
        return [frame for frame in list(self.frames)[-count:] if now - frame.received_at <= self.max_age_seconds]


class ObservationSynchronizer:
    """Timestamped camera buffers; ROS zero stamps remain explicitly marked as receipt-time."""
    def __init__(self, enabled_views: list[str], max_age_seconds: float = 1.0):
        self.buffers = {name: TimestampedBuffer(max_age_seconds=max_age_seconds) for name in enabled_views}
        self.tactile: dict = {}
        self.tactile_received_at: float | None = None
        self.robot_state: dict = {}
        self.history: deque[dict] = deque(maxlen=8)

    def push_image(self, frame: CameraFrame) -> None:
        if frame.name in self.buffers:
            self.buffers[frame.name].append(frame)

    def snapshot(self, *, now: float | None = None, perception: dict | None = None) -> MultiViewObservation:
        now = time.monotonic() if now is None else now
        images = {name: buffer.latest(now=now) for name, buffer in self.buffers.items()}
        return MultiViewObservation(images, dict(self.tactile), dict(self.robot_state), perception or {}, list(self.history))

    def freshness(self, *, now: float | None = None) -> dict[str, float | None]:
        now = time.monotonic() if now is None else now
        return {name: (now - frame.received_at if (frame := buf.latest(now=now)) else None) for name, buf in self.buffers.items()}


def ros_stamp_seconds(stamp) -> tuple[float | None, str]:
    try:
        value = float(stamp.to_sec())
    except (AttributeError, TypeError, ValueError):
        value = 0.0
    return (value, "header") if value > 0 else (None, "receipt_time_header_zero")

