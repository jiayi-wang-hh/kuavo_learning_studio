from __future__ import annotations

import time


class BinaryContactProcessor:
    """Three normal-force channels/finger, at least two fingers, sustained 100 ms."""
    FIELDS = ("normal_force1", "normal_force2", "normal_force3")

    def __init__(self, thresholds: list[float] | None, valid_status_values: list[int] | None,
                 sustain_seconds: float = 0.1):
        if thresholds is not None and len(thresholds) != 5:
            raise ValueError("Provide one normal-force threshold per finger")
        self.thresholds = thresholds
        self.valid_status_values = set(valid_status_values) if valid_status_values is not None else None
        self.sustain_seconds = sustain_seconds
        self._since: dict[str, float | None] = {"left": None, "right": None}

    def _hand(self, fingers, hand: str, now: float) -> int | None:
        if self.valid_status_values is None or self.thresholds is None or fingers is None or len(fingers) != 5:
            self._since[hand] = None
            return None
        contacts = []
        for i, finger in enumerate(fingers):
            status = getattr(finger, "status", None)
            values = [getattr(finger, field, None) for field in self.FIELDS]
            if status not in self.valid_status_values or any(v is None for v in values):
                self._since[hand] = None
                return None
            try:
                force_sum = sum(float(v) for v in values)
            except (TypeError, ValueError):
                self._since[hand] = None
                return None
            contacts.append(force_sum > self.thresholds[i])
        active = sum(contacts) >= 2
        if not active:
            self._since[hand] = None
            return 0
        if self._since[hand] is None:
            self._since[hand] = now
        return int(now - self._since[hand] >= self.sustain_seconds)

    def process(self, msg, *, now: float | None = None) -> dict[str, int | None]:
        now = time.monotonic() if now is None else now
        left = getattr(msg, "left_hand", None)
        right = getattr(msg, "right_hand", None)
        return {
            "left_multifinger_contact": self._hand(left, "left", now),
            "right_multifinger_contact": self._hand(right, "right", now),
        }

