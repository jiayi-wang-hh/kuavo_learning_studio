"""Conservative CV/tactile event candidates used only to schedule scene reasoning."""
from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass(frozen=True)
class TriggerEvent:
    event_type: str
    source: str
    evidence: dict
    timestamp: float


class EventTrigger:
    """Debounce meaningful evidence; never decides which skill should run."""
    def __init__(self, *, confidence_threshold: float = 0.5, persistence_hits: int = 2,
                 cooldown_seconds: float = 3.0, position_delta_px: float = 30.0):
        self.confidence_threshold = confidence_threshold
        self.persistence_hits = max(1, int(persistence_hits))
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self.position_delta_px = max(0.0, position_delta_px)
        self._counts: dict[str, int] = {}
        self._last_emit: dict[str, float] = {}
        self._previous_contacts: dict[str, int | None] = {}
        self._previous_boxes: dict[str, tuple[float, float, float, float]] = {}
        self._seen_tracks: set[str] = set()
        self._active_keys: set[str] = set()

    def _candidate(self, key: str, event_type: str, source: str, evidence: dict, now: float,
                   required_hits: int | None = None) -> TriggerEvent | None:
        self._active_keys.add(key)
        self._counts[key] = self._counts.get(key, 0) + 1
        if self._counts[key] < (required_hits or self.persistence_hits):
            return None
        if now - self._last_emit.get(key, float("-inf")) < self.cooldown_seconds:
            return None
        self._last_emit[key] = now
        return TriggerEvent(event_type, source, evidence, now)

    def update(self, perception: dict, tactile: dict, robot_state: dict, *, now: float | None = None) -> list[TriggerEvent]:
        now = time.monotonic() if now is None else now
        self._active_keys = set()
        events: list[TriggerEvent] = []
        tracks = perception.get("candidate_tracks", []) if isinstance(perception, dict) else []
        current_ids: set[str] = set()
        for track in tracks:
            if not track.get("track_valid", False):
                continue
            ident = str(track.get("track_id", "unknown"))
            partial = bool(track.get("partial", False))
            if partial:
                ev = self._candidate("partial:" + ident, "SCENE_UNCERTAIN", "cv", {"track_id": ident, "partial": True}, now)
                if ev: events.append(ev)
            confidence = track.get("confidence")
            confidence_ok = confidence is not None and confidence >= self.confidence_threshold
            if not partial and track.get("track_source") != "csrt" and confidence_ok:
                current_ids.add(ident)
            if not partial and track.get("track_source") != "csrt" and confidence_ok and track.get("body_side_violation") is True:
                ev = self._candidate("outside:" + ident, "POSSIBLE_PACKAGE_OUTSIDE_PLATFORM", "cv", track, now)
                if ev: events.append(ev)
            box = track.get("bbox_xyxy")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                box = tuple(float(v) for v in box)
                old = self._previous_boxes.get(ident)
                if old and not partial and track.get("track_source") != "csrt":
                    old_c = ((old[0] + old[2]) / 2, (old[1] + old[3]) / 2)
                    new_c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                    delta = ((old_c[0] - new_c[0]) ** 2 + (old_c[1] - new_c[1]) ** 2) ** 0.5
                    if delta >= self.position_delta_px and confidence_ok:
                        ev = self._candidate("moved:" + ident, "PACKAGE_POSITION_CHANGED", "cv", {"track_id": ident, "delta_px": delta, "confidence": confidence}, now)
                        if ev: events.append(ev)
                self._previous_boxes[ident] = box
        for ident in self._seen_tracks - current_ids:
            ev = self._candidate("lost:" + ident, "PACKAGE_DETECTION_LOST", "cv", {"track_id": ident}, now)
            if ev: events.append(ev)
        if current_ids:
            self._seen_tracks = current_ids
        if perception and perception.get("geometry_validity") == "unreliable" and perception.get("package_visible") == "unknown":
            ev = self._candidate("uncertain", "SCENE_UNCERTAIN", "cv", {"geometry_validity": "unreliable"}, now)
            if ev: events.append(ev)

        for hand in ("left", "right"):
            key = f"{hand}_multifinger_contact"
            current = tactile.get(key) if isinstance(tactile, dict) else None
            previous = self._previous_contacts.get(hand)
            if current not in (0, 1):
                if current is None and previous in (0, 1):
                    ev = self._candidate("tactile_unknown:" + hand, "TACTILE_UNKNOWN", "tactile", {"hand": hand}, now, required_hits=1)
                    if ev: events.append(ev)
                self._previous_contacts[hand] = None
                continue
            if previous in (0, 1) and current != previous:
                etype = "CONTACT_ESTABLISHED" if current == 1 else "CONTACT_LOST"
                ev = self._candidate(f"contact:{hand}:{current}", etype, "tactile", {"hand": hand, "previous": previous, "current": current}, now, required_hits=1)
                if ev: events.append(ev)
            self._previous_contacts[hand] = current

        for key in list(self._counts):
            if key not in self._active_keys:
                self._counts[key] = 0
        # Robot context is attached, not used to turn sensor evidence into a skill decision.
        return [TriggerEvent(e.event_type, e.source, {**e.evidence, "robot_state": robot_state or {}}, e.timestamp) for e in events]
