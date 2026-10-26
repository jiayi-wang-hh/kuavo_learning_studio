from __future__ import annotations

import time
from collections import deque
from typing import Callable

from .models import Detection, TrackEvidence


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _clip_polygon(poly: list[tuple[float, float]], clip: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clipping for convex platform polygons."""
    output = poly
    signed = sum(clip[i][0] * clip[(i + 1) % len(clip)][1] - clip[(i + 1) % len(clip)][0] * clip[i][1] for i in range(len(clip)))
    orientation = 1 if signed >= 0 else -1
    for i, a in enumerate(clip):
        b = clip[(i + 1) % len(clip)]
        input_points, output = output, []
        if not input_points:
            break
        def inside(p):
            return orientation * ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= -1e-9
        def intersection(p, q):
            dx1, dy1 = q[0] - p[0], q[1] - p[1]
            dx2, dy2 = b[0] - a[0], b[1] - a[1]
            den = dx1 * dy2 - dy1 * dx2
            if abs(den) < 1e-12:
                return q
            t = ((a[0] - p[0]) * dy2 - (a[1] - p[1]) * dx2) / den
            return p[0] + t * dx1, p[1] + t * dy1
        previous = input_points[-1]
        for current in input_points:
            if inside(current):
                if not inside(previous):
                    output.append(intersection(previous, current))
                output.append(current)
            elif inside(previous):
                output.append(intersection(previous, current))
            previous = current
    return output


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(sum(points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points))) / 2)


def box_platform_overlap(box: tuple[float, float, float, float], platform_polygon: list[tuple[float, float]], *, complete_object: bool) -> float | None:
    """Return box overlap only for a complete, head-camera object candidate."""
    if not complete_object or len(platform_polygon) < 3:
        return None
    x1, y1, x2, y2 = box
    area = _area(box)
    if area <= 0:
        return None
    footprint = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return min(1.0, max(0.0, polygon_area(_clip_polygon(footprint, platform_polygon)) / area))


def crosses_body_edge(box: tuple[float, float, float, float], edge: list[tuple[float, float]], *, complete_object: bool,
                      platform_polygon: list[tuple[float, float]] | None = None) -> bool | None:
    if not complete_object or len(edge) != 2:
        return None
    if not platform_polygon:
        return None
    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    inside_reference = (sum(p[0] for p in platform_polygon) / len(platform_polygon),
                        sum(p[1] for p in platform_polygon) / len(platform_polygon))
    a, b = edge
    cross = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    return cross(center) * cross(inside_reference) < 0


class CandidateTracker:
    """Optional external detector + short-lived CSRT tracker evidence."""
    def __init__(self, detector: Callable | None = None, max_track_age_seconds: float = 1.5):
        self.detector = detector
        self.max_track_age_seconds = max_track_age_seconds
        self._tracks: dict[str, tuple[TrackEvidence, object | None, float, float]] = {}
        self._counter = 0

    def update(self, image, *, now: float | None = None) -> list[TrackEvidence]:
        now = time.monotonic() if now is None else now
        try:
            import numpy as np
            if not isinstance(image, np.ndarray):
                image = np.asarray(image)
        except ImportError:
            pass
        detections: list[Detection] = list(self.detector(image) if self.detector else [])
        # Existing CSRT tracks are evidence only; expired/failed tracks become UNKNOWN.
        try:
            import cv2
        except ImportError:
            cv2 = None
        gray_or_color = image
        updated: dict[str, tuple[TrackEvidence, object | None, float, float]] = {}
        unmatched = set(self._tracks)
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox_xyxy
            det_box = detection.bbox_xyxy
            best_id, best_overlap = None, 0.0
            for old_id in unmatched:
                old = self._tracks[old_id][0]
                ox1, oy1, ox2, oy2 = old.bbox_xyxy
                intersection = _area((max(x1, ox1), max(y1, oy1), min(x2, ox2), min(y2, oy2)))
                union = _area(det_box) + _area(old.bbox_xyxy) - intersection
                overlap = intersection / union if union > 0 else 0.0
                if overlap > best_overlap:
                    best_id, best_overlap = old_id, overlap
            if best_id is not None and best_overlap >= 0.2:
                ident = best_id
                unmatched.discard(best_id)
            else:
                self._counter += 1
                ident = f"package_candidate_{self._counter}"
            tracker = None
            factory = None
            if cv2 is not None:
                factory = getattr(cv2, "TrackerCSRT_create", None)
                if factory is None and hasattr(cv2, "legacy"):
                    factory = getattr(cv2.legacy, "TrackerCSRT_create", None)
            if factory is not None:
                tracker = factory()
                tracker.init(gray_or_color, (x1, y1, x2 - x1, y2 - y1))
            updated[ident] = (TrackEvidence(ident, detection.bbox_xyxy, 0.0, True, detection.source, detection.partial, detection.confidence), tracker, now, now)
        # Track only previously seeded candidates. Never treat tracking as recovery after expiry.
        for ident in unmatched:
            old, tracker, detected_at, last_update = self._tracks[ident]
            if ident in updated:
                continue
            age = now - detected_at
            if tracker is None or cv2 is None or age >= self.max_track_age_seconds:
                continue
            ok, rect = tracker.update(gray_or_color)
            if not ok:
                continue
            x, y, w, h = rect
            updated[ident] = (TrackEvidence(ident, (x, y, x + w, y + h), age, True, "csrt", old.partial, old.confidence), tracker, detected_at, now)
        self._tracks = updated
        return [item[0] for item in updated.values()]


class PerceptionAdapter:
    """Combines optional candidates with conservative head-view ROI geometry."""
    def __init__(self, tracker: CandidateTracker, platform_polygon: list[list[float]] | None = None,
                 body_side_edge: list[list[float]] | None = None,
                 image_size: tuple[int, int] | None = None):
        self.tracker = tracker
        self.platform = [tuple(map(float, p)) for p in (platform_polygon or [])]
        self.body_edge = [tuple(map(float, p)) for p in (body_side_edge or [])]
        self.image_size = image_size

    def inspect(self, image, *, view_name: str = "head", now: float | None = None) -> dict:
        candidates = self.tracker.update(image, now=now) if view_name == "head" else []
        evidence = []
        for track in candidates:
            complete = not track.partial and track.source != "csrt"
            overlap = box_platform_overlap(track.bbox_xyxy, self.platform, complete_object=complete)
            violation = crosses_body_edge(track.bbox_xyxy, self.body_edge, complete_object=complete,
                                          platform_polygon=self.platform)
            evidence.append({
                "track_id": track.track_id, "bbox_xyxy": list(track.bbox_xyxy),
                "track_age_seconds": track.age_seconds, "track_valid": track.valid,
                "track_source": track.source, "partial": track.partial,
                "confidence": track.confidence,
                "overlap_ratio": overlap, "body_side_violation": violation,
                "geometry_validity": "unreliable" if overlap is None else "bbox_approximation",
            })
        valid_geometry = [item for item in evidence if item["overlap_ratio"] is not None]
        violations = [item["body_side_violation"] for item in evidence if item["body_side_violation"] is not None]
        if any(violations):
            location, body_violation = "body_side_or_platform_edge", True
        elif evidence and valid_geometry and all(item["overlap_ratio"] == 1.0 for item in valid_geometry):
            location, body_violation = "platform_candidate", False if violations else "unknown"
        elif evidence and valid_geometry:
            location, body_violation = "platform_edge_or_outside", False if violations else "unknown"
        else:
            location, body_violation = "unknown", "unknown"
        return {
            "package_visible": True if evidence else "unknown",
            "package_partially_visible": any(x["partial"] for x in evidence) if evidence else "unknown",
            "package_occluded": "unknown", "package_location": location,
            "body_side_violation": body_violation,
            "overlap_ratio": valid_geometry[0]["overlap_ratio"] if len(valid_geometry) == 1 else None,
            "geometry_validity": "bbox_approximation" if valid_geometry else "unreliable",
            "candidate_tracks": evidence,
            "platform_geometry_available": bool(self.platform),
            "body_side_edge_available": bool(self.body_edge),
            "geometry_view": "head_camera_only",
            "geometry_coordinate_system": "image pixels, origin at top-left" if self.platform else "unknown",
            "head_image_size_px": list(self.image_size) if self.image_size else None,
            "platform_polygon_px": [list(point) for point in self.platform] if self.platform else None,
            "body_side_edge_px": [list(point) for point in self.body_edge] if self.body_edge else None,
            "body_side_region_hint": "the exterior side beyond body_side_edge in the head image; do not apply this fixed ROI to wrist images" if self.body_edge else "unknown",
            "visual_evidence": "optional detector/tracker candidates; VLM always receives original image",
        }


class TemporalHistory:
    def __init__(self, capacity: int = 8):
        self.items = deque(maxlen=capacity)

    def append(self, observation: dict) -> None:
        self.items.append(observation)

    def snapshot(self) -> list[dict]:
        return list(self.items)

