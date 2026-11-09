"""Synchronized, low-overhead image and decision diagnostics for live dry-run."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time
from typing import Any


PACKAGE_LABELS = {"blue mailer bag", "pink padded envelope"}


def recommendation_fields(recommendation) -> dict[str, Any] | None:
    if recommendation is None:
        return None
    return {
        "scene_state": getattr(recommendation, "scene_state", None),
        "task_progress": getattr(recommendation, "task_progress", None),
        "recommended_skill": getattr(recommendation, "recommended_skill", None),
        "target_object": getattr(recommendation, "target_object", None),
        "reason": getattr(recommendation, "reason", None),
        "requires_verification": getattr(recommendation, "requires_verification", None),
    }


def _event_summaries(events) -> list[dict[str, Any]]:
    output = []
    for event in events or []:
        if isinstance(event, dict):
            evidence = dict(event.get("evidence", {}) or {})
            event_type, source, timestamp = event.get("event_type"), event.get("source"), event.get("timestamp")
        else:
            evidence = dict(getattr(event, "evidence", {}) or {})
            event_type, source, timestamp = (getattr(event, "event_type", None),
                                             getattr(event, "source", None),
                                             getattr(event, "timestamp", None))
        evidence.pop("robot_state", None)
        output.append({"event_type": event_type,
                       "source": source,
                       "timestamp": timestamp,
                       "evidence": evidence})
    return output


def _is_potential_failure(perception: dict, events: list[dict]) -> bool:
    if any(event.get("event_type") == "POSSIBLE_PACKAGE_OUTSIDE_PLATFORM" for event in events):
        return True
    for track in perception.get("candidate_tracks", []):
        if (track.get("label") in PACKAGE_LABELS
                and track.get("mask_available") is True
                and track.get("mask_body_side_extension") is True):
            return True
    return False


def classify_live_outcome(perception: dict, events: list[dict], recommendation,
                          *, accepted: bool, controller_decision: dict | None) -> str:
    fields = recommendation_fields(recommendation) or {}
    skill = fields.get("recommended_skill")
    tracks = [track for track in perception.get("candidate_tracks", [])
              if track.get("label") in PACKAGE_LABELS]
    potential = _is_potential_failure(perception, events)

    if skill == "recover_package" and not accepted:
        return "vlm_recommended_recover_package_rejected_stale"
    if skill == "recover_package" and controller_decision:
        reason = str(controller_decision.get("decision_reason", "")).lower()
        if (controller_decision.get("controller_state") == "STOPPED"
                and ("safety" in reason or "robot state" in reason)):
            return "vlm_recommended_recover_package_controller_rejected_safety_unavailable"
    if potential and skill == "normal_sort":
        return "cv_potential_failure_vlm_recommended_normal_sort"
    if not tracks:
        return "cv_failed_to_detect_package"
    relation_established = any(
        track.get("mask_geometry_valid") is True
        and (track.get("platform_visible_pixels") or 0) > 0
        for track in tracks
    )
    if not relation_established:
        return "cv_detected_package_platform_relation_uncertain"
    if skill == "recover_package":
        return "vlm_recommended_recover_package_controller_pending_or_rejected"
    return "package_detected_no_confirmed_recovery_event"


def cv_detection_summary(perception: dict) -> dict[str, Any]:
    tracks = [track for track in perception.get("candidate_tracks", [])
              if track.get("label") in PACKAGE_LABELS]
    return {
        "package_detected_by_cv": bool(tracks),
        "packages": [{
            "track_id": track.get("track_id"), "label": track.get("label"),
            "partial": track.get("partial"), "occluded": track.get("occluded"),
            "dino_bbox_xyxy": track.get("detector_bbox_xyxy"),
            "mask_available": track.get("mask_available"),
            "mask_intersects_platform": (track.get("platform_visible_pixels", 0) or 0) > 0,
            "mask_intersects_body_side_region": (track.get("body_side_visible_pixels", 0) or 0) > 0,
            "platform_visible_pixels": track.get("platform_visible_pixels"),
            "body_side_visible_pixels": track.get("body_side_visible_pixels"),
            "visible_mask_extends_body_side": track.get("mask_body_side_extension"),
        } for track in tracks],
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items() if key != "mask"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def save_live_snapshot(output_dir: str | Path, observation, cv_events, *, freshness: dict,
                       recommendation=None, outcome: str | None = None,
                       controller_decision: dict | None = None) -> str | None:
    """Write the exact head frame, ROI, DINO boxes, SAM2 masks and associated result."""
    frame = observation.images.get("head") if observation and observation.images else None
    if frame is None:
        return None
    try:
        from PIL import Image, ImageDraw
        import numpy as np

        root = Path(output_dir).expanduser()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        directory = root / f"{stamp}_{time.monotonic_ns()}"
        directory.mkdir(parents=True, exist_ok=False)
        image = frame.image if isinstance(frame.image, Image.Image) else Image.fromarray(np.asarray(frame.image))
        image = image.convert("RGB")
        image.save(directory / "head_rgb.png")
        roi_image = image.copy()
        roi_draw = ImageDraw.Draw(roi_image)
        diagnostic = observation.diagnostics or {}
        polygon = diagnostic.get("platform_polygon_px", [])
        edge = diagnostic.get("body_side_edge_px", [])
        if len(polygon) >= 3:
            polygon_points = [tuple(map(float, point)) for point in polygon]
            roi_draw.line(polygon_points + [polygon_points[0]], fill=(255, 215, 0), width=3)
        if len(edge) == 2:
            roi_draw.line([tuple(map(float, point)) for point in edge], fill=(255, 0, 0), width=4)
        roi_image.save(directory / "head_platform_roi.png")
        detector_audit = diagnostic.get("detector_audit") or {}
        if detector_audit:
            (directory / "dino_candidate_audit.json").write_text(
                json.dumps(_jsonable(detector_audit), ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8")
            audit_overlay = image.copy()
            audit_draw = ImageDraw.Draw(audit_overlay)
            # Raw threshold-zero model queries are faint; the grounded text
            # candidates, arm boxes, and final decisions are progressively
            # emphasized so the saved artifact explains each gate.
            for query in detector_audit.get("queries", []):
                for candidate in query.get("raw_candidates", []):
                    box = candidate.get("box")
                    if box and candidate.get("postprocess_audit", {}).get("both_thresholds_match_iou", 0) < 0.99:
                        audit_draw.rectangle(tuple(map(float, box)), outline=(130, 130, 130), width=1)
                for candidate in query.get("postprocess_candidates", []):
                    box = candidate.get("box")
                    if box:
                        audit_draw.rectangle(tuple(map(float, box)), outline=(255, 190, 0), width=2)
            for arm in detector_audit.get("arm_candidates", []):
                box = arm.get("box")
                if box:
                    audit_draw.rectangle(tuple(map(float, box)), outline=(0, 190, 255), width=3)
            for candidate in detector_audit.get("filter_audit", []):
                box = candidate.get("box")
                if not box:
                    continue
                validity = candidate.get("candidate_validity")
                color = ((0, 220, 40) if validity == "valid" else
                         (255, 0, 210) if validity == "ambiguous" else (235, 40, 40))
                audit_draw.rectangle(tuple(map(float, box)), outline=color, width=3)
                audit_draw.text((float(box[0]), max(0, float(box[1]) - 13)),
                    f"{candidate.get('label') or candidate.get('raw_phrase')} {candidate.get('confidence', 0):.2f} {validity}",
                    fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            audit_overlay.save(directory / "dino_candidate_overlay.png")
        if len(polygon) >= 3:
            roi_mask = Image.new("L", image.size, 0)
            ImageDraw.Draw(roi_mask).polygon([tuple(map(float, point)) for point in polygon], fill=255)
            roi_crop = image.convert("RGBA")
            roi_crop.putalpha(roi_mask)
            xs, ys = [point[0] for point in polygon], [point[1] for point in polygon]
            roi_crop.crop((max(0, int(min(xs))), max(0, int(min(ys))),
                           min(image.width, int(max(xs)) + 1), min(image.height, int(max(ys)) + 1))) \
                .save(directory / "platform_roi_crop.png")

        overlay = image.convert("RGBA")
        mask_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        track_meta = []
        evidence_by_id = {str(item.get("track_id")): item
                          for item in observation.perception.get("candidate_tracks", [])}
        for index, track in enumerate(diagnostic.get("tracks", [])):
            ident = str(track.get("track_id", index))
            color = ((30, 220, 90, 100), (40, 150, 255, 100), (230, 60, 180, 100))[index % 3]
            mask = track.get("mask")
            mask_path = None
            if mask is not None:
                mask_array = np.asarray(mask, dtype=bool)
                if mask_array.shape == (image.height, image.width):
                    mask_layer_array = np.asarray(mask_layer).copy()
                    mask_layer_array[mask_array] = color
                    mask_layer = Image.fromarray(mask_layer_array, mode="RGBA")
                    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", ident)
                    mask_path = f"sam2_mask_{safe_id}.png"
                    Image.fromarray(mask_array.astype(np.uint8) * 255, mode="L").save(directory / mask_path)
            dino_box = track.get("detector_bbox_xyxy")
            if dino_box:
                draw.rectangle(tuple(map(float, dino_box)), outline=(255, 140, 0, 255), width=3)
            mask_box = track.get("mask_bbox_xyxy")
            if mask_box:
                draw.rectangle(tuple(map(float, mask_box)), outline=(0, 255, 80, 255), width=2)
            draw.text((float((dino_box or mask_box or [0, 0])[0]),
                       max(0, float((dino_box or mask_box or [0, 0])[1]) - 14)),
                      f"{track.get('label')} {ident}", fill=(255, 255, 255, 255),
                      stroke_width=2, stroke_fill=(0, 0, 0, 255))
            evidence = evidence_by_id.get(ident, {})
            track_meta.append({
                "track_id": ident, "label": track.get("label"),
                "confidence": track.get("confidence"),
                "dino_bbox_xyxy": dino_box, "sam2_mask_bbox_xyxy": mask_box,
                "sam2_mask_file": mask_path,
                "mask_available": mask is not None,
                "mask_intersects_platform": (evidence.get("platform_visible_pixels", 0) or 0) > 0,
                "mask_intersects_body_side_region": (evidence.get("body_side_visible_pixels", 0) or 0) > 0,
                "platform_visible_pixels": evidence.get("platform_visible_pixels"),
                "body_side_visible_pixels": evidence.get("body_side_visible_pixels"),
                "mask_body_side_extension": evidence.get("mask_body_side_extension"),
                "partial": evidence.get("partial"), "occluded": evidence.get("occluded"),
                "ambiguous": track.get("ambiguous", evidence.get("ambiguous", False)),
                "detector_diagnostics": track.get("detector_diagnostics", {}),
                "mask_confidence": evidence.get("mask_confidence"),
            })
        overlay = Image.alpha_composite(overlay, mask_layer)
        overlay.convert("RGB").save(directory / "head_detection_overlay.png")

        event_summaries = _event_summaries(cv_events)
        if not event_summaries:
            event_summaries = _event_summaries(diagnostic.get("cv_event_candidates", []))
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "head_received_at_monotonic": frame.received_at,
            "head_header_timestamp": frame.timestamp,
            "head_timestamp_source": frame.timestamp_source,
            "head_image_size": list(image.size),
            "vlm_input_camera_frames": {
                name: {"header_timestamp": camera.timestamp,
                      "received_at_monotonic": camera.received_at,
                      "timestamp_source": camera.timestamp_source}
                for name, camera in observation.images.items() if camera is not None
            },
            "roi": {"platform_polygon_px": polygon, "body_side_edge_px": edge},
            "cv_processing_seconds_before_vlm_submission": diagnostic.get("cv_processing_seconds"),
            "cv_package_detected": bool(track_meta), "cv_tracks": track_meta,
            "cv_event_candidates": event_summaries,
            "detector_audit_file": "dino_candidate_audit.json" if detector_audit else None,
            "detector_overlay_file": "dino_candidate_overlay.png" if detector_audit else None,
            "all_event_candidates": _event_summaries(diagnostic.get("event_candidates", [])),
            "raw_vlm_recommendation": recommendation_fields(recommendation),
            "vlm_inference_timing": getattr(recommendation, "raw", {}).get("inference_timing", {}),
            "outcome": outcome,
            "freshness": freshness,
            "controller_decision": controller_decision,
        }
        (directory / "snapshot.json").write_text(
            json.dumps(_jsonable(metadata), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return str(directory)
    except Exception:
        import logging
        logging.getLogger("kuavo.recovery.diagnostics").exception("Could not save live diagnostic snapshot")
        return None
