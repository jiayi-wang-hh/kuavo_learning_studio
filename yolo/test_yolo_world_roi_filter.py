#!/usr/bin/env python3
"""
YOLO-World + spatial priors + per-side visual temporal tracking.

Pipeline
--------
YOLO-World
  -> generic toy geometric filter
  -> left/right toy assignment
  -> visual tracker (CSRT preferred, KCF/MIL fallback)
  -> constant-velocity prediction only if BOTH detector and visual tracker fail

Track source:
  DETECTED  : current frame has a matched YOLO-World toy detection
  TRACKED   : YOLO missed it, but the image-based tracker found it
  PREDICTED : both YOLO and image tracker failed; short motion-model fallback
  LOST      : fallback exceeded --max-missing
  UNINITIALIZED : that side has never had a valid detection

Important:
- left/right spatial split is used mainly for detection initialization/re-detection.
- once initialized, the visual tracker may move through the center region.
- a track can be initialized late; frame 0 does not need to contain both toys.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2


CLASSES = ["toy", "basket", "robot gripper"]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--model", default="yolov8s-worldv2.pt")
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.03)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/yolo_world_visual_tracking"),
    )

    # Generic toy filtering for YOLO detections.
    p.add_argument("--toy-max-area-ratio", type=float, default=0.07)
    p.add_argument("--toy-max-width-ratio", type=float, default=0.35)
    p.add_argument("--toy-max-height-ratio", type=float, default=0.35)
    p.add_argument("--toy-bottom-limit", type=float, default=0.96)
    p.add_argument("--roi-xmin", type=float, default=0.05)
    p.add_argument("--roi-xmax", type=float, default=0.95)
    p.add_argument("--roi-ymin", type=float, default=0.30)
    p.add_argument("--roi-ymax", type=float, default=0.88)
    p.add_argument("--gripper-overlap-reject", type=float, default=0.55)

    # Side-aware detection assignment.
    p.add_argument("--side-split-x", type=float, default=0.50)
    p.add_argument("--side-margin", type=float, default=0.03)
    p.add_argument("--left-roi-xmax", type=float, default=None)
    p.add_argument("--right-roi-xmin", type=float, default=None)

    # Visual tracker.
    p.add_argument(
        "--visual-tracker",
        choices=["auto", "csrt", "kcf", "mil", "none"],
        default="auto",
        help="Image-based tracker used between YOLO detections.",
    )
    p.add_argument(
        "--tracker-max-jump",
        type=float,
        default=0.12,
        help="Reject tracker update if normalized center jump from previous box exceeds this value.",
    )
    p.add_argument(
        "--tracker-max-area-change",
        type=float,
        default=2.5,
        help="Reject tracker update if bbox area changes by more than this multiplicative factor.",
    )
    p.add_argument(
        "--tracker-min-area-change",
        type=float,
        default=0.35,
        help="Reject tracker update if bbox area shrinks below this fraction of previous area.",
    )
    p.add_argument(
        "--reinit-tracker-every-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reinitialize visual tracker whenever YOLO provides a matched detection.",
    )

    # Motion-model fallback; only used after visual tracker failure.
    p.add_argument(
        "--max-missing",
        type=int,
        default=5,
        help="Maximum consecutive PREDICTED frames after both detector and visual tracker fail.",
    )
    p.add_argument(
        "--assoc-max-dist",
        type=float,
        default=0.20,
        help="Maximum normalized distance between predicted track center and a YOLO candidate.",
    )
    p.add_argument("--velocity-alpha", type=float, default=0.60)
    p.add_argument("--box-alpha", type=float, default=0.80)
    p.add_argument("--prediction-decay", type=float, default=0.85)

    p.add_argument("--draw-raw-toys", action="store_true")
    p.add_argument("--require-both-toys", action="store_true")
    return p.parse_args()


def box_area(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def center_px(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def box_center_norm(
    box: Tuple[float, float, float, float],
    w: int,
    h: int,
) -> Tuple[float, float]:
    cx, cy = center_px(box)
    return cx / w, cy / h


def normalized_center_distance(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
) -> float:
    ax, ay = center_px(a)
    bx, by = center_px(b)
    dx = (ax - bx) / frame_w
    dy = (ay - by) / frame_h
    return (dx * dx + dy * dy) ** 0.5


def intersection_over_a(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = box_area(a)
    return inter / aa if aa > 0 else 0.0


def shift_box(
    box: Tuple[float, float, float, float],
    vx: float,
    vy: float,
    frame_w: int,
    frame_h: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    x1 += vx
    x2 += vx
    y1 += vy
    y2 += vy

    if x1 < 0:
        x1 = 0.0
        x2 = bw
    if y1 < 0:
        y1 = 0.0
        y2 = bh
    if x2 > frame_w:
        x2 = float(frame_w)
        x1 = max(0.0, x2 - bw)
    if y2 > frame_h:
        y2 = float(frame_h)
        y1 = max(0.0, y2 - bh)

    return (x1, y1, x2, y2)


def ema_box(
    old_box: Tuple[float, float, float, float],
    new_box: Tuple[float, float, float, float],
    alpha: float,
) -> Tuple[float, float, float, float]:
    return tuple(
        (1.0 - alpha) * old + alpha * new
        for old, new in zip(old_box, new_box)
    )


def xyxy_to_xywh(box, frame_w=None, frame_h=None):
    x1, y1, x2, y2 = box

    if frame_w is not None:
        x1 = max(0, min(x1, frame_w - 1))
        x2 = max(0, min(x2, frame_w))

    if frame_h is not None:
        y1 = max(0, min(y1, frame_h - 1))
        y2 = max(0, min(y2, frame_h))

    x = int(round(x1))
    y = int(round(y1))
    w = int(round(x2 - x1))
    h = int(round(y2 - y1))

    w = max(1, w)
    h = max(1, h)

    return (x, y, w, h)


def xywh_to_xyxy(box):
    x, y, w, h = box
    return (float(x), float(y), float(x + w), float(y + h))


def toy_rejection_reason(
    box,
    frame_w,
    frame_h,
    gripper_boxes,
    args,
) -> Optional[str]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    area_ratio = (bw * bh) / float(frame_w * frame_h)
    width_ratio = bw / float(frame_w)
    height_ratio = bh / float(frame_h)
    cx, cy = box_center_norm(box, frame_w, frame_h)
    bottom = y2 / frame_h

    if area_ratio > args.toy_max_area_ratio:
        return f"too_large_area={area_ratio:.3f}"
    if width_ratio > args.toy_max_width_ratio:
        return f"too_wide={width_ratio:.3f}"
    if height_ratio > args.toy_max_height_ratio:
        return f"too_tall={height_ratio:.3f}"
    if not (args.roi_xmin <= cx <= args.roi_xmax):
        return f"center_x_outside_roi={cx:.3f}"
    if not (args.roi_ymin <= cy <= args.roi_ymax):
        return f"center_y_outside_roi={cy:.3f}"
    if bottom > args.toy_bottom_limit:
        return f"touches_bottom={bottom:.3f}"

    # Only YOLO detections use this exclusion.
    # We do NOT apply it to the visual tracker, because during a real grasp
    # the toy is expected to overlap the gripper.
    if args.gripper_overlap_reject <= 1.0:
        for gb in gripper_boxes:
            overlap = intersection_over_a(box, gb)
            if overlap > args.gripper_overlap_reject:
                return f"overlap_gripper={overlap:.3f}"

    return None


def side_assignment(
    box,
    frame_w,
    frame_h,
    args,
) -> Tuple[Optional[str], Optional[str]]:
    cx, _ = box_center_norm(box, frame_w, frame_h)

    left_xmax = (
        args.left_roi_xmax
        if args.left_roi_xmax is not None
        else args.side_split_x - args.side_margin
    )
    right_xmin = (
        args.right_roi_xmin
        if args.right_roi_xmin is not None
        else args.side_split_x + args.side_margin
    )

    if cx <= left_xmax:
        return "left_toy", None
    if cx >= right_xmin:
        return "right_toy", None
    return None, f"ambiguous_side_center_x={cx:.3f}"


def get_tracker_creator(name: str):
    """Return (resolved_name, creator) or (None, None)."""
    candidates = [name] if name != "auto" else ["csrt", "kcf", "mil"]

    for candidate in candidates:
        attr = {
            "csrt": "TrackerCSRT_create",
            "kcf": "TrackerKCF_create",
            "mil": "TrackerMIL_create",
        }[candidate]

        creator = getattr(cv2, attr, None)
        if creator is not None:
            return candidate, creator

        legacy = getattr(cv2, "legacy", None)
        if legacy is not None:
            creator = getattr(legacy, attr, None)
            if creator is not None:
                return candidate, creator

    return None, None


def tracker_box_is_sane(
    old_box,
    new_box,
    frame_w,
    frame_h,
    args,
) -> Tuple[bool, str]:
    x1, y1, x2, y2 = new_box

    if x2 <= x1 or y2 <= y1:
        return False, "invalid_box"

    if x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
        return False, "outside_frame"

    jump = normalized_center_distance(
        old_box, new_box, frame_w, frame_h
    )
    if jump > args.tracker_max_jump:
        return False, f"tracker_jump={jump:.3f}"

    old_area = max(box_area(old_box), 1.0)
    ratio = box_area(new_box) / old_area
    if ratio > args.tracker_max_area_change:
        return False, f"tracker_area_growth={ratio:.3f}"
    if ratio < args.tracker_min_area_change:
        return False, f"tracker_area_shrink={ratio:.3f}"

    return True, ""


@dataclass
class SideTrack:
    name: str
    box: Optional[Tuple[float, float, float, float]] = None
    vx: float = 0.0
    vy: float = 0.0

    # Number of consecutive motion-model fallback frames.
    missing: int = 0

    initialized: bool = False
    confidence: float = 0.0
    source: str = "UNINITIALIZED"

    # Image tracker state.
    visual_tracker: object = None
    visual_tracker_name: str = ""
    visual_tracker_ready: bool = False
    tracker_fail_reason: str = ""

    def predicted_box(self, frame_w, frame_h):
        if not self.initialized or self.box is None:
            return None
        return shift_box(
            self.box,
            self.vx,
            self.vy,
            frame_w,
            frame_h,
        )

    def reset_visual_tracker(self):
        self.visual_tracker = None
        self.visual_tracker_ready = False
        self.visual_tracker_name = ""

    def init_visual_tracker(self, frame, box, requested_name):
        if requested_name == "none":
            self.reset_visual_tracker()
            return False

        resolved_name, creator = get_tracker_creator(requested_name)
        if creator is None:
            self.reset_visual_tracker()
            self.tracker_fail_reason = (
                "no_supported_opencv_tracker"
            )
            return False

        tracker = creator()

        bbox_xywh = xyxy_to_xywh(
            box,
            frame.shape[1],
            frame.shape[0],
        )

        try:
            ok = tracker.init(frame, bbox_xywh)
        except cv2.error as exc:
            self.reset_visual_tracker()
            self.tracker_fail_reason = f"tracker_init_cv2_error={exc}"
            return False

        # OpenCV 不同版本：
        # 有的返回 True
        # 有的成功时返回 None
        if ok is False:
            self.reset_visual_tracker()
            self.tracker_fail_reason = "tracker_init_failed"
            return False

        self.visual_tracker = tracker
        self.visual_tracker_name = resolved_name
        self.visual_tracker_ready = True
        self.tracker_fail_reason = ""
        return True


def choose_candidate_for_track(
    track: SideTrack,
    candidates: list[dict],
    frame_w: int,
    frame_h: int,
    assoc_max_dist: float,
) -> Tuple[Optional[dict], Optional[float]]:
    if not candidates:
        return None, None

    if not track.initialized or track.box is None or track.source == "LOST":
        best = max(candidates, key=lambda d: d["conf"])
        return best, None

    pred = track.predicted_box(frame_w, frame_h)
    if pred is None:
        best = max(candidates, key=lambda d: d["conf"])
        return best, None

    scored = [
        (
            normalized_center_distance(
                pred, d["box"], frame_w, frame_h
            ),
            d,
        )
        for d in candidates
    ]
    scored.sort(key=lambda x: x[0])
    best_dist, best = scored[0]

    if best_dist <= assoc_max_dist:
        return best, best_dist

    return None, best_dist


def update_motion_from_observation(
    track: SideTrack,
    observed_box,
    args,
    smooth_box: bool,
):
    if track.box is None or not track.initialized:
        track.box = observed_box
        track.vx = 0.0
        track.vy = 0.0
        return

    old_cx, old_cy = center_px(track.box)
    new_cx, new_cy = center_px(observed_box)

    measured_vx = new_cx - old_cx
    measured_vy = new_cy - old_cy

    track.vx = (
        (1.0 - args.velocity_alpha) * track.vx
        + args.velocity_alpha * measured_vx
    )
    track.vy = (
        (1.0 - args.velocity_alpha) * track.vy
        + args.velocity_alpha * measured_vy
    )

    if smooth_box:
        track.box = ema_box(
            track.box,
            observed_box,
            args.box_alpha,
        )
    else:
        track.box = observed_box


def update_side_track(
    track: SideTrack,
    candidates: list[dict],
    frame,
    frame_w: int,
    frame_h: int,
    args,
) -> dict:
    """
    Priority:
      1. YOLO detection
      2. image-based visual tracker
      3. constant-velocity prediction
      4. LOST
    """
    candidate, assoc_dist = choose_candidate_for_track(
        track,
        candidates,
        frame_w,
        frame_h,
        args.assoc_max_dist,
    )

    # 1) Fresh detector observation.
    if candidate is not None:
        new_box = candidate["box"]

        reinitializing = (
            not track.initialized
            or track.box is None
            or track.source in {"LOST", "UNINITIALIZED"}
        )

        update_motion_from_observation(
            track,
            new_box,
            args,
            smooth_box=not reinitializing,
        )

        track.initialized = True
        track.missing = 0
        track.confidence = float(candidate["conf"])
        track.source = "DETECTED"
        track.tracker_fail_reason = ""

        if (
            args.reinit_tracker_every_detection
            or not track.visual_tracker_ready
            or reinitializing
        ):
            track.init_visual_tracker(
                frame,
                track.box,
                args.visual_tracker,
            )

        return {
            "matched_detection": candidate,
            "assoc_dist": assoc_dist,
            "tracker_ok": track.visual_tracker_ready,
            "tracker_reason": track.tracker_fail_reason,
        }

    # 2) Detector missed -> try the real image tracker.
    if (
        track.initialized
        and track.box is not None
        and track.visual_tracker_ready
        and track.visual_tracker is not None
    ):
        try:
            ok, xywh = track.visual_tracker.update(frame)
        except Exception as exc:
            ok = False
            xywh = None
            track.tracker_fail_reason = (
                f"tracker_exception={type(exc).__name__}"
            )

        if ok and xywh is not None:
            tracked_box = xywh_to_xyxy(xywh)
            sane, reason = tracker_box_is_sane(
                track.box,
                tracked_box,
                frame_w,
                frame_h,
                args,
            )

            if sane:
                update_motion_from_observation(
                    track,
                    tracked_box,
                    args,
                    smooth_box=False,
                )
                track.source = "TRACKED"
                track.confidence = 0.0
                track.missing = 0
                track.tracker_fail_reason = ""

                return {
                    "matched_detection": None,
                    "assoc_dist": assoc_dist,
                    "tracker_ok": True,
                    "tracker_reason": "",
                }

            track.tracker_fail_reason = reason
        else:
            if not track.tracker_fail_reason:
                track.tracker_fail_reason = "tracker_update_failed"

        track.reset_visual_tracker()

    # 3) Both detector and image tracker failed -> short prediction fallback.
    if track.initialized and track.box is not None:
        if track.missing < args.max_missing:
            predicted = track.predicted_box(frame_w, frame_h)
            if predicted is not None:
                track.box = predicted

            track.vx *= args.prediction_decay
            track.vy *= args.prediction_decay
            track.missing += 1
            track.confidence = 0.0
            track.source = "PREDICTED"

            return {
                "matched_detection": None,
                "assoc_dist": assoc_dist,
                "tracker_ok": False,
                "tracker_reason": track.tracker_fail_reason,
            }

        track.source = "LOST"
        track.confidence = 0.0
        track.missing += 1
        track.reset_visual_tracker()

        return {
            "matched_detection": None,
            "assoc_dist": assoc_dist,
            "tracker_ok": False,
            "tracker_reason": track.tracker_fail_reason,
        }

    track.source = "UNINITIALIZED"
    return {
        "matched_detection": None,
        "assoc_dist": assoc_dist,
        "tracker_ok": False,
        "tracker_reason": track.tracker_fail_reason,
    }


def draw_box(frame, box, label, color, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    text_thickness = 2
    (tw, th), _ = cv2.getTextSize(
        label,
        font,
        font_scale,
        text_thickness,
    )
    ty1 = max(0, y1 - th - 8)
    cv2.rectangle(
        frame,
        (x1, ty1),
        (x1 + tw + 6, y1),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 3, y1 - 5),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def main():
    # Keep reusable filtering/tracking helpers importable from the deployment
    # environment, where YOLO itself may run in an external worker Conda env.
    from ultralytics import YOLOWorld
    args = parse_args()

    if not args.video.is_file():
        raise FileNotFoundError(args.video)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = YOLOWorld(args.model)
    model.set_classes(CLASSES)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {args.video}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    frame_h = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    num_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    out_video = (
        args.output_dir
        / f"{args.video.stem}_yoloworld_visual_tracking.mp4"
    )
    out_csv = (
        args.output_dir
        / f"{args.video.stem}_yoloworld_visual_tracking.csv"
    )

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"Cannot create output video: {out_video}"
        )

    csv_file = out_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    )

    fields = [
        "frame",
        "time_s",
        "class",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "accepted",
        "rejection_reason",
        "side_center_x",
        "side_center_y",
        "track_source",
        "missing_frames",
        "vx_px",
        "vy_px",
        "assoc_dist",
        "visual_tracker",
        "tracker_ok",
        "tracker_reason",
    ]
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=fields,
    )
    csv_writer.writeheader()

    resolved_tracker, _ = get_tracker_creator(
        args.visual_tracker
    )
    if args.visual_tracker == "none":
        tracker_msg = "disabled"
    elif resolved_tracker is None:
        tracker_msg = (
            "NOT AVAILABLE -> will use prediction fallback. "
            "Install opencv-contrib-python for CSRT/KCF."
        )
    else:
        tracker_msg = resolved_tracker

    print(f"Input: {args.video}")
    print(f"Frames: {num_frames}")
    print(f"Size: {frame_w}x{frame_h}")
    print(f"FPS: {fps:.3f}")
    print(f"YOLO classes: {CLASSES}")
    print(f"Visual tracker: {tracker_msg}")
    print(
        "Track priority: "
        "DETECTED > TRACKED > PREDICTED > LOST"
    )

    left_track = SideTrack("left_toy")
    right_track = SideTrack("right_toy")

    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Keep an unannotated frame for tracker init/update.
        raw_frame = frame.copy()

        results = model.predict(
            raw_frame,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        result = results[0]

        raw_detections = []
        if result.boxes is not None:
            for det in result.boxes:
                cls_id = int(det.cls[0].item())
                conf = float(det.conf[0].item())
                xyxy = tuple(
                    float(x)
                    for x in det.xyxy[0].tolist()
                )

                if 0 <= cls_id < len(CLASSES):
                    raw_detections.append(
                        {
                            "class": CLASSES[cls_id],
                            "conf": conf,
                            "box": xyxy,
                        }
                    )

        gripper_boxes = [
            d["box"]
            for d in raw_detections
            if d["class"] == "robot gripper"
        ]

        accepted_non_toys = []
        rejected_toys = []
        generic_toys = []

        # Generic geometry filter.
        for d in raw_detections:
            if d["class"] != "toy":
                accepted_non_toys.append(d)
                continue

            reason = toy_rejection_reason(
                d["box"],
                frame_w,
                frame_h,
                gripper_boxes,
                args,
            )

            if reason is None:
                generic_toys.append(d)
            else:
                dd = dict(d)
                dd["reason"] = reason
                rejected_toys.append(dd)

        # Assign valid YOLO toy detections to initial left/right sides.
        left_candidates = []
        right_candidates = []

        for d in generic_toys:
            side, reason = side_assignment(
                d["box"],
                frame_w,
                frame_h,
                args,
            )

            if side is None:
                dd = dict(d)
                dd["reason"] = reason
                rejected_toys.append(dd)
                continue

            dd = dict(d)
            dd["class"] = side

            if side == "left_toy":
                left_candidates.append(dd)
            else:
                right_candidates.append(dd)

        # Update side tracks.
        left_update = update_side_track(
            left_track,
            left_candidates,
            raw_frame,
            frame_w,
            frame_h,
            args,
        )
        right_update = update_side_track(
            right_track,
            right_candidates,
            raw_frame,
            frame_w,
            frame_h,
            args,
        )

        matched_left = left_update[
            "matched_detection"
        ]
        matched_right = right_update[
            "matched_detection"
        ]

        for d in left_candidates:
            if d is not matched_left:
                dd = dict(d)
                dd["reason"] = (
                    "not_selected_by_left_track"
                )
                rejected_toys.append(dd)

        for d in right_candidates:
            if d is not matched_right:
                dd = dict(d)
                dd["reason"] = (
                    "not_selected_by_right_track"
                )
                rejected_toys.append(dd)

        # Draw spatial prior guides.
        rx1 = int(args.roi_xmin * frame_w)
        rx2 = int(args.roi_xmax * frame_w)
        ry1 = int(args.roi_ymin * frame_h)
        ry2 = int(args.roi_ymax * frame_h)

        cv2.rectangle(
            frame,
            (rx1, ry1),
            (rx2, ry2),
            (120, 120, 120),
            1,
        )
        cv2.putText(
            frame,
            "YOLO toy ROI",
            (rx1 + 4, ry1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )

        split_x = int(
            args.side_split_x * frame_w
        )
        margin_l = int(
            (args.side_split_x - args.side_margin)
            * frame_w
        )
        margin_r = int(
            (args.side_split_x + args.side_margin)
            * frame_w
        )

        cv2.line(
            frame,
            (split_x, 0),
            (split_x, frame_h),
            (90, 90, 255),
            1,
        )
        cv2.line(
            frame,
            (margin_l, 0),
            (margin_l, frame_h),
            (90, 90, 160),
            1,
        )
        cv2.line(
            frame,
            (margin_r, 0),
            (margin_r, frame_h),
            (90, 90, 160),
            1,
        )

        # Draw basket / gripper detections.
        for d in accepted_non_toys:
            cls_name = d["class"]

            color = (
                (40, 200, 200)
                if cls_name == "basket"
                else (60, 200, 60)
            )

            draw_box(
                frame,
                d["box"],
                f"{cls_name} {d['conf']:.2f}",
                color,
                2,
            )

            cx, cy = box_center_norm(
                d["box"],
                frame_w,
                frame_h,
            )

            csv_writer.writerow(
                {
                    "frame": frame_id,
                    "time_s": (
                        frame_id / fps
                        if fps > 0
                        else 0
                    ),
                    "class": cls_name,
                    "confidence": (
                        f"{d['conf']:.6f}"
                    ),
                    "x1": f"{d['box'][0]:.2f}",
                    "y1": f"{d['box'][1]:.2f}",
                    "x2": f"{d['box'][2]:.2f}",
                    "y2": f"{d['box'][3]:.2f}",
                    "accepted": True,
                    "rejection_reason": "",
                    "side_center_x": f"{cx:.4f}",
                    "side_center_y": f"{cy:.4f}",
                    "track_source": "",
                    "missing_frames": "",
                    "vx_px": "",
                    "vy_px": "",
                    "assoc_dist": "",
                    "visual_tracker": "",
                    "tracker_ok": "",
                    "tracker_reason": "",
                }
            )

        # Draw left/right toy tracks.
        for track, update, color in (
            (
                left_track,
                left_update,
                (255, 120, 40),
            ),
            (
                right_track,
                right_update,
                (255, 40, 140),
            ),
        ):
            if (
                track.box is None
                or track.source
                in {"UNINITIALIZED", "LOST"}
            ):
                continue

            if track.source == "DETECTED":
                label = (
                    f"{track.name} DET "
                    f"{track.confidence:.2f}"
                )
                thickness = 2

            elif track.source == "TRACKED":
                label = (
                    f"{track.name} TRACK "
                    f"[{track.visual_tracker_name}]"
                )
                thickness = 2

            else:
                label = (
                    f"{track.name} PRED "
                    f"miss={track.missing}"
                )
                thickness = 1

            draw_box(
                frame,
                track.box,
                label,
                color,
                thickness,
            )

            cx, cy = box_center_norm(
                track.box,
                frame_w,
                frame_h,
            )

            assoc_dist = update["assoc_dist"]

            csv_writer.writerow(
                {
                    "frame": frame_id,
                    "time_s": (
                        frame_id / fps
                        if fps > 0
                        else 0
                    ),
                    "class": track.name,
                    "confidence": (
                        f"{track.confidence:.6f}"
                    ),
                    "x1": (
                        f"{track.box[0]:.2f}"
                    ),
                    "y1": (
                        f"{track.box[1]:.2f}"
                    ),
                    "x2": (
                        f"{track.box[2]:.2f}"
                    ),
                    "y2": (
                        f"{track.box[3]:.2f}"
                    ),
                    "accepted": True,
                    "rejection_reason": "",
                    "side_center_x": f"{cx:.4f}",
                    "side_center_y": f"{cy:.4f}",
                    "track_source": track.source,
                    "missing_frames": track.missing,
                    "vx_px": f"{track.vx:.3f}",
                    "vy_px": f"{track.vy:.3f}",
                    "assoc_dist": (
                        ""
                        if assoc_dist is None
                        else f"{assoc_dist:.5f}"
                    ),
                    "visual_tracker": (
                        track.visual_tracker_name
                    ),
                    "tracker_ok": (
                        update["tracker_ok"]
                    ),
                    "tracker_reason": (
                        update["tracker_reason"]
                    ),
                }
            )

        # Optional rejected toy diagnostics.
        for d in rejected_toys:
            if args.draw_raw_toys:
                draw_box(
                    frame,
                    d["box"],
                    (
                        f"REJECT toy "
                        f"{d['conf']:.2f} "
                        f"{d['reason']}"
                    ),
                    (100, 100, 100),
                    1,
                )

            cx, cy = box_center_norm(
                d["box"],
                frame_w,
                frame_h,
            )

            csv_writer.writerow(
                {
                    "frame": frame_id,
                    "time_s": (
                        frame_id / fps
                        if fps > 0
                        else 0
                    ),
                    "class": d.get(
                        "class", "toy"
                    ),
                    "confidence": (
                        f"{d['conf']:.6f}"
                    ),
                    "x1": f"{d['box'][0]:.2f}",
                    "y1": f"{d['box'][1]:.2f}",
                    "x2": f"{d['box'][2]:.2f}",
                    "y2": f"{d['box'][3]:.2f}",
                    "accepted": False,
                    "rejection_reason": (
                        d["reason"]
                    ),
                    "side_center_x": f"{cx:.4f}",
                    "side_center_y": f"{cy:.4f}",
                    "track_source": "",
                    "missing_frames": "",
                    "vx_px": "",
                    "vy_px": "",
                    "assoc_dist": "",
                    "visual_tracker": "",
                    "tracker_ok": "",
                    "tracker_reason": "",
                }
            )

        basket_count = sum(
            d["class"] == "basket"
            for d in accepted_non_toys
        )
        gripper_count = sum(
            d["class"] == "robot gripper"
            for d in accepted_non_toys
        )

        left_valid = left_track.source in {
            "DETECTED",
            "TRACKED",
            "PREDICTED",
        }
        right_valid = right_track.source in {
            "DETECTED",
            "TRACKED",
            "PREDICTED",
        }

        status = (
            f"frame={frame_id} "
            f"L={left_track.source} "
            f"R={right_track.source} "
            f"basket={basket_count} "
            f"gripper={gripper_count}"
        )

        if (
            args.require_both_toys
            and not (left_valid and right_valid)
        ):
            status += " [WARN missing_side_toy]"

        cv2.putText(
            frame,
            status,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        if frame_id % 50 == 0:
            print(status)

        frame_id += 1

    cap.release()
    writer.release()
    csv_file.close()

    print("\nDONE")
    print(f"Annotated video: {out_video}")
    print(f"CSV:             {out_csv}")


if __name__ == "__main__":
    main()
