#!/usr/bin/env python3
"""Offline visual-only benchmark for YOLO/CSRT, YOLO/SAM, DINO/SAM and SAM 3.

The script has no robot-state input by design.  It produces a common track CSV
and annotated MP4, allowing the perception candidates to be compared before
they are connected to the phase-aware failure trigger.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


def load_option_a_baseline():
    """Load the established Option-A implementation without duplicating it."""
    source = Path(__file__).resolve().parents[1] / "test_yolo_world_roi_filter.py"
    spec = importlib.util.spec_from_file_location("option_a_baseline", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Option A baseline: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Observation:
    track_id: int
    label: str
    box: tuple[float, float, float, float]
    confidence: float
    source: str
    mask: np.ndarray | None = None
    rejection_reason: str = ""


@dataclass
class TrackState:
    label: str
    tracker: Any
    box: tuple[float, float, float, float]
    confidence: float
    occlusion_frames: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--backend", required=True, choices=("yolo_csrt", "yolo_sam", "grounding_dino_sam", "sam3"))
    parser.add_argument("--input-dir", type=Path, default="/media/data/jiayi/dataset/toy_success")
    parser.add_argument(
        "--prompt", default="toy",
        help="Open-vocabulary name of the moving target(s). For this setup use `toy`.",
    )
    parser.add_argument(
        "--context-classes", default="basket,robot arm,robot gripper",
        help="Known static/robot scene classes used for detection context, not motion targets.",
    )
    parser.add_argument(
        "--scene", choices=("toy_table", "generic"), default="toy_table",
        help="toy_table assigns the two target instances to left_toy/right_toy by image x position.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0", help="Ultralytics device (e.g. 0, cpu).")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.03)
    parser.add_argument("--dino-conf", type=float, default=0.2)
    parser.add_argument("--basket-conf", type=float, default=0.02, help="Lower detection threshold used only for static basket anchors.")
    parser.add_argument("--basket-init-frames", type=int, default=8, help="Consistent observations required before freezing a basket anchor.")
    parser.add_argument("--basket-association-iou", type=float, default=0.20)
    parser.add_argument("--basket-max-count", type=int, default=2)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--yolo-model", default="yolov8s-worldv2.pt")
    parser.add_argument("--sam-model", default="sam2.1_b.pt")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_s.yaml", help="SAM 2.1 model config, relative to the sam2 checkout or absolute.")
    parser.add_argument("--sam2-checkpoint", default="checkpoints/sam2.1_hiera_small.pt", help="SAM 2.1 checkpoint path.")
    parser.add_argument("--sam2-vos-optimized", action="store_true", help="Enable SAM 2.1 video VOS compilation when supported.")
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument("--sam2-require-two-toys", action=argparse.BooleanOptionalAction, default=False, help="Fail a rollout when either left_toy or right_toy has no valid YOLO initialization box.")
    else:
        parser.add_argument("--sam2-require-two-toys", action="store_true", default=False, help="Fail a rollout when either left_toy or right_toy has no valid YOLO initialization box.")
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--dino-box-threshold", type=float, default=0.20)
    parser.add_argument("--dino-text-threshold", type=float, default=0.20)
    parser.add_argument("--dino-nms-iou", type=float, default=0.50)
    parser.add_argument("--association-iou", type=float, default=0.25)
    parser.add_argument("--detect-every", type=int, default=1, help="A: detector refresh interval.")
    parser.add_argument("--toy-max-area-ratio", type=float, default=0.07)
    parser.add_argument("--toy-min-aspect", type=float, default=0.35)
    parser.add_argument("--toy-max-aspect", type=float, default=2.8)
    parser.add_argument("--gripper-overlap-iou", type=float, default=0.30)
    parser.add_argument("--gripper-association-iou", type=float, default=0.20)
    parser.add_argument("--occlusion-hold-frames", type=int, default=12)
    parser.add_argument("--init-confirm-frames", type=int, default=2)
    # Option A deliberately uses the same defaults as
    # test_yolo_world_roi_filter.py.
    parser.add_argument("--toy-max-width-ratio", type=float, default=0.35)
    parser.add_argument("--toy-max-height-ratio", type=float, default=0.35)
    parser.add_argument("--toy-bottom-limit", type=float, default=0.96)
    parser.add_argument("--roi-xmin", type=float, default=0.05)
    parser.add_argument("--roi-xmax", type=float, default=0.95)
    parser.add_argument("--roi-ymin", type=float, default=0.30)
    parser.add_argument("--roi-ymax", type=float, default=0.88)
    parser.add_argument("--gripper-overlap-reject", type=float, default=0.55)
    parser.add_argument("--side-split-x", type=float, default=0.50)
    parser.add_argument("--side-margin", type=float, default=0.03)
    parser.add_argument("--left-roi-xmax", type=float, default=None)
    parser.add_argument("--right-roi-xmin", type=float, default=None)
    parser.add_argument("--visual-tracker", choices=("auto", "csrt", "kcf", "mil", "none"), default="auto")
    parser.add_argument("--tracker-max-jump", type=float, default=0.12)
    parser.add_argument("--tracker-max-area-change", type=float, default=2.5)
    parser.add_argument("--tracker-min-area-change", type=float, default=0.35)
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument("--reinit-tracker-every-detection", action=argparse.BooleanOptionalAction, default=True)
    else:
        parser.add_argument("--reinit-tracker-every-detection", action="store_true", default=True)
        parser.add_argument("--no-reinit-tracker-every-detection", action="store_false", dest="reinit_tracker_every_detection")
    parser.add_argument("--max-missing", type=int, default=5)
    parser.add_argument("--assoc-max-dist", type=float, default=0.20)
    parser.add_argument("--velocity-alpha", type=float, default=0.60)
    parser.add_argument("--box-alpha", type=float, default=0.80)
    parser.add_argument("--prediction-decay", type=float, default=0.85)
    return parser.parse_args()


def xyxy_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return inter / union if union else 0.0


def mask_iou(a: np.ndarray | None, b: np.ndarray | None, box_a, box_b) -> float:
    if a is None or b is None:
        return xyxy_iou(box_a, box_b)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def csrt_creator():
    creator = getattr(cv2, "TrackerCSRT_create", None)
    if creator is None and hasattr(cv2, "legacy"):
        creator = getattr(cv2.legacy, "TrackerCSRT_create", None)
    if creator is None:
        raise RuntimeError("CSRT unavailable: install opencv-contrib-python in this environment.")
    return creator


class YoloDetector:
    def __init__(self, args: argparse.Namespace):
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:
            raise RuntimeError("YOLO backend needs `pip install ultralytics`.") from exc
        self.model = YOLOWorld(args.yolo_model)
        self.target_labels = [p.strip() for p in args.prompt.split(",") if p.strip()]
        self.context_labels = [p.strip() for p in args.context_classes.split(",") if p.strip()]
        self.labels = self.target_labels + self.context_labels
        self.model.set_classes(self.labels)
        self.args = args

    def detect(self, frame: np.ndarray) -> list[Observation]:
        result = self.model.predict(frame, conf=min(self.args.conf, self.args.basket_conf), iou=self.args.iou, device=self.args.device, verbose=False)[0]
        observations = []
        for box in result.boxes:
            score = float(box.conf[0])
            cls = int(box.cls[0])
            xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
            if self.labels[cls] == "basket" and score < self.args.basket_conf:
                continue
            if self.labels[cls] != "basket" and score < self.args.conf:
                continue
            # Context is retained in the CSV/video. Only target labels enter
            # temporal tracking below, so an arm/gripper box cannot become a
            # false "object motion" signal.
            observations.append(Observation(-1, self.labels[cls], xyxy, score, "DETECTED"))
        return observations

    def is_target(self, label: str) -> bool:
        return label in self.target_labels


def name_toy_instances(observations: list[Observation], frame_width: int) -> list[Observation]:
    """Map the two same-class toy instances to stable semantic sides per frame.

    This is deliberately geometric rather than a text prompt such as "left toy":
    the latter is often interpreted inconsistently by open-vocabulary detectors.
    """
    toys = [o for o in observations if o.label == "toy"]
    if len(toys) >= 2:
        for index, obs in enumerate(sorted(toys, key=lambda o: (o.box[0] + o.box[2]) / 2)):
            obs.label = "left_toy" if index == 0 else "right_toy" if index == 1 else "toy_extra"
    return observations


def box_center_inside(box, container, margin: float = 0.10) -> bool:
    """Whether a box centre is inside an expanded container box."""
    x1, y1, x2, y2 = container
    width, height = x2 - x1, y2 - y1
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return x1 - margin * width <= cx <= x2 + margin * width and y1 - margin * height <= cy <= y2 + margin * height


def scene_filter(detections, frame_shape, args):
    """Reject implausible *new* toy candidates while retaining context boxes.

    A toy behind a gripper is a legitimate temporary disappearance; this gate
    only rejects a candidate that looks like a newly detected gripper, rather
    than deleting an established track because it overlaps the gripper.
    """
    height, width = frame_shape[:2]
    if args.scene != "toy_table":
        return detections, [], []
    grippers = [o for o in detections if o.label == "robot gripper"]
    tables = [o for o in detections if o.label == "table"]
    kept, rejected = [], []
    for obs in detections:
        if obs.label != "toy":
            kept.append(obs)
            continue
        x1, y1, x2, y2 = obs.box
        area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height)
        aspect = (x2 - x1) / max(1.0, y2 - y1)
        reason = ""
        if area_ratio > args.toy_max_area_ratio:
            reason = "toy_area_too_large"
        elif not args.toy_min_aspect <= aspect <= args.toy_max_aspect:
            reason = "toy_aspect_out_of_range"
        elif tables and not any(box_center_inside(obs.box, table.box) for table in tables):
            reason = "toy_center_outside_table"
        elif any(xyxy_iou(obs.box, gripper.box) >= args.gripper_overlap_iou for gripper in grippers):
            reason = "new_toy_overlaps_gripper"
        if reason:
            obs.source, obs.rejection_reason = "REJECTED", reason
            rejected.append(obs)
        else:
            kept.append(obs)
    return kept, rejected, grippers


class SamBoxSegmenter:
    """SAM-on-box adapter; uses Ultralytics' stable SAM interface."""
    def __init__(self, args: argparse.Namespace):
        try:
            from ultralytics import SAM
        except ImportError as exc:
            raise RuntimeError("SAM backend needs `pip install ultralytics`.") from exc
        self.model, self.args = SAM(args.sam_model), args

    def apply(self, frame: np.ndarray, observations: list[Observation]) -> list[Observation]:
        if not observations:
            return observations
        boxes = [list(o.box) for o in observations]
        result = self.model(frame, bboxes=boxes, device=self.args.device, verbose=False)[0]
        masks = result.masks.data.cpu().numpy().astype(bool) if result.masks is not None else []
        for index, obs in enumerate(observations):
            if index < len(masks):
                obs.mask = masks[index]
        return observations


class GreedyAssociator:
    def __init__(self, threshold: float, max_tracks: int | None = None):
        self.max_tracks = max_tracks
        self.threshold, self.next_id, self.previous = threshold, 1, []

    def assign(self, detections: list[Observation]) -> list[Observation]:
        remaining = set(range(len(self.previous)))
        for det in detections:
            candidates = [(mask_iou(det.mask, old.mask, det.box, old.box), i) for i, old in enumerate(self.previous) if i in remaining]
            score, index = max(candidates, default=(0.0, -1))
            if score >= self.threshold:
                det.track_id = self.previous[index].track_id
                det.label = self.previous[index].label
                remaining.remove(index)
            elif self.max_tracks is not None and len(self.previous) >= self.max_tracks:
                det.source, det.rejection_reason = "REJECTED", "two_target_cap_or_unassociated_toy"
            else:
                det.track_id, self.next_id = self.next_id, self.next_id + 1
        self.previous = [det for det in detections if det.source != "REJECTED"]
        return detections


@dataclass
class MaskTrack:
    track_id: int
    label: str
    box: tuple[float, float, float, float]
    confidence: float
    mask: np.ndarray | None
    missing: int = 0


class GripperIdentityAssigner:
    """Turn generic detector instances into persistent left/right identities.

    The detector is always prompted with ``robot gripper``.  ``left`` and
    ``right`` are assigned only once, from the first two simultaneously visible
    instances, then inherited by temporal box-IoU association.
    """
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.tracks: dict[int, tuple[str, tuple[float, float, float, float]]] = {}

    def assign(self, observations: list[Observation]) -> list[Observation]:
        grippers = [obs for obs in observations if obs.label == "robot gripper" and obs.source == "DETECTED"]
        if not self.tracks and len(grippers) >= 2:
            ordered = sorted(grippers, key=lambda item: (item.box[0] + item.box[2]) / 2)
            for index, obs in enumerate(ordered[:2]):
                identity = "left_gripper" if index == 0 else "right_gripper"
                self.tracks[index + 1] = (identity, obs.box)
                obs.track_id, obs.label = index + 1, identity
            for obs in ordered[2:]:
                obs.source, obs.rejection_reason = "REJECTED", "unassociated_or_extra_gripper"
            return observations
        available = set(self.tracks)
        for obs in grippers:
            score, track_id = max(((xyxy_iou(obs.box, state[1]), candidate_id) for candidate_id, state in self.tracks.items() if candidate_id in available), default=(0.0, -1))
            if score >= self.threshold:
                identity, _ = self.tracks[track_id]
                self.tracks[track_id] = (identity, obs.box)
                obs.track_id, obs.label = track_id, identity
                available.remove(track_id)
            else:
                obs.source, obs.rejection_reason = "REJECTED", "unassociated_or_extra_gripper"
        return observations


@dataclass
class BasketAnchor:
    box: tuple[float, float, float, float]
    history: deque
    confirmations: int = 1
    frozen: bool = False


class StaticBasketAnchors:
    """Confirm basket boxes temporally, median-fuse them, then keep them static."""
    def __init__(self, args):
        self.args = args
        self.anchors: list[BasketAnchor] = []

    @staticmethod
    def fused_box(history):
        return tuple(float(v) for v in np.median(np.asarray(history), axis=0))

    def apply(self, observations: list[Observation]) -> list[Observation]:
        candidates = [o for o in observations if o.label == "basket" and o.source == "DETECTED"]
        other = [o for o in observations if o not in candidates]
        used, output = set(), []
        for candidate in candidates:
            score, index = max(((xyxy_iou(candidate.box, anchor.box), i) for i, anchor in enumerate(self.anchors) if i not in used), default=(0.0, -1))
            if score < self.args.basket_association_iou:
                if len(self.anchors) >= self.args.basket_max_count:
                    candidate.source, candidate.rejection_reason = "REJECTED", "extra_or_unassociated_basket"
                    other.append(candidate)
                    continue
                anchor = BasketAnchor(candidate.box, deque([candidate.box], maxlen=self.args.basket_init_frames))
                self.anchors.append(anchor); index = len(self.anchors) - 1
            else:
                anchor = self.anchors[index]
                if not anchor.frozen:
                    anchor.history.append(candidate.box)
                    anchor.confirmations += 1
                    anchor.box = self.fused_box(anchor.history)
                    if anchor.confirmations >= self.args.basket_init_frames:
                        anchor.frozen = True
            used.add(index)
        for index, anchor in enumerate(self.anchors):
            label = f"basket_{index + 1}"
            source = "STATIC_ANCHOR" if anchor.frozen else "BASKET_CONFIRMING"
            output.append(Observation(-(index + 1), label, anchor.box, 1.0, source))
        return other + output


class TemporalMaskBackend:
    """B/C: geometry-gated, two-target, temporally persistent SAM tracks."""
    def __init__(self, args, detector):
        self.args, self.detector = args, detector
        self.segmenter = SamBoxSegmenter(args)
        self.tracks: dict[int, MaskTrack] = {}
        self.next_id = 1
        self.gripper_ids = GripperIdentityAssigner(args.gripper_association_iou)
        self.basket_anchors = StaticBasketAnchors(args)

    def process(self, frame, frame_index):
        detections, rejected, grippers = scene_filter(self.detector.detect(frame), frame.shape, self.args)
        if self.args.scene == "toy_table" and not self.tracks:
            name_toy_instances(detections, frame.shape[1])
        targets = [d for d in detections if self.detector.is_target(d.label) or d.label in {"left_toy", "right_toy"}]
        context = [d for d in detections if d not in targets]
        if self.args.scene == "toy_table" and not self.tracks and not {d.label for d in targets} >= {"left_toy", "right_toy"}:
            for d in targets:
                d.source, d.rejection_reason = "REJECTED", "awaiting_two_initial_toys"
            return context + rejected + targets
        targets = self.segmenter.apply(frame, targets)
        pairs = []
        for index, det in enumerate(targets):
            for track_id, track in self.tracks.items():
                pairs.append((mask_iou(det.mask, track.mask, det.box, track.box), index, track_id))
        matched_detections, matched_tracks, output = set(), set(), []
        for score, index, track_id in sorted(pairs, reverse=True):
            if score < self.args.association_iou or index in matched_detections or track_id in matched_tracks:
                continue
            det, old = targets[index], self.tracks[track_id]
            det.track_id, det.label, det.source = track_id, old.label, "DETECTED"
            self.tracks[track_id] = MaskTrack(track_id, old.label, det.box, det.confidence, det.mask)
            output.append(det); matched_detections.add(index); matched_tracks.add(track_id)
        for index, det in enumerate(targets):
            if index in matched_detections:
                continue
            if self.args.scene == "toy_table" and len(self.tracks) >= 2:
                det.source, det.rejection_reason = "REJECTED", "two_target_cap_or_unassociated_toy"
                rejected.append(det); continue
            if self.args.scene == "toy_table" and det.label not in {"left_toy", "right_toy"}:
                det.source, det.rejection_reason = "REJECTED", "missing_initial_side_identity"
                rejected.append(det); continue
            det.track_id = self.next_id; self.next_id += 1
            self.tracks[det.track_id] = MaskTrack(det.track_id, det.label, det.box, det.confidence, det.mask)
            output.append(det)
        for track_id, track in list(self.tracks.items()):
            if track_id in matched_tracks or any(o.track_id == track_id for o in output):
                continue
            if track.missing >= self.args.max_missing:
                output.append(Observation(track_id, track.label, track.box, 0.0, "LOST"))
                del self.tracks[track_id]; continue
            track.missing += 1
            occluded = any(xyxy_iou(track.box, g.box) >= self.args.gripper_overlap_iou for g in grippers)
            output.append(Observation(track_id, track.label, track.box, track.confidence, "OCCLUDED_BY_GRIPPER" if occluded else "PREDICTED", track.mask))
        return self.basket_anchors.apply(self.gripper_ids.assign(output + context + rejected))


class BaselineYoloCsrtBackend:
    """Adapter around test_yolo_world_roi_filter.py's exact A state machine."""
    def __init__(self, args):
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:
            raise RuntimeError("YOLO backend needs `pip install ultralytics`.") from exc
        self.baseline = load_option_a_baseline()
        self.args = args
        self.model = YOLOWorld(args.yolo_model)
        self.model.set_classes(self.baseline.CLASSES)
        self.left = self.baseline.SideTrack("left_toy")
        self.right = self.baseline.SideTrack("right_toy")
        self.gripper_ids = GripperIdentityAssigner(args.gripper_association_iou)
        self.basket_anchors = StaticBasketAnchors(args)

    def process(self, frame, frame_index):
        result = self.model.predict(frame, conf=min(self.args.conf, self.args.basket_conf), iou=self.args.iou, device=self.args.device, verbose=False)[0]
        raw, generic, rejected = [], [], []
        if result.boxes is not None:
            for det in result.boxes:
                cls_id = int(det.cls[0].item())
                if 0 <= cls_id < len(self.baseline.CLASSES):
                    cls_name, confidence = self.baseline.CLASSES[cls_id], float(det.conf[0].item())
                    if confidence < (self.args.basket_conf if cls_name == "basket" else self.args.conf):
                        continue
                    raw.append({"class": cls_name, "conf": confidence, "box": tuple(float(v) for v in det.xyxy[0].tolist())})
        frame_h, frame_w = frame.shape[:2]
        grippers = [d["box"] for d in raw if d["class"] == "robot gripper"]
        context = []
        for detection in raw:
            if detection["class"] != "toy":
                context.append(detection)
                continue
            reason = self.baseline.toy_rejection_reason(detection["box"], frame_w, frame_h, grippers, self.args)
            if reason is None:
                generic.append(detection)
            else:
                rejected.append({**detection, "reason": reason})

        sides = {"left_toy": [], "right_toy": []}
        for detection in generic:
            side, reason = self.baseline.side_assignment(detection["box"], frame_w, frame_h, self.args)
            if side is None:
                rejected.append({**detection, "reason": reason})
            else:
                sides[side].append({**detection, "class": side})

        updates = [
            (self.left, self.baseline.update_side_track(self.left, sides["left_toy"], frame, frame_w, frame_h, self.args), 1),
            (self.right, self.baseline.update_side_track(self.right, sides["right_toy"], frame, frame_w, frame_h, self.args), 2),
        ]
        for candidates, (_, update, _) in zip((sides["left_toy"], sides["right_toy"]), updates):
            matched = update["matched_detection"]
            rejected.extend({**d, "reason": "not_selected_by_side_track"} for d in candidates if d is not matched)

        observations = [Observation(-1, d["class"], d["box"], d["conf"], "DETECTED") for d in context]
        for track, _, track_id in updates:
            if track.box is not None and track.source not in {"UNINITIALIZED", "LOST"}:
                observations.append(Observation(track_id, track.name, track.box, track.confidence, track.source))
        observations.extend(Observation(-1, d["class"], d["box"], d["conf"], "REJECTED", rejection_reason=d["reason"]) for d in rejected)
        return self.basket_anchors.apply(self.gripper_ids.assign(observations))


class YoloCsrtBackend:
    def __init__(self, args):
        self.detector, self.args, self.creator = YoloDetector(args), args, csrt_creator()
        self.tracks: dict[int, TrackState] = {}
        self.pending: dict[str, tuple[tuple[float, float, float, float], int]] = {}
        self.next_id = 1
        self.identities_initialized = False

    def process(self, frame: np.ndarray, frame_index: int) -> list[Observation]:
        if frame_index % self.args.detect_every == 0:
            detections, rejected, grippers = scene_filter(self.detector.detect(frame), frame.shape, self.args)
            # Assign left/right exactly once, when the two physical toys are
            # first seen. Later frames inherit identity through association;
            # they must not swap simply because their image x order changes.
            if self.args.scene == "toy_table" and (not self.identities_initialized or not self.tracks):
                name_toy_instances(detections, frame.shape[1])
                self.identities_initialized = any(det.label == "left_toy" for det in detections) and any(det.label == "right_toy" for det in detections)
            targets = [det for det in detections if self.detector.is_target(det.label) or det.label in {"left_toy", "right_toy"}]
            unmatched = set(self.tracks)
            accepted = []
            for det in targets:
                # The detector returns generic "toy" every frame. Match
                # against either existing toy first, then restore that track's
                # initial left/right identity below.
                candidates = [(xyxy_iou(det.box, stored.box), tid) for tid, stored in self.tracks.items() if tid in unmatched]
                score, tid = max(candidates, default=(0.0, -1))
                if score < self.args.association_iou:
                    if self.args.scene == "toy_table" and self.identities_initialized and self.tracks:
                        det.source, det.rejection_reason = "REJECTED", "two_target_cap_or_unassociated_toy"
                        rejected.append(det)
                        continue
                    old_box, count = self.pending.get(det.label, (det.box, 0))
                    count = count + 1 if xyxy_iou(det.box, old_box) >= self.args.association_iou else 1
                    self.pending[det.label] = (det.box, count)
                    if count < self.args.init_confirm_frames:
                        det.source, det.rejection_reason = "REJECTED", "awaiting_temporal_confirmation"
                        rejected.append(det)
                        continue
                    tid, self.next_id = self.next_id, self.next_id + 1
                    self.pending.pop(det.label, None)
                else:
                    unmatched.discard(tid)
                    det.label = self.tracks[tid].label
                x1, y1, x2, y2 = det.box
                tracker = self.creator()
                x = int(round(float(x1)))
                y = int(round(float(y1)))
                w = int(round(float(x2 - x1)))
                h = int(round(float(y2 - y1)))

                if w <= 1 or h <= 1:
                    continue

                tracker = self.creator()
                tracker.init(frame, (x, y, w, h))
                self.tracks[tid] = TrackState(det.label, tracker, det.box, det.confidence)
                accepted.append(Observation(tid, det.label, det.box, det.confidence, "DETECTED"))
            held = []
            for tid in unmatched:
                track = self.tracks[tid]
                is_occluded = any(xyxy_iou(track.box, g.box) >= self.args.gripper_overlap_iou for g in grippers)
                if is_occluded and track.occlusion_frames < self.args.occlusion_hold_frames:
                    track.occlusion_frames += 1
                    ok, (x, y, w, h) = track.tracker.update(frame)
                    if ok:
                        track.box = (float(x), float(y), float(x + w), float(y + h))
                    held.append(Observation(tid, track.label, track.box, track.confidence, "OCCLUDED_BY_GRIPPER"))
                else:
                    del self.tracks[tid]
            context = [det for det in detections if det not in targets]
            return accepted + held + context + rejected
        output = []
        for tid, track in list(self.tracks.items()):
            ok, (x, y, w, h) = track.tracker.update(frame)
            if ok:
                box = (float(x), float(y), float(x+w), float(y+h))
                track.box = box
                output.append(Observation(tid, track.label, box, track.confidence, "TRACKED"))
        return output


class DetectorSamBackend:
    def __init__(self, args, detector):
        self.detector, self.segmenter = detector, SamBoxSegmenter(args)
        self.associator = GreedyAssociator(args.association_iou, max_tracks=2 if args.scene == "toy_table" else None)
        self.identities_initialized = False
    def process(self, frame, frame_index):
        detections, rejected, _ = scene_filter(self.detector.detect(frame), frame.shape, self.detector.args)
        if self.detector.args.scene == "toy_table" and (not self.identities_initialized or not self.associator.previous):
            name_toy_instances(detections, frame.shape[1])
            self.identities_initialized = any(det.label == "left_toy" for det in detections) and any(det.label == "right_toy" for det in detections)
        targets = [det for det in detections if self.detector.is_target(det.label) or det.label in {"left_toy", "right_toy"}]
        context = [det for det in detections if det not in targets]
        return self.associator.assign(self.segmenter.apply(frame, targets)) + context + rejected


class GroundingDinoDetector:
    def __init__(self, args):
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Grounding DINO needs `pip install transformers torch torchvision`.") from exc
        self.torch, self.processor = torch, AutoProcessor.from_pretrained(args.dino_model)
        # self.model = AutoModelForZeroShotObjectDetection.from_pretrained(args.dino_model).to(args.device).eval()
        if str(args.device).isdigit():
            self.device = f"cuda:{args.device}"
        else:
            self.device = args.device

        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            args.dino_model
        ).to(self.device).eval()
        self.target_labels = [p.strip() for p in args.prompt.split(",") if p.strip()]
        self.context_labels = [p.strip() for p in args.context_classes.split(",") if p.strip()]
        self.labels = self.target_labels + self.context_labels
        self.device = args.device if args.device in {"cpu", "mps"} or str(args.device).startswith("cuda") else f"cuda:{args.device}"
        # self.conf = min(args.conf, args.basket_conf)
        self.conf = args.dino_conf
        self.args = args
    def detect(self, frame):
        from PIL import Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(
            images=image, 
            text=". ".join(self.labels) + ".", 
            return_tensors="pt"
        ).to(self.device)
        with self.torch.no_grad(): outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=self.conf, text_threshold=self.conf, target_sizes=[image.size[::-1]])[0]
        observations = []
        for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            label, score = str(label), float(score)
            if score < (self.args.basket_conf if label == "basket" else self.args.conf):
                continue
            observations.append(Observation(-1, label, tuple(float(v) for v in box.tolist()), score, "DETECTED"))
        return observations

    def is_target(self, label: str) -> bool:
        return label in self.target_labels


def build_backend(args):
    if args.backend == "yolo_csrt": return BaselineYoloCsrtBackend(args)
    if args.backend == "yolo_sam": return TemporalMaskBackend(args, YoloDetector(args))
    if args.backend == "grounding_dino_sam": return TemporalMaskBackend(args, GroundingDinoDetector(args))
    raise NotImplementedError("SAM 3 is a video-session backend; run sam3_adapter.py for option D.")


def draw(frame, observations):
    canvas = frame.copy()
    for obs in observations:
        color = (50, 220, 50) if obs.source == "DETECTED" else (0, 180, 255)
        if obs.mask is not None:
            overlay = canvas.copy(); overlay[obs.mask] = color; canvas = cv2.addWeighted(canvas, 0.70, overlay, 0.30, 0)
        x1, y1, x2, y2 = map(int, obs.box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"{obs.label} #{obs.track_id} {obs.confidence:.2f} {obs.source}", (x1, max(16, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    return canvas


def run_one_video(args, video_path, backend):
    # rollout01_head.mp4 -> rollout01
    rollout_name = video_path.stem.removesuffix("_head")

    # 每个 rollout 单独保存，防止覆盖
    output_dir = args.output_dir / rollout_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Processing {video_path.name} ===")
    print(f"Output: {output_dir}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_dir / "annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / args.stride,
        (width, height),
    )

    fields = [
        "frame",
        "time_s",
        "track_id",
        "label",
        "confidence",
        "source",
        "x1",
        "y1",
        "x2",
        "y2",
        "mask_area_px",
        "rejection_reason",
        "latency_ms",
    ]

    source_counts = {}
    rows = 0
    frame_index = 0
    processed_frames = 0

    with (output_dir / "tracks.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:

        csv_writer = csv.DictWriter(
            fh,
            fieldnames=fields,
        )
        csv_writer.writeheader()

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            if frame_index % args.stride:
                frame_index += 1
                continue

            started = time.perf_counter()

            observations = backend.process(
                frame,
                frame_index,
            )

            latency = (
                time.perf_counter() - started
            ) * 1000

            for obs in observations:
                source_counts[obs.source] = (
                    source_counts.get(obs.source, 0) + 1
                )
                rows += 1

                csv_writer.writerow(
                    dict(
                        frame=frame_index,
                        time_s=frame_index / fps,
                        track_id=obs.track_id,
                        label=obs.label,
                        confidence=f"{obs.confidence:.5f}",
                        source=obs.source,
                        x1=f"{obs.box[0]:.2f}",
                        y1=f"{obs.box[1]:.2f}",
                        x2=f"{obs.box[2]:.2f}",
                        y2=f"{obs.box[3]:.2f}",
                        mask_area_px=(
                            int(obs.mask.sum())
                            if obs.mask is not None
                            else ""
                        ),
                        rejection_reason=obs.rejection_reason,
                        latency_ms=f"{latency:.2f}",
                    )
                )

            writer.write(
                draw(frame, observations)
            )

            processed_frames += 1
            frame_index += 1

    cap.release()
    writer.release()

    summary = {
        "backend": args.backend,
        "video": str(video_path),
        "rollout": rollout_name,
        "prompt": args.prompt,
        "fps": fps,
        "stride": args.stride,
        "processed_frames": processed_frames,
        "track_rows": rows,
        "source_counts": source_counts,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))

    return summary


def main():
    args = parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    video_paths = sorted(args.input_dir.glob("*_head.mp4"))

    if not video_paths:
        raise FileNotFoundError(
            f"No *_head.mp4 videos found in {args.input_dir}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Found {len(video_paths)} head-camera videos "
        f"in {args.input_dir}"
    )

    # Video-propagation backends operate on an entire rollout at once.
    if args.backend in {"yolo_sam", "grounding_dino_sam", "sam3"}:
        if args.backend == "yolo_sam":
            from sam2_video_adapter import run_yolo_sam2 as run_video_backend
        elif args.backend == "grounding_dino_sam":
            from sam2_video_adapter import run_dino_sam2 as run_video_backend
        else:
            from sam3_adapter import run_sam3 as run_video_backend

        for i, video_path in enumerate(
            video_paths,
            start=1,
        ):
            print(
                f"\n[{i}/{len(video_paths)}] "
                f"{video_path.name}"
            )

            rollout_name = (
                video_path.stem.removesuffix("_head")
            )

            output_dir = (
                args.output_dir / rollout_name
            )
            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Give the video backend its current rollout and output directory.
            args.video = video_path
            original_output_dir = args.output_dir
            args.output_dir = output_dir

            run_video_backend(args)

            args.output_dir = original_output_dir

        return

    summaries = []

    for i, video_path in enumerate(
        video_paths,
        start=1,
    ):
        backend = build_backend(args)
        print(
            f"\n[{i}/{len(video_paths)}] "
            f"{video_path.name}"
        )

        summary = run_one_video(
            args=args,
            video_path=video_path,
            backend=backend,
        )

        summaries.append(summary)

    # 保存整个 batch 的 summary
    batch_summary = {
        "backend": args.backend,
        "input_dir": str(args.input_dir),
        "num_videos": len(video_paths),
        "videos": summaries,
    }

    (
        args.output_dir / "batch_summary.json"
    ).write_text(
        json.dumps(
            batch_summary,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n=== DONE ===")
    print(
        f"Processed {len(video_paths)} videos."
    )
    print(
        f"Results: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
