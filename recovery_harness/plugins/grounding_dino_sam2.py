"""Live detector plugin reusing the project's Grounding DINO detector and SAM 2.1.

Grounding DINO refreshes every N unique camera frames. Between refreshes SAM 2.1
uses its previous low-resolution mask prompt to track each visible package on
the current image. No checkpoint is downloaded by this plugin.
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from ..models import Detection
from ..detection import GroundingDinoPromptRunner, arm_box_geometry

log = logging.getLogger("kuavo.recovery.grounding_dino_sam2")


def _load_existing_dino_factory():
    repo_root = Path(__file__).resolve().parents[2]
    source = repo_root / "yolo" / "motion_tracking_benchmark" / "sam2_video_adapter.py"
    if not source.is_file():
        raise FileNotFoundError(f"Existing Grounding DINO + SAM2 adapter source is missing: {source}")
    spec = importlib.util.spec_from_file_location("kuavo_existing_sam2_video_adapter", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import existing detection helpers from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _box_iou(left, right) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union else 0.0


class GroundingDinoSam2LiveDetector:
    def __init__(self, *, dino_model: str, sam2_source_root: str, sam2_config: str,
                 sam2_checkpoint: str, device: str = "cuda:1",
                 prompt: str = "blue mailer bag,pink padded envelope,yellow tray",
                 context_classes: str = "robot arm,robot gripper",
                 package_labels: tuple[str, ...] = ("blue mailer bag", "pink padded envelope"),
                 static_classes: str = "yellow tray", dino_box_threshold: float = 0.20,
                 dino_text_threshold: float = 0.10, arm_box_threshold: float = 0.15,
                 dino_nms_iou: float = 0.50, min_target_area_ratio: float = 0.004,
                 max_target_area_ratio: float = 0.15, min_target_aspect: float = 0.35,
                 max_target_aspect: float = 2.8, arm_overlap_max: float = 0.35,
                 arm_partial_overlap: float = 0.02, detect_every_frames: int = 5,
                 max_target_instances: int = 6, max_missed_detection_updates: int = 3,
                 candidate_pipeline_mode: str = "legacy",
                 reliable_arm_min_score: float = 0.30):
        self.dino_model = Path(dino_model).expanduser()
        self.sam2_source_root = Path(sam2_source_root).expanduser()
        self.sam2_config = Path(sam2_config).expanduser()
        self.sam2_checkpoint = Path(sam2_checkpoint).expanduser()
        for path, label in ((self.dino_model, "local Grounding DINO model directory"),
                            (self.sam2_source_root / "sam2" / "build_sam.py", "SAM 2 source"),
                            (self.sam2_config, "SAM 2.1 config"),
                            (self.sam2_checkpoint, "SAM 2.1 checkpoint")):
            if not path.exists():
                raise FileNotFoundError(f"Required {label} is missing: {path}")
        if not self.dino_model.is_dir():
            raise FileNotFoundError(f"Grounding DINO must use an existing local model directory, got: {self.dino_model}")
        if detect_every_frames < 1:
            raise ValueError("detect_every_frames must be >= 1")

        try:
            import cv2
            import numpy as np
            import torch
        except ImportError as exc:
            raise RuntimeError(f"Grounding DINO + SAM2 plugin dependency is missing: {exc.name}; install the existing pipeline requirements") from exc
        self.cv2, self.np, self.torch = cv2, np, torch
        if str(self.sam2_source_root) not in sys.path:
            sys.path.insert(0, str(self.sam2_source_root))
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(f"Grounding DINO + SAM2 Python dependency is missing: {exc.name}; source SAM2 at {self.sam2_source_root} and install the local model dependencies") from exc

        # Reuse the successful C pipeline's detector implementation and thresholds.
        args = SimpleNamespace(
            prompt=prompt, context_classes=context_classes, static_classes=static_classes,
            dino_model=str(self.dino_model), device=device,
            dino_box_threshold=dino_box_threshold, dino_text_threshold=dino_text_threshold,
            arm_box_threshold=arm_box_threshold, dino_nms_iou=dino_nms_iou,
            min_target_area_ratio=min_target_area_ratio, max_target_area_ratio=max_target_area_ratio,
            min_target_aspect=min_target_aspect, max_target_aspect=max_target_aspect,
            basket_conf=0.02, conf=dino_box_threshold,
        )
        adapter_module = _load_existing_dino_factory()
        self._fraction_covered = adapter_module._fraction_covered
        self.package_labels = set(package_labels)
        self.candidate_pipeline_mode = str(candidate_pipeline_mode)
        self.reliable_arm_min_score = float(reliable_arm_min_score)
        self._candidate_runner = None
        if self.candidate_pipeline_mode == "legacy":
            self._dino = adapter_module._make_dino_detector(args)
        else:
            if self.candidate_pipeline_mode not in {"per_class", "joint_plus_per_class", "legacy_joint", "generic"}:
                raise ValueError(f"Unsupported candidate_pipeline_mode: {self.candidate_pipeline_mode}")
            self._dino = None
            self._candidate_runner = GroundingDinoPromptRunner(
                str(self.dino_model), device=device, package_labels=package_labels,
                joint_prompt=". ".join(label.strip() for label in prompt.split(",") if label.strip()) + ".",
                box_threshold=dino_box_threshold, text_threshold=dino_text_threshold,
                arm_box_threshold=arm_box_threshold,
                arm_text_threshold=min(dino_text_threshold, 0.05),
                min_area_ratio=min_target_area_ratio,
                max_area_ratio=max_target_area_ratio, nms_iou=dino_nms_iou)
        try:
            # SAM2's builder resolves config names through Hydra's `pkg://sam2`
            # search path; passing the absolute YAML path makes Hydra look for
            # an invalid config name such as `media/data/...`. Keep the full
            # path for existence checks, but pass the package-relative name.
            sam2_package_root = self.sam2_source_root / "sam2"
            try:
                config_name = self.sam2_config.relative_to(sam2_package_root).as_posix()
            except ValueError as exc:
                raise ValueError(f"SAM2 config must be inside {sam2_package_root}: {self.sam2_config}") from exc
            model = build_sam2(config_name, str(self.sam2_checkpoint), device=device, mode="eval")
        except Exception as exc:
            raise RuntimeError(f"Failed to load local SAM 2.1 model (no download attempted): config={self.sam2_config}, checkpoint={self.sam2_checkpoint}: {exc}") from exc
        self._sam2 = SAM2ImagePredictor(model)
        self.device = device
        self.arm_overlap_max = float(arm_overlap_max)
        self.arm_partial_overlap = float(arm_partial_overlap)
        self.detect_every_frames = int(detect_every_frames)
        self.max_target_instances = int(max_target_instances)
        self.max_missed_detection_updates = int(max_missed_detection_updates)
        self._frame_index = -1
        self._next_track_id = 1
        self._tracks: dict[int, dict[str, Any]] = {}
        self._arm_boxes: list[tuple[float, float, float, float]] = []
        self._last_detection_audit: dict[str, Any] = {}
        self._last_detection_time: float | None = None
        self._loaded = True
        log.info("Loaded existing Grounding DINO + SAM 2.1 once: dino=%s sam2=%s", self.dino_model, self.sam2_checkpoint)

    def _refresh_detections(self, rgb, now: float) -> tuple[set[int], list[tuple[float, float, float, float]]]:
        bgr = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
        if self._candidate_runner is not None:
            result = self._candidate_runner.detect(bgr, query_mode=self.candidate_pipeline_mode)
            accepted = result["detections"]
            self._last_detection_audit = result["diagnostics"]
            frame_h, frame_w = rgb.shape[:2]
            arm_boxes = []
            for arm in result["arm_candidates"]:
                geometry = arm_box_geometry(arm["box"], frame_w, frame_h)
                if (geometry["reliable_arm_geometry"]
                        and float(arm.get("confidence") or 0.0) >= self.reliable_arm_min_score):
                    arm_boxes.append(tuple(arm["box"]))
            # The pure package gate has already applied confidence, geometry,
            # NMS, and conservative arm-conflict handling. Do not reapply the
            # old overlap-only rejection here.
        else:
            items = self._dino(bgr)
            package_items, arm_boxes = [], []
            for item in items:
                if item.get("arm_context_only"):
                    arm_boxes.append(tuple(item["box"]))
                elif item.get("class") in self.package_labels:
                    package_items.append(item)

            # Preserve the legacy path byte-for-byte in its filtering policy.
            accepted = []
            for item in package_items:
                overlap = max((self._fraction_covered(item["box"], arm) for arm in arm_boxes), default=0.0)
                if overlap >= self.arm_overlap_max:
                    continue
                x1, y1, x2, y2 = item["box"]
                frame_h, frame_w = rgb.shape[:2]
                touches_frame = x1 <= 1 or y1 <= 1 or x2 >= frame_w - 1 or y2 >= frame_h - 1
                accepted.append({**item, "partial": bool(touches_frame or overlap >= self.arm_partial_overlap),
                                 "occluded": overlap >= self.arm_partial_overlap})
            accepted = sorted(accepted, key=lambda item: item["confidence"], reverse=True)[:self.max_target_instances]

        available = set(self._tracks)
        matched: set[int] = set()
        for item in accepted:
            choices = [( _box_iou(item["box"], track["box"]), track_id)
                       for track_id, track in self._tracks.items()
                       if track_id in available and track["label"] == item["class"]]
            score, track_id = max(choices, default=(0.0, -1))
            if score >= 0.20:
                available.remove(track_id)
                track = self._tracks[track_id]
                track.update({"box": tuple(item["box"]), "confidence": float(item["confidence"]),
                              "dino_box": tuple(item["box"]),
                              "label": item["class"], "partial": item["partial"],
                              "occluded": item["occluded"], "missed_updates": 0,
                              "last_detection_time": now,
                              "ambiguous": bool(item.get("ambiguous", False)),
                              "diagnostics": dict(item.get("diagnostics") or {}),
                              "mask_arm_boxes": list(arm_boxes)})
            elif len(self._tracks) < self.max_target_instances:
                track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[track_id] = {"box": tuple(item["box"]), "confidence": float(item["confidence"]),
                    "dino_box": tuple(item["box"]),
                    "label": item["class"], "partial": item["partial"], "occluded": item["occluded"],
                    "missed_updates": 0, "last_detection_time": now, "low_res_logits": None, "mask": None,
                    "mask_confidence": None, "ambiguous": bool(item.get("ambiguous", False)),
                    "diagnostics": dict(item.get("diagnostics") or {}),
                    "mask_arm_boxes": list(arm_boxes)}
            else:
                continue
            matched.add(track_id)
        for track_id in list(available):
            track = self._tracks[track_id]
            track["missed_updates"] += 1
            track["partial"] = True
            track["occluded"] = True
            if track["missed_updates"] > self.max_missed_detection_updates:
                del self._tracks[track_id]
        self._last_detection_time = now
        return matched, arm_boxes

    def __call__(self, image, *, now: float | None = None) -> list[Detection]:
        now = __import__("time").monotonic() if now is None else now
        rgb = self.np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return []
        rgb = self.np.ascontiguousarray(rgb[:, :, :3].astype(self.np.uint8, copy=False))
        self._frame_index += 1
        refresh = self._frame_index % self.detect_every_frames == 0
        refreshed_ids, arm_boxes = self._refresh_detections(rgb, now) if refresh else (set(), [])
        if refresh:
            self._arm_boxes = list(arm_boxes)
        self._sam2.set_image(rgb)
        output = []
        for track_id, track in list(self._tracks.items()):
            def retain_box_only_candidate():
                track["partial"] = True
                track["occluded"] = True
                track["mask"] = None
                box = track.get("dino_box") or track["box"]
                output.append(Detection(
                    bbox_xyxy=tuple(box), confidence=track["confidence"], label=track["label"],
                    source="grounding_dino_sam2", partial=True, mask=None,
                    mask_confidence=None, occluded=True,
                    detector_age_seconds=max(0.0, now - track["last_detection_time"]),
                    detector_bbox_xyxy=track.get("dino_box"),
                    ambiguous=bool(track.get("ambiguous", False)),
                    diagnostics=dict(track.get("diagnostics") or {})))

            prompt_box = self.np.asarray(track["box"], dtype=self.np.float32)
            mask_input = track["low_res_logits"]
            masks, scores, low_res_masks = self._sam2.predict(
                box=prompt_box, mask_input=mask_input, multimask_output=False, return_logits=False)
            if masks is None or len(masks) == 0 or not self.np.asarray(masks[0]).any():
                retain_box_only_candidate()
                continue
            mask = self.np.asarray(masks[0], dtype=bool)
            # Match the existing offline C pipeline: erase padded robot-arm
            # regions from each propagated package mask before ROI evaluation.
            frame_h, frame_w = mask.shape
            for arm_x1, arm_y1, arm_x2, arm_y2 in self._arm_boxes:
                pad_x, pad_y = (arm_x2 - arm_x1) * 0.04, (arm_y2 - arm_y1) * 0.04
                x1, y1 = max(0, int(arm_x1 - pad_x)), max(0, int(arm_y1 - pad_y))
                x2, y2 = min(frame_w, int(arm_x2 + pad_x)), min(frame_h, int(arm_y2 + pad_y))
                mask[y1:y2, x1:x2] = False
            if not mask.any():
                retain_box_only_candidate()
                continue
            track["mask"] = mask
            track["low_res_logits"] = self.np.asarray(low_res_masks[0], dtype=self.np.float32)[None, ...]
            track["mask_confidence"] = float(scores[0]) if len(scores) else None
            ys, xs = self.np.nonzero(mask)
            track["box"] = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
            touches_frame = xs.min() <= 1 or ys.min() <= 1 or xs.max() >= frame_w - 2 or ys.max() >= frame_h - 2
            partial = bool(track["partial"] or touches_frame)
            output.append(Detection(
                bbox_xyxy=track["box"], confidence=track["confidence"], label=track["label"],
                source="grounding_dino_sam2", partial=partial, mask=mask,
                mask_confidence=track["mask_confidence"], occluded=bool(track["occluded"] or touches_frame),
                detector_age_seconds=max(0.0, now - track["last_detection_time"]),
                detector_bbox_xyxy=track.get("dino_box"),
                ambiguous=bool(track.get("ambiguous", False)),
                diagnostics=dict(track.get("diagnostics") or {})))
        return output

    def diagnostic_data(self) -> dict[str, Any]:
        """Return current DINO prompts, raw candidates, filters, and conflicts."""
        return self._last_detection_audit


def create_detector(config: dict[str, Any]):
    """Factory used by `perception.detector_plugin`; called once per process."""
    if not isinstance(config, dict):
        raise TypeError("grounding_dino_sam2 detector_plugin_config must be a mapping")
    options = dict(config)
    package_labels = tuple(options.pop("package_labels", ("blue mailer bag", "pink padded envelope")))
    return GroundingDinoSam2LiveDetector(package_labels=package_labels, **options)
