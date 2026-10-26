"""Options B/C: detector initialization + SAM 2.1 multi-object propagation."""
from __future__ import annotations

import csv
import json
from contextlib import nullcontext

import cv2
import numpy as np


CLASSES = ("toy", "basket", "robot gripper")


def _labels(args):
    targets = [p.strip() for p in args.prompt.split(",") if p.strip()]
    contexts = [p.strip() for p in args.context_classes.split(",") if p.strip()]
    return targets, list(dict.fromkeys(targets + contexts))


def _box_from_mask(mask):
    ys, xs = np.where(mask)
    return None if not len(xs) else (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2-x1) * max(0., y2-y1)
    union = max(0., a[2]-a[0]) * max(0., a[3]-a[1]) + max(0., b[2]-b[0]) * max(0., b[3]-b[1]) - inter
    return inter / union if union else 0.


def _fraction_covered(box, region):
    """Fraction of a candidate box covered by a contextual region box."""
    x1, y1, x2, y2 = max(box[0], region[0]), max(box[1], region[1]), min(box[2], region[2]), min(box[3], region[3])
    intersection = max(0., x2-x1) * max(0., y2-y1)
    area = max(0., box[2]-box[0]) * max(0., box[3]-box[1])
    return intersection / area if area else 0.


def _color_seed_mask(frame, box, label):
    """Build a conservative chromatic foreground seed for named color bags."""
    lower = label.lower()
    if "blue" not in lower and "pink" not in lower:
        return None
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation = hsv[:, :, 0], hsv[:, :, 1]
    if "blue" in lower:
        selected = (hue >= 85) & (hue <= 135) & (saturation >= 35)
    else:
        selected = (((hue >= 140) & (hue <= 179)) | (hue <= 10)) & (saturation >= 35)
    kernel = np.ones((9, 9), dtype=np.uint8)
    selected = cv2.morphologyEx(selected.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    count, components, stats, _ = cv2.connectedComponentsWithStats(selected, 8)
    min_component = max(20, int(selected.size * 0.002))
    retained = np.zeros_like(selected, dtype=bool)
    for component_id in range(1, count):
        if stats[component_id, cv2.CC_STAT_AREA] >= min_component:
            retained |= components == component_id
    if retained.mean() < 0.02:
        return None
    retained = cv2.dilate(retained.astype(np.uint8), np.ones((7, 7), dtype=np.uint8)).astype(bool)
    full_mask = np.zeros(frame.shape[:2], dtype=bool)
    full_mask[y1:y2, x1:x2] = retained
    return full_mask


def _classwise_nms(items, threshold, labels=None):
    kept = []
    for label in dict.fromkeys(labels or CLASSES):
        candidates = sorted((item for item in items if item["class"] == label), key=lambda item: item["confidence"], reverse=True)
        while candidates:
            best = candidates.pop(0); kept.append(best)
            candidates = [item for item in candidates if _iou(best["box"], item["box"]) < threshold]
    return kept


def _deduplicate_overlapping_instances(items, threshold=0.45):
    """Remove same-object proposals that overlap or contain one another."""
    kept = []
    for item in sorted(items, key=lambda x: x["confidence"], reverse=True):
        def area(box):
            return max(0., box[2]-box[0]) * max(0., box[3]-box[1])
        def overlap_over_smaller(a, b):
            x1, y1 = max(a[0], b[0]), max(a[1], b[1])
            x2, y2 = min(a[2], b[2]), min(a[3], b[3])
            intersection = max(0., x2-x1) * max(0., y2-y1)
            return intersection / min(area(a), area(b)) if min(area(a), area(b)) else 0.
        overlap_index = next((i for i, previous in enumerate(kept)
                              if _iou(item["box"], previous["box"]) >= threshold
                              or overlap_over_smaller(item["box"], previous["box"]) >= 0.85), None)
        if overlap_index is None:
            kept.append(item)
            continue
        previous = kept[overlap_index]
        # When duplicate boxes have similar scores, prefer the tighter box so
        # one large proposal does not merge adjacent packages into a track.
        if (abs(item["confidence"] - previous["confidence"]) <= 0.20
                and area(item["box"]) < area(previous["box"])):
            kept[overlap_index] = item
    return kept


def _split_nested_package_proposals(items):
    """Split a high-confidence merged proposal around a tighter package box."""
    candidates = [dict(item) for item in items]
    removed = set()

    def area(box):
        return max(0., box[2]-box[0]) * max(0., box[3]-box[1])

    def contained_fraction(outer, inner):
        x1, y1 = max(outer[0], inner[0]), max(outer[1], inner[1])
        x2, y2 = min(outer[2], inner[2]), min(outer[3], inner[3])
        overlap = max(0., x2-x1) * max(0., y2-y1)
        return overlap / area(inner) if area(inner) else 0.

    for i, outer in enumerate(candidates):
        if i in removed:
            continue
        for j, inner in enumerate(candidates):
            if i == j or j in removed or outer["class"] != inner["class"]:
                continue
            outer_area, inner_area = area(outer["box"]), area(inner["box"])
            if outer_area < inner_area * 1.6 or contained_fraction(outer["box"], inner["box"]) < 0.85:
                continue
            if outer["confidence"] < inner["confidence"] + 0.20:
                # A lower-confidence enclosing box usually duplicates a more
                # precise instance proposal.
                removed.add(i)
                break

            ox1, oy1, ox2, oy2 = outer["box"]
            ix1, iy1, ix2, iy2 = inner["box"]
            pieces = [
                (ox1, oy1, ox2, min(oy2, iy1)),
                (ox1, max(oy1, iy2), ox2, oy2),
                (ox1, max(oy1, iy1), min(ox2, ix1), min(oy2, iy2)),
                (max(ox1, ix2), max(oy1, iy1), ox2, min(oy2, iy2)),
            ]
            pieces = [box for box in pieces if area(box) > 0 and 0.35 <= (box[2]-box[0]) / max(1., box[3]-box[1]) <= 2.8]
            if pieces:
                outer["box"] = max(pieces, key=area)

    return [item for index, item in enumerate(candidates) if index not in removed]


def _make_yolo_detector(args):
    from ultralytics import YOLOWorld
    targets, labels = _labels(args)
    static_labels = {p.strip() for p in args.static_classes.split(",") if p.strip()}
    model = YOLOWorld(args.yolo_model); model.set_classes(labels)
    def detect(frame):
        result = model.predict(frame, conf=min(args.conf, args.basket_conf), iou=args.iou, device=args.device, verbose=False)[0]
        items = []
        for det in (result.boxes if result.boxes is not None else []):
            cls_id, confidence = int(det.cls[0].item()), float(det.conf[0].item())
            if 0 <= cls_id < len(labels):
                label = labels[cls_id]
                threshold = min(args.conf, args.basket_conf) if label in static_labels or label == "basket" else args.conf
                if confidence >= threshold:
                    items.append({"class": label, "confidence": confidence, "box": tuple(float(v) for v in det.xyxy[0].tolist())})
        return items
    return detect


def _make_dino_detector(args):
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
    processor = AutoProcessor.from_pretrained(args.dino_model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.dino_model).to(device).eval()
    targets, labels = _labels(args)
    static_labels = {p.strip() for p in args.static_classes.split(",") if p.strip()}
    arm_labels = {label for label in labels if "arm" in label.lower() or "gripper" in label.lower()}
    detection_threshold = min(args.dino_box_threshold, args.basket_conf, args.arm_box_threshold)
    query = ". ".join(labels) + "."
    # A crowded target prompt often assigns the arm's appearance to a nearby
    # package phrase. Detect robot geometry with an isolated context query too.
    arm_query = "robot arm. robot gripper."
    def detect(frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image, text=query, return_tensors="pt").to(device)
        arm_inputs = processor(images=image, text=arm_query, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
            arm_outputs = model(**arm_inputs)
        result = processor.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=detection_threshold, text_threshold=args.dino_text_threshold, target_sizes=[image.size[::-1]])[0]
        items = []
        # Recent Transformers releases put token IDs in `labels`; actual
        # grounded phrases live in `text_labels`.
        grounded_labels = result.get("text_labels", result["labels"])
        for box, score, label in zip(result["boxes"], result["scores"], grounded_labels):
            label, confidence = str(label).lower().strip(" ."), float(score)
            # DINO may return a text span combining several prompts; it is not
            # a reliable semantic class and must never become a SAM prompt.
            if label not in labels:
                # Grounding DINO may return the noun phrase head (for example
                # "bag" for the prompt "mailing bag"). Accept it only when it
                # maps to exactly one requested/context label.
                matches = [candidate for candidate in labels if label and label in candidate.split()]
                if len(matches) != 1:
                    continue
                label = matches[0]
            if not label:
                continue
            if label in arm_labels:
                # Context detections from a crowded target query can be
                # confused with packages; arms are detected independently
                # below with a dedicated prompt.
                continue
            label_threshold = min(args.dino_box_threshold, args.basket_conf) if label in static_labels or label == "basket" else args.dino_box_threshold
            if confidence >= label_threshold:
                items.append({"class": label, "confidence": confidence, "box": tuple(float(v) for v in box.tolist())})
        arm_result = processor.post_process_grounded_object_detection(
            arm_outputs, arm_inputs.input_ids,
            threshold=args.arm_box_threshold, text_threshold=min(args.dino_text_threshold, 0.05),
            target_sizes=[image.size[::-1]],
        )[0]
        arm_items = []
        for box, score in zip(arm_result["boxes"], arm_result["scores"]):
            box = tuple(float(v) for v in box.tolist())
            area_ratio = max(0., box[2]-box[0]) * max(0., box[3]-box[1]) / float(frame.shape[0] * frame.shape[1])
            if args.min_target_area_ratio <= area_ratio <= args.max_target_area_ratio:
                arm_items.append({"class": "robot arm", "confidence": float(score), "box": box, "arm_context_only": True})
        # Apply plausible object-size limits before NMS. Otherwise a broad
        # package/tabletop phrase match can suppress the tighter real package
        # proposal and then itself be discarded later.
        frame_area = float(frame.shape[0] * frame.shape[1])
        def plausible_object_box(item):
            x1, y1, x2, y2 = item["box"]
            box_w, box_h = x2-x1, y2-y1
            area_ratio = box_w * box_h / frame_area
            aspect = box_w / max(1., box_h)
            return (args.min_target_area_ratio <= area_ratio <= args.max_target_area_ratio
                    and args.min_target_aspect <= aspect <= args.max_target_aspect)
        items = [item for item in items if plausible_object_box(item)]
        return _classwise_nms(items, args.dino_nms_iou, targets) + _classwise_nms(arm_items, args.dino_nms_iou, ["robot arm"])
    return detect


def _find_initial_prompts(args, baseline, basket_anchors, observation_type, detector):
    """Find highest-confidence valid left/right boxes independently over video."""
    best = {"left_toy": None, "right_toy": None}
    cap, frame_index = cv2.VideoCapture(str(args.video)), 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        raw = detector(frame)
        basket_anchors.apply([observation_type(-1, "basket", item["box"], item["confidence"], "DETECTED") for item in raw if item["class"] == "basket"])
        grippers = [item["box"] for item in raw if item["class"] == "robot gripper"]
        for item in raw:
            if item["class"] != "toy": continue
            if baseline.toy_rejection_reason(item["box"], frame.shape[1], frame.shape[0], grippers, args) is not None: continue
            side, _ = baseline.side_assignment(item["box"], frame.shape[1], frame.shape[0], args)
            if side is not None and (best[side] is None or item["confidence"] > best[side]["confidence"]):
                best[side] = {**item, "frame": frame_index}
        frame_index += 1
    cap.release()
    prompts = {side: item for side, item in best.items() if item is not None}
    if not prompts: raise RuntimeError("SAM 2.1 initialization failed: no baseline-valid toy box was found in this video.")
    if args.sam2_require_two_toys and len(prompts) != 2:
        raise RuntimeError(f"SAM 2.1 initialization found only {sorted(prompts)}; use --no-sam2-require-two-toys to track the visible toy only.")
    return prompts, basket_anchors.apply([])


def _find_generic_initial_prompts(args, detector):
    """Seed up to N individual moving objects and median-freeze static targets."""
    targets, _ = _labels(args)
    static_labels = {p.strip() for p in args.static_classes.split(",") if p.strip()}
    dynamic_labels = set(targets) - static_labels
    candidates_by_frame = []
    arm_boxes_by_frame = {}
    static_clusters = {label: [] for label in targets if label in static_labels}
    cap, frame_index = cv2.VideoCapture(str(args.video)), 0
    frame_diagonal = 1.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_diagonal = float(np.hypot(frame.shape[1], frame.shape[0]))
            if frame_index % max(1, args.detect_every) == 0:
                items = detector(frame)
                current_static = [item for item in items if item["class"] in static_clusters]
                arm_labels = {label for label in _labels(args)[1]
                              if "arm" in label.lower() or "gripper" in label.lower()}
                robot_regions = [item["box"] for item in items if item.get("arm_context_only")]
                if robot_regions:
                    arm_boxes_by_frame[frame_index] = robot_regions
                frame_area = float(frame.shape[0] * frame.shape[1])
                # Static anchors also need a plausible object-sized detection;
                # otherwise a broad tabletop phrase match can become a frozen tray.
                def plausible_object_box(item):
                    x1, y1, x2, y2 = item["box"]
                    box_w, box_h = x2-x1, y2-y1
                    area_ratio = box_w * box_h / frame_area
                    aspect = box_w / max(1., box_h)
                    return (args.min_target_area_ratio <= area_ratio <= args.max_target_area_ratio
                            and args.min_target_aspect <= aspect <= args.max_target_aspect)
                current_static = [item for item in current_static if plausible_object_box(item)]
                candidates = [item for item in items if item["class"] in dynamic_labels and
                              item["confidence"] >= args.min_target_confidence and
                              plausible_object_box(item) and
                              all(_iou(item["box"], fixed["box"]) < 0.80 for fixed in current_static) and
                              all(_fraction_covered(item["box"], region) < args.arm_overlap_max for region in robot_regions)]
                moving = _deduplicate_overlapping_instances(candidates)
                moving.sort(key=lambda item: item["confidence"], reverse=True)
                moving = moving[:max(1, args.max_target_instances)]
                if moving:
                    candidates_by_frame.append((frame_index, moving))

                # Associate repeated fixed-object detections and retain the
                # best-supported cluster for a median, frozen box.
                for item in current_static:
                    label = item["class"]
                    if label not in static_clusters:
                        continue
                    clusters = static_clusters[label]
                    match = max(clusters, key=lambda c: _iou(c["boxes"][-1], item["box"]), default=None)
                    if match is not None and _iou(match["boxes"][-1], item["box"]) >= args.static_anchor_iou:
                        match["boxes"].append(item["box"])
                        match["scores"].append(item["confidence"])
                    else:
                        clusters.append({"boxes": [item["box"]], "scores": [item["confidence"]]})
            frame_index += 1
    finally:
        cap.release()

    # Seed each distinct package at its first reliable observation. This gives
    # early frames coverage while allowing additional packages to enter later.
    # SAM2 accepts per-object prompts at different conditioning frames.
    best_dynamic = []
    for seed_frame, items in candidates_by_frame:
        for item in items:
            x1, y1, x2, y2 = item["box"]
            center = ((x1+x2)/2, (y1+y2)/2)
            duplicate = False
            for previous in best_dynamic:
                px1, py1, px2, py2 = previous["box"]
                pcenter = ((px1+px2)/2, (py1+py2)/2)
                same_class = item["class"] == previous["class"]
                if ((same_class and _iou(item["box"], previous["box"]) >= 0.35)
                        or (not same_class and _iou(item["box"], previous["box"]) >= 0.75)
                        or (same_class and np.hypot(center[0]-pcenter[0], center[1]-pcenter[1]) < 0.06*frame_diagonal)):
                    duplicate = True
                    break
            if not duplicate:
                best_dynamic.append({**item, "label": item["class"], "frame": seed_frame})
                if len(best_dynamic) >= max(1, args.max_target_instances):
                    break
        if len(best_dynamic) >= max(1, args.max_target_instances):
            break

    anchors = []
    for label, clusters in static_clusters.items():
        if not clusters:
            continue
        cluster = max(clusters, key=lambda c: (len(c["boxes"]), sum(c["scores"])))
        boxes = np.asarray(cluster["boxes"], dtype=np.float32)
        median_box = np.median(boxes, axis=0).tolist()
        anchors.append({
            "label": label,
            "box": median_box,
            "observations": len(cluster["boxes"]),
            "mean_confidence": float(np.mean(cluster["scores"])),
            "source": "STATIC_ANCHOR" if len(cluster["boxes"]) >= args.static_anchor_confirm_frames else "STATIC_ANCHOR_SINGLE",
        })
    if not best_dynamic and not anchors:
        raise RuntimeError("SAM 2.1 initialization failed: no requested target was detected in this video.")
    return best_dynamic, anchors, arm_boxes_by_frame


def _run_sam2(args, detector_factory, backend_name):
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise RuntimeError("This backend needs Meta SAM 2.1. Install facebookresearch/sam2 and provide --sam2-config/--sam2-checkpoint.") from exc
    from run_benchmark import Observation, StaticBasketAnchors, load_option_a_baseline
    if not args.video.is_file(): raise FileNotFoundError(args.video)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detector = detector_factory(args)
    if args.scene == "generic":
        prompt_items, static_anchors, arm_boxes_by_frame = _find_generic_initial_prompts(args, detector)
        baskets = []
    else:
        prompts, baskets = _find_initial_prompts(args, load_option_a_baseline(), StaticBasketAnchors(args), Observation, detector)
        prompt_items = [{"label": label, **item} for label, item in prompts.items()]
        static_anchors = []
        arm_boxes_by_frame = {}
    labels, masks = {}, {}
    if prompt_items:
        predictor = build_sam2_video_predictor(
            args.sam2_config,
            args.sam2_checkpoint,
            device=(f"cuda:{args.device}" if str(args.device).isdigit() else args.device),
            vos_optimized=args.sam2_vos_optimized,
        )
        if torch.cuda.is_available():
            autocast = torch.autocast("cuda", dtype=torch.bfloat16)
        elif str(args.device) == "cpu":
            # SAM 2 stores propagation memory in BF16; keep CPU linear layers in
            # the matching autocast context when this host has no CUDA device.
            autocast = torch.autocast("cpu", dtype=torch.bfloat16)
        else:
            autocast = nullcontext()
        with torch.inference_mode(), autocast:
            state = predictor.init_state(str(args.video))
            seed_cap = cv2.VideoCapture(str(args.video))
            for object_id, prompt in enumerate(prompt_items, 1):
                label = prompt["label"]
                seed_cap.set(cv2.CAP_PROP_POS_FRAMES, prompt["frame"])
                ok, seed_frame = seed_cap.read()
                seed_mask = _color_seed_mask(seed_frame, prompt["box"], label) if ok else None
                if seed_mask is not None:
                    predictor.add_new_mask(state, frame_idx=prompt["frame"], obj_id=object_id, mask=seed_mask)
                    prompt["seed_mode"] = "color_mask"
                else:
                    predictor.add_new_points_or_box(state, frame_idx=prompt["frame"], obj_id=object_id, box=np.asarray(prompt["box"], dtype=np.float32))
                    prompt["seed_mode"] = "box"
                labels[object_id] = label
            seed_cap.release()
            start = min(item["frame"] for item in prompt_items)
            for frame_id, object_ids, logits in predictor.propagate_in_video(state, start_frame_idx=start):
                masks[int(frame_id)] = {int(object_id): np.squeeze((logits[i] > 0).detach().cpu().numpy()).astype(bool) for i, object_id in enumerate(object_ids) if int(object_id) in labels}
    cap = cv2.VideoCapture(str(args.video)); fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(args.output_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    fields = ["frame", "time_s", "track_id", "label", "confidence", "source", "x1", "y1", "x2", "y2", "mask_area_px", "rejection_reason", "latency_ms"]
    rows = frame_id = 0
    with (args.output_dir / "tracks.csv").open("w", newline="", encoding="utf-8") as fh:
        out = csv.DictWriter(fh, fieldnames=fields); out.writeheader()
        while True:
            ok, frame = cap.read()
            if not ok: break
            canvas = frame.copy()
            arm_frame = min(arm_boxes_by_frame, key=lambda sample: abs(sample-frame_id), default=None)
            arm_regions = arm_boxes_by_frame.get(arm_frame, []) if arm_frame is not None and abs(arm_frame-frame_id) <= max(1, args.detect_every) else []
            for object_id, mask in masks.get(frame_id, {}).items():
                if arm_regions:
                    mask = mask.copy()
                    for region in arm_regions:
                        x1, y1, x2, y2 = region
                        pad_x, pad_y = (x2-x1)*0.04, (y2-y1)*0.04
                        x1, y1 = max(0, int(x1-pad_x)), max(0, int(y1-pad_y))
                        x2, y2 = min(width, int(x2+pad_x)), min(height, int(y2+pad_y))
                        mask[y1:y2, x1:x2] = False
                box = _box_from_mask(mask)
                if box is None: continue
                palette = ((255, 120, 40), (255, 40, 140), (60, 220, 80), (0, 190, 255), (210, 70, 220), (255, 220, 40))
                label, color = labels[object_id], palette[(object_id - 1) % len(palette)]
                overlay = canvas.copy(); overlay[mask] = color; canvas = cv2.addWeighted(canvas, .70, overlay, .30, 0)
                x1, y1, x2, y2 = map(int, box); cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                cv2.putText(canvas, f"{label} #{object_id} SAM2.1", (x1, max(16, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
                out.writerow(dict(frame=frame_id, time_s=frame_id/fps, track_id=object_id, label=label, confidence="", source="SAM2_PROPAGATED", x1=f"{box[0]:.2f}", y1=f"{box[1]:.2f}", x2=f"{box[2]:.2f}", y2=f"{box[3]:.2f}", mask_area_px=int(mask.sum()), rejection_reason="", latency_ms="")); rows += 1
            for index, anchor in enumerate(static_anchors, 1):
                x1, y1, x2, y2 = map(int, anchor["box"])
                color = (40, 210, 220)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                cv2.putText(canvas, f"{anchor['label']} STATIC_ANCHOR", (x1, max(16, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
                out.writerow(dict(frame=frame_id, time_s=frame_id/fps, track_id=-index, label=anchor["label"], confidence=f"{anchor['mean_confidence']:.5f}", source=anchor["source"], x1=f"{anchor['box'][0]:.2f}", y1=f"{anchor['box'][1]:.2f}", x2=f"{anchor['box'][2]:.2f}", y2=f"{anchor['box'][3]:.2f}", mask_area_px="", rejection_reason="", latency_ms="")); rows += 1
            for basket in baskets:
                x1, y1, x2, y2 = map(int, basket.box); cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 200, 200), 2)
                cv2.putText(canvas, f"{basket.label} {basket.source}", (x1, max(16, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, (40, 200, 200), 1, cv2.LINE_AA)
                out.writerow(dict(frame=frame_id, time_s=frame_id/fps, track_id=basket.track_id, label=basket.label, confidence="1.00000", source=basket.source, x1=f"{basket.box[0]:.2f}", y1=f"{basket.box[1]:.2f}", x2=f"{basket.box[2]:.2f}", y2=f"{basket.box[3]:.2f}", mask_area_px="", rejection_reason=basket.rejection_reason, latency_ms="")); rows += 1
            writer.write(canvas); frame_id += 1
    cap.release(); writer.release()
    summary = {"backend": backend_name, "video": str(args.video), "initial_prompts": prompt_items, "static_anchors": static_anchors, "basket_anchors": [{"label": item.label, "box": item.box, "source": item.source} for item in baskets], "processed_frames": frame_id, "track_rows": rows}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def run_yolo_sam2(args): return _run_sam2(args, _make_yolo_detector, "yolo_sam2_video")
def run_dino_sam2(args): return _run_sam2(args, _make_dino_detector, "grounding_dino_sam2_video")
