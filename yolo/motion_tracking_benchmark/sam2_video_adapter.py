"""Options B/C: detector initialization + SAM 2.1 multi-object propagation."""
from __future__ import annotations

import csv
import json
from contextlib import nullcontext

import cv2
import numpy as np


CLASSES = ("toy", "basket", "robot gripper")


def _box_from_mask(mask):
    ys, xs = np.where(mask)
    return None if not len(xs) else (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2-x1) * max(0., y2-y1)
    union = max(0., a[2]-a[0]) * max(0., a[3]-a[1]) + max(0., b[2]-b[0]) * max(0., b[3]-b[1]) - inter
    return inter / union if union else 0.


def _classwise_nms(items, threshold):
    kept = []
    for label in CLASSES:
        candidates = sorted((item for item in items if item["class"] == label), key=lambda item: item["confidence"], reverse=True)
        while candidates:
            best = candidates.pop(0); kept.append(best)
            candidates = [item for item in candidates if _iou(best["box"], item["box"]) < threshold]
    return kept


def _make_yolo_detector(args):
    from ultralytics import YOLOWorld
    model = YOLOWorld(args.yolo_model); model.set_classes(list(CLASSES))
    def detect(frame):
        result = model.predict(frame, conf=min(args.conf, args.basket_conf), iou=args.iou, device=args.device, verbose=False)[0]
        items = []
        for det in (result.boxes if result.boxes is not None else []):
            cls_id, confidence = int(det.cls[0].item()), float(det.conf[0].item())
            if 0 <= cls_id < len(CLASSES):
                label = CLASSES[cls_id]
                if confidence >= (args.basket_conf if label == "basket" else args.conf):
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
    query = ". ".join(CLASSES) + "."
    def detect(frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image, text=query, return_tensors="pt").to(device)
        threshold = min(args.dino_box_threshold, args.basket_conf)
        with torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=threshold, text_threshold=args.dino_text_threshold, target_sizes=[image.size[::-1]])[0]
        items = []
        for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            label, confidence = str(label).lower().strip(" ."), float(score)
            # DINO may return a text span combining several prompts; it is not
            # a reliable semantic class and must never become a SAM prompt.
            if label not in CLASSES:
                continue
            if confidence >= (args.basket_conf if label == "basket" else args.dino_box_threshold):
                items.append({"class": label, "confidence": confidence, "box": tuple(float(v) for v in box.tolist())})
        return _classwise_nms(items, args.dino_nms_iou)
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


def _run_sam2(args, detector_factory, backend_name):
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise RuntimeError("This backend needs Meta SAM 2.1. Install facebookresearch/sam2 and provide --sam2-config/--sam2-checkpoint.") from exc
    from run_benchmark import Observation, StaticBasketAnchors, load_option_a_baseline
    if not args.video.is_file(): raise FileNotFoundError(args.video)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts, baskets = _find_initial_prompts(args, load_option_a_baseline(), StaticBasketAnchors(args), Observation, detector_factory(args))
    predictor = build_sam2_video_predictor(args.sam2_config, args.sam2_checkpoint, vos_optimized=args.sam2_vos_optimized)
    labels, masks = {}, {}
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()
    with torch.inference_mode(), autocast:
        state = predictor.init_state(str(args.video))
        for object_id, label in enumerate(("left_toy", "right_toy"), 1):
            if label in prompts:
                prompt = prompts[label]
                predictor.add_new_points_or_box(state, frame_idx=prompt["frame"], obj_id=object_id, box=np.asarray(prompt["box"], dtype=np.float32))
                labels[object_id] = label
        start = min(item["frame"] for item in prompts.values())
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
            for object_id, mask in masks.get(frame_id, {}).items():
                box = _box_from_mask(mask)
                if box is None: continue
                label, color = labels[object_id], ((255, 120, 40) if object_id == 1 else (255, 40, 140))
                overlay = canvas.copy(); overlay[mask] = color; canvas = cv2.addWeighted(canvas, .70, overlay, .30, 0)
                x1, y1, x2, y2 = map(int, box); cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                cv2.putText(canvas, f"{label} SAM2.1", (x1, max(16, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
                out.writerow(dict(frame=frame_id, time_s=frame_id/fps, track_id=object_id, label=label, confidence="", source="SAM2_PROPAGATED", x1=f"{box[0]:.2f}", y1=f"{box[1]:.2f}", x2=f"{box[2]:.2f}", y2=f"{box[3]:.2f}", mask_area_px=int(mask.sum()), rejection_reason="", latency_ms="")); rows += 1
            for basket in baskets:
                x1, y1, x2, y2 = map(int, basket.box); cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 200, 200), 2)
                cv2.putText(canvas, f"{basket.label} {basket.source}", (x1, max(16, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, (40, 200, 200), 1, cv2.LINE_AA)
                out.writerow(dict(frame=frame_id, time_s=frame_id/fps, track_id=basket.track_id, label=basket.label, confidence="1.00000", source=basket.source, x1=f"{basket.box[0]:.2f}", y1=f"{basket.box[1]:.2f}", x2=f"{basket.box[2]:.2f}", y2=f"{basket.box[3]:.2f}", mask_area_px="", rejection_reason=basket.rejection_reason, latency_ms="")); rows += 1
            writer.write(canvas); frame_id += 1
    cap.release(); writer.release()
    summary = {"backend": backend_name, "video": str(args.video), "initial_prompts": prompts, "basket_anchors": [{"label": item.label, "box": item.box, "source": item.source} for item in baskets], "processed_frames": frame_id, "track_rows": rows}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def run_yolo_sam2(args): return _run_sam2(args, _make_yolo_detector, "yolo_sam2_video")
def run_dino_sam2(args): return _run_sam2(args, _make_dino_detector, "grounding_dino_sam2_video")
