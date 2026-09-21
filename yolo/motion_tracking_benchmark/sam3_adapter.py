"""Option D adapter using SAM 3's official stateful video-predictor API."""
from __future__ import annotations

import csv
import json
import time

import cv2
import numpy as np


_SAM3_PREDICTOR = None


def _numpy(value):
    """Accept SAM 3 tensors/arrays without making torch a direct dependency."""
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _get_predictor(build_sam3_video_predictor):
    """Build SAM3 once and reuse it across rollout sessions in one batch."""
    global _SAM3_PREDICTOR
    if _SAM3_PREDICTOR is None:
        _SAM3_PREDICTOR = build_sam3_video_predictor()
    return _SAM3_PREDICTOR


def _select_track_ids(outputs, limit=2):
    """Choose stable SAM3 IDs without assuming frame zero has detections.

    Prefer the earliest frame where the complete expected object set is
    present.  If no frame contains ``limit`` objects, retain the best IDs seen
    anywhere instead of discarding every later detection because frame zero
    happened to be empty.
    """
    best_by_id = {}
    for frame_index, out in sorted(outputs.items()):
        ids = _numpy(out.get("out_obj_ids", []))
        scores = _numpy(out.get("out_probs", []))
        candidates = [
            (int(obj_id), float(scores[index]) if index < len(scores) else 0.0)
            for index, obj_id in enumerate(ids)
        ]
        for obj_id, score in candidates:
            previous = best_by_id.get(obj_id)
            if previous is None or score > previous[0]:
                best_by_id[obj_id] = (score, frame_index)
        if len(candidates) >= limit:
            selected = sorted(candidates, key=lambda item: item[1], reverse=True)[:limit]
            return {obj_id for obj_id, _ in selected}, frame_index, "earliest_complete_frame"

    ranked = sorted(
        best_by_id.items(),
        key=lambda item: (-item[1][0], item[1][1], item[0]),
    )
    return {obj_id for obj_id, _ in ranked[:limit]}, None, "global_fallback"


def run_sam3(args):
    try:
        from sam3.model_builder import build_sam3_video_predictor
        from sam3.visualization_utils import render_masklet_frame
    except ImportError as exc:
        raise RuntimeError(
            "Option D requires Meta SAM 3. Install facebookresearch/sam3 and its "
            "checkpoint prerequisites; see this directory's README."
        ) from exc

    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    predictor = _get_predictor(build_sam3_video_predictor)
    started = time.perf_counter()
    session = predictor.handle_request({"type": "start_session", "resource_path": str(args.video)})
    session_id = session["session_id"]
    # SAM 3 accepts an open-vocabulary phrase.  A comma-delimited CLI prompt is
    # converted to a short natural phrase to keep this option comparable to A-C.
    prompt = args.prompt.replace(",", " or ")
    outputs = {}
    try:
        predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": prompt})
        for response in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session_id}):
            outputs[int(response["frame_index"])] = response["outputs"]
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    elapsed_ms = (time.perf_counter() - started) * 1000

    # The task has exactly two toys.  SAM 3 supplies stable object IDs; select
    # them from the earliest complete frame rather than requiring frame zero to
    # contain both instances.  IDs remain fixed after selection.
    raw_track_rows = sum(len(_numpy(out.get("out_obj_ids", []))) for out in outputs.values())
    allowed_ids, selection_frame, selection_method = _select_track_ids(outputs)
    if outputs:
        for frame_index, out in outputs.items():
            keep_indices = [i for i, obj_id in enumerate(_numpy(out["out_obj_ids"])) if int(obj_id) in allowed_ids]
            for key in ("out_obj_ids", "out_probs", "out_boxes_xywh", "out_binary_masks"):
                out[key] = _numpy(out[key])[keep_indices]

    fields = ["frame", "time_s", "track_id", "label", "confidence", "source", "x1", "y1", "x2", "y2", "mask_area_px", "rejection_reason", "latency_ms"]
    with (args.output_dir / "tracks.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader()
        for frame_index, out in sorted(outputs.items()):
            ids, probs = _numpy(out["out_obj_ids"]), _numpy(out["out_probs"])
            boxes, masks = _numpy(out["out_boxes_xywh"]), _numpy(out["out_binary_masks"])
            for obj_id, probability, (x, y, w, h), mask in zip(ids, probs, boxes, masks):
                # SAM 3 boxes are normalized XYWH; persist pixels, matching A-C.
                writer.writerow(dict(frame=frame_index, time_s=frame_index / fps, track_id=int(obj_id), label=args.prompt, confidence=f"{float(probability):.5f}", source="SAM3_TRACKED", x1=f"{float(x)*width:.2f}", y1=f"{float(y)*height:.2f}", x2=f"{float(x+w)*width:.2f}", y2=f"{float(y+h)*height:.2f}", mask_area_px=int(np.asarray(mask).sum()), rejection_reason="", latency_ms=f"{elapsed_ms / max(1, len(outputs)):.2f}"))

    cap = cv2.VideoCapture(str(args.video))
    writer = cv2.VideoWriter(str(args.output_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if index in outputs:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.cvtColor(render_masklet_frame(rgb, outputs[index], frame_idx=index), cv2.COLOR_RGB2BGR)
        writer.write(frame)
        index += 1
    cap.release(); writer.release()
    summary = {"backend": "sam3", "video": str(args.video), "prompt": args.prompt, "fps": fps, "processed_frames": len(outputs), "raw_track_rows": raw_track_rows, "track_rows": sum(len(_numpy(out["out_obj_ids"])) for out in outputs.values()), "selected_track_ids": sorted(allowed_ids), "selection_frame": selection_frame, "selection_method": selection_method, "total_inference_ms": round(elapsed_ms, 2)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
