"""Option D adapter using SAM 3's official stateful video-predictor API."""
from __future__ import annotations

import csv
import json
import time

import cv2
import numpy as np


def _numpy(value):
    """Accept SAM 3 tensors/arrays without making torch a direct dependency."""
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


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

    started = time.perf_counter()
    predictor = build_sam3_video_predictor()
    session = predictor.handle_request({"type": "start_session", "resource_path": str(args.video)})
    session_id = session["session_id"]
    # SAM 3 accepts an open-vocabulary phrase.  A comma-delimited CLI prompt is
    # converted to a short natural phrase to keep this option comparable to A-C.
    prompt = args.prompt.replace(",", " or ")
    predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": prompt})
    outputs = {}
    for response in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session_id}):
        outputs[int(response["frame_index"])] = response["outputs"]
    elapsed_ms = (time.perf_counter() - started) * 1000

    # The task has exactly two toys.  SAM 3 supplies stable object IDs; select
    # the two most confident initial instances, then retain only those IDs for
    # the full video rather than letting a later false instance become a track.
    if outputs:
        first = outputs[min(outputs)]
        first_ids, first_scores = _numpy(first["out_obj_ids"]), _numpy(first["out_probs"])
        allowed_ids = {int(first_ids[i]) for i in np.argsort(first_scores)[-2:]}
        for frame_index, out in outputs.items():
            keep_indices = [i for i, obj_id in enumerate(_numpy(out["out_obj_ids"])) if int(obj_id) in allowed_ids]
            for key in ("out_obj_ids", "out_probs", "out_boxes_xywh", "out_binary_masks"):
                out[key] = out[key][keep_indices]

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
    summary = {"backend": "sam3", "video": str(args.video), "prompt": args.prompt, "fps": fps, "processed_frames": len(outputs), "track_rows": sum(len(_numpy(out["out_obj_ids"])) for out in outputs.values()), "total_inference_ms": round(elapsed_ms, 2)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
