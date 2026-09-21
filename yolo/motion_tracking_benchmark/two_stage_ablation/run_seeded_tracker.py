#!/usr/bin/env python3
"""Run CSRT, SAM 2.1, or SAM 3 from one shared seed manifest."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path


FIELDS = ("rollout", "frame", "side", "x1", "y1", "x2", "y2", "score", "source")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("csrt", "sam2", "sam3"))
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--video-suffix", default="_head.mp4")
    parser.add_argument("--sam2-config")
    parser.add_argument("--sam2-checkpoint")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_seeds(path):
    result = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"rollout", "frame", "side", "x1", "y1", "x2", "y2"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            side = row["side"]
            if side not in {"left_toy", "right_toy"}:
                continue
            rollout, frame = row["rollout"], int(row["frame"])
            result[rollout][side] = (frame, tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2")))
    for rollout, sides in result.items():
        if set(sides) != {"left_toy", "right_toy"}:
            raise ValueError(f"{rollout}: seed manifest must contain both toy identities")
        if len({value[0] for value in sides.values()}) != 1:
            raise ValueError(f"{rollout}: both tracker seeds must use the same frame")
    if not result:
        raise ValueError(f"No complete seed pairs in {path}")
    return result


def find_video(root, rollout, suffix):
    names = (f"{rollout}{suffix}", f"{rollout}.mp4")
    found = [path for name in names for path in root.rglob(name)]
    found = sorted(set(found))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one video for {rollout!r} under {root}; found {found}")
    return found[0]


def mask_box(mask):
    import numpy as np
    ys, xs = np.where(mask)
    return None if not len(xs) else (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def csrt_creator(cv2):
    creator = getattr(cv2, "TrackerCSRT_create", None)
    if creator is None and hasattr(cv2, "legacy"):
        creator = getattr(cv2.legacy, "TrackerCSRT_create", None)
    if creator is None:
        raise RuntimeError("CSRT requires opencv-contrib-python")
    return creator


def run_csrt(video, rollout, seeds):
    import cv2
    seed_frame = next(iter(seeds.values()))[0]
    cap, frame_id, trackers, rows = cv2.VideoCapture(str(video)), 0, {}, []
    creator = csrt_creator(cv2)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id == seed_frame:
            for side, (_, (x1, y1, x2, y2)) in seeds.items():
                tracker = creator()
                tracker.init(frame, (round(x1), round(y1), round(x2 - x1), round(y2 - y1)))
                trackers[side] = tracker
                rows.append(dict(rollout=rollout, frame=frame_id, side=side, x1=x1, y1=y1,
                                 x2=x2, y2=y2, score=1.0, source="SEED"))
        elif frame_id > seed_frame:
            for side, tracker in trackers.items():
                ok_track, (x, y, width, height) = tracker.update(frame)
                if ok_track:
                    rows.append(dict(rollout=rollout, frame=frame_id, side=side, x1=x, y1=y,
                                     x2=x + width, y2=y + height, score=1.0, source="CSRT"))
        frame_id += 1
    cap.release()
    if not trackers:
        raise RuntimeError(f"{video} ended before seed frame {seed_frame}")
    return rows


def run_sam2(video, rollout, seeds, args):
    if not args.sam2_config or not args.sam2_checkpoint:
        raise ValueError("SAM 2.1 requires --sam2-config and --sam2-checkpoint")
    try:
        import numpy as np
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise RuntimeError("Install the official facebookresearch/sam2 package") from exc
    predictor = build_sam2_video_predictor(args.sam2_config, args.sam2_checkpoint)
    seed_frame = next(iter(seeds.values()))[0]
    labels = {1: "left_toy", 2: "right_toy"}
    rows = []
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()
    with torch.inference_mode(), autocast:
        state = predictor.init_state(str(video))
        for object_id, side in labels.items():
            predictor.add_new_points_or_box(
                state, frame_idx=seed_frame, obj_id=object_id,
                box=np.asarray(seeds[side][1], dtype=np.float32),
            )
        for frame_id, object_ids, logits in predictor.propagate_in_video(state, start_frame_idx=seed_frame):
            for index, object_id in enumerate(object_ids):
                object_id = int(object_id)
                if object_id not in labels:
                    continue
                mask = np.squeeze((logits[index] > 0).detach().cpu().numpy()).astype(bool)
                box = mask_box(mask)
                if box is not None:
                    rows.append(dict(rollout=rollout, frame=int(frame_id), side=labels[object_id],
                                     x1=box[0], y1=box[1], x2=box[2], y2=box[3], score=1.0,
                                     source="SAM2_PROPAGATED"))
    return rows


def center_xyxy(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def box_iou(a, b):
    """Compute IoU for two absolute-pixel XYXY boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def sam3_mask_box(mask):
    """Return an absolute-pixel XYXY box from one SAM3 binary mask.

    SAM3's video postprocessor returns ``out_binary_masks`` at the original
    video resolution with shape ``(N, H, W)``.  Keep this conversion local to
    the SAM3 backend so the independently working SAM2 path is unchanged.
    """
    import numpy as np

    mask = np.asarray(mask)
    # Some SAM3 revisions retain a singleton channel dimension per object.
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected one SAM3 mask with shape (H, W), got {mask.shape}")
    if mask.dtype != np.bool_:
        mask = mask > 0
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    )


def run_sam3(video, rollout, seeds):
    try:
        import numpy as np
        from sam3.model_builder import build_sam3_video_predictor
    except ImportError as exc:
        raise RuntimeError(
            "SAM3 backend requires the SAM3 package to be installed."
        ) from exc

    import cv2

    capture = cv2.VideoCapture(str(video))
    width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Cannot read video dimensions: {video}")

    predictor = build_sam3_video_predictor()

    rows = []

    for side in ("left_toy", "right_toy"):
        seed_frame, seed_box = seeds[side]

        x1, y1, x2, y2 = seed_box

        # Independent session for each toy.
        session = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(video),
            }
        )
        session_id = session["session_id"]

        # Since each toy uses its own session, obj_id=0 is sufficient.
        obj_id = 0

        # Route the seed through SAM3's *instance tracker* point API.  The
        # underlying tracker represents a box as two special point prompts:
        # label 2 is the top-left corner and label 3 is the bottom-right
        # corner.  With rel_coordinates=True, the tracker converts these
        # normalized original-video coordinates to its square model canvas.
        # This is exactly the transform performed by its internal ``box=``
        # path before that path appends labels 2 and 3.  Using the public
        # ``bounding_boxes`` request field would instead select SAM3's
        # semantic/visual-prompt path, not this explicit obj_id tracker path.
        tracker_box_points = [
            [x1 / width, y1 / height],
            [x2 / width, y2 / height],
        ]
        response = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": seed_frame,
                "points": tracker_box_points,
                "point_labels": [2, 3],
                "obj_id": obj_id,
                "rel_coordinates": True,
            }
        )

        initial = response["outputs"]

        # Diagnostic only: do not reject, replace, or fall back when the
        # initial mask disagrees with the initializer seed.
        initial_ids = np.asarray(initial.get("out_obj_ids", []), dtype=np.int64)
        initial_masks = np.asarray(initial.get("out_binary_masks", []))
        initial_box = None
        matching_indices = np.flatnonzero(initial_ids == obj_id)
        if len(matching_indices) > 0:
            initial_index = int(matching_indices[0])
            if initial_index < len(initial_masks):
                initial_box = sam3_mask_box(initial_masks[initial_index])
        init_iou = box_iou(seed_box, initial_box) if initial_box is not None else None

        print(
            f"[SAM3 INIT] {rollout}/{side}: "
            f"seed_frame={seed_frame}, "
            f"seed_box={seed_box}, "
            f"tracker_box_points_normalized={tracker_box_points}, "
            f"point_labels={[2, 3]}, "
            f"obj_id={obj_id}, "
            f"output_keys={list(initial.keys()) if isinstance(initial, dict) else type(initial)}, "
            f"initial_mask_box={initial_box}, "
            f"init_iou={init_iou}"
        )
        if init_iou is None:
            print(
                f"[SAM3 INIT WARNING] {rollout}/{side}: no non-empty mask for "
                f"obj_id={obj_id} on seed frame {seed_frame}"
            )
        elif init_iou < 0.10:
            print(
                f"[SAM3 INIT WARNING] {rollout}/{side}: initial mask/seed IoU "
                f"is {init_iou:.4f} (< 0.10); propagation will track an "
                f"incorrectly initialized mask"
            )

        nonempty_frames = 0
        matched_frames = 0
        debug_frames_printed = 0

        for item in predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "start_frame_index": seed_frame,
            }
        ):
            frame_id = int(item["frame_index"])

            if frame_id < seed_frame:
                continue

            output = item["outputs"]

            out_ids = np.asarray(
                output.get("out_obj_ids", []),
                dtype=np.int64,
            )

            out_boxes = np.asarray(
                output.get("out_boxes_xywh", []),
                dtype=np.float32,
            )

            out_masks = np.asarray(
                output.get("out_binary_masks", []),
            )

            out_scores = np.asarray(
                output.get("out_probs", []),
                dtype=np.float32,
            )

            if len(out_ids) > 0:
                nonempty_frames += 1

            for index, tracked_id in enumerate(out_ids):
                if int(tracked_id) != obj_id:
                    continue

                if index >= len(out_masks):
                    raise ValueError(
                        f"SAM3 frame {frame_id}: object index {index} has no corresponding "
                        f"mask; ids shape={out_ids.shape}, masks shape={out_masks.shape}"
                    )

                matched_frames += 1
                raw_box = out_boxes[index] if index < len(out_boxes) else None
                score = float(out_scores[index]) if index < len(out_scores) else 0.0
                mask = out_masks[index]

                # SAM3's authoritative spatial output is the full-resolution
                # binary mask.  Its out_boxes_xywh field is normalized XYWH
                # derived from a mask inside SAM3's postprocessor.  Derive the
                # benchmark box from the returned final mask instead, yielding
                # absolute-pixel XYXY and avoiding assumptions about a box
                # field that has varied across SAM3 revisions.
                box = sam3_mask_box(mask)

                if rollout == "rollout21" and side == "left_toy" and debug_frames_printed < 3:
                    mask_array = np.asarray(mask)
                    print(
                        f"[SAM3 DEBUG] frame={frame_id}, obj_id={int(tracked_id)}, "
                        f"output_keys={list(output.keys())}, "
                        f"raw_box={None if raw_box is None else np.asarray(raw_box).tolist()}, "
                        f"raw_box_shape={None if raw_box is None else np.asarray(raw_box).shape}, "
                        f"mask_shape={mask_array.shape}, "
                        f"mask_pixels={int(np.count_nonzero(mask_array))}, "
                        f"final_xyxy={box}"
                    )
                    debug_frames_printed += 1

                if box is None:
                    continue

                rows.append(
                    {
                        "rollout": rollout,
                        "frame": frame_id,
                        "side": side,
                        "x1": box[0],
                        "y1": box[1],
                        "x2": box[2],
                        "y2": box[3],
                        "score": score,
                        "source": "SAM3_PROPAGATED",
                    }
                )

        print(
            f"[SAM3 SUMMARY] {rollout}/{side}: "
            f"nonempty_frames={nonempty_frames}, "
            f"matched_frames={matched_frames}"
        )

        predictor.handle_request(
            {
                "type": "close_session",
                "session_id": session_id,
            }
        )

    return rows


def main():
    args = arguments()
    seeds_by_rollout = load_seeds(args.seeds)
    rows, videos = [], {}
    for rollout, seeds in sorted(seeds_by_rollout.items()):
        video = find_video(args.video_dir, rollout, args.video_suffix)
        videos[rollout] = str(video)
        print(f"Tracking {rollout} with {args.backend}: {video}")
        if args.backend == "csrt":
            rows.extend(run_csrt(video, rollout, seeds))
        elif args.backend == "sam2":
            rows.extend(run_sam2(video, rollout, seeds, args))
        else:
            rows.extend(run_sam3(video, rollout, seeds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    metadata = args.metadata or args.output.with_suffix(".metadata.json")
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps({"backend": args.backend, "seeds": str(args.seeds),
                                    "videos": videos, "rows": len(rows)}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.output}; metadata: {metadata}")


if __name__ == "__main__":
    main()
