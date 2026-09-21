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


def run_sam3(video, rollout, seeds):
    try:
        import numpy as np
        from sam3.model_builder import build_sam3_video_predictor
    except ImportError as exc:
        raise RuntimeError(
            "SAM3 backend requires the SAM3 package to be installed."
        ) from exc

    import cv2

    cap = cv2.VideoCapture(str(video))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    predictor = build_sam3_video_predictor()

    rows = []

    for side in ("left_toy", "right_toy"):
        seed_frame, seed_box = seeds[side]

        x1, y1, x2, y2 = seed_box

        # Use the center of the initializer bbox as a positive tracker point.
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

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

        response = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": seed_frame,
                "points": [[cx, cy]],
                "point_labels": [1],
                "obj_id": obj_id,

                # cx, cy are absolute image pixel coordinates.
                "rel_coordinates": False,
            }
        )

        initial = response["outputs"]

        print(
            f"[SAM3 INIT] {rollout}/{side}: "
            f"seed_frame={seed_frame}, "
            f"point=({cx:.1f}, {cy:.1f}), "
            f"obj_id={obj_id}, "
            f"output_keys={list(initial.keys()) if isinstance(initial, dict) else type(initial)}"
        )

        nonempty_frames = 0
        matched_frames = 0

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

            out_scores = np.asarray(
                output.get("out_probs", []),
                dtype=np.float32,
            )

            if len(out_ids) > 0:
                nonempty_frames += 1

            for tracked_id, box, score in zip(
                out_ids,
                out_boxes,
                out_scores,
            ):
                if int(tracked_id) != obj_id:
                    continue

                matched_frames += 1

                x, y, w, h = map(float, box)

                rows.append(
                    {
                        "rollout": rollout,
                        "frame": frame_id,
                        "side": side,
                        "x1": x * width,
                        "y1": y * height,
                        "x2": (x + w) * width,
                        "y2": (y + h) * height,
                        "score": float(score),
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
