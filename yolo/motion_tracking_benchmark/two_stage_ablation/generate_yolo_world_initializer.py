#!/usr/bin/env python3
"""Export raw open-vocabulary initializer detections on GT-listed frames.

The GT file is used only to select rollout/frame pairs.  Its box coordinates
are never used during prediction or identity association.
"""
from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

SIDES = ("left_toy", "right_toy")
OUTPUT_FIELDS = ("rollout", "frame", "side", "x1", "y1", "x2", "y2", "score")


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    score: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO-World on GT-selected video frames and export initializer predictions."
    )
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("yolo_world", "grounding_dino", "sam3"), default="yolo_world"
    )
    parser.add_argument("--model", default="yolov8s-worldv2.pt")
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--conf", type=float, default=0.03)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=10)
    parser.add_argument(
        "--video-suffix",
        default="_head.mp4",
        help="Preferred suffix appended to a rollout name; recursive fallback search is also used.",
    )
    return parser.parse_args()


def is_visible(value: str | None) -> bool:
    return str(value or "1").strip().lower() in {"1", "true", "yes"}


def load_sampled_frames(path: Path) -> dict[str, set[int]]:
    sampled: dict[str, set[int]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"rollout", "frame", "side", "x1", "y1", "x2", "y2", "visible"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            if row["side"] not in SIDES or not is_visible(row.get("visible")):
                continue
            # A visible row without a complete box is not an annotated sample.
            if any(not str(row.get(name, "")).strip() for name in ("x1", "y1", "x2", "y2")):
                raise ValueError(f"{path}:{line_number}: visible box has empty coordinates")
            try:
                frame = int(row["frame"])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid frame {row['frame']!r}") from exc
            sampled.setdefault(row["rollout"].strip(), set()).add(frame)
    if not sampled:
        raise ValueError(f"{path} has no visible, fully annotated frames")
    return sampled


def find_video(video_dir: Path, rollout: str, suffix: str) -> Path:
    direct_names = (f"{rollout}{suffix}", f"{rollout}.mp4")
    for name in direct_names:
        candidate = video_dir / name
        if candidate.is_file():
            return candidate
    matches = []
    for name in direct_names:
        matches.extend(video_dir.rglob(name))
    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    if not unique:
        raise FileNotFoundError(
            f"No video found for rollout {rollout!r}; tried {', '.join(direct_names)} under {video_dir}"
        )
    raise RuntimeError(f"Multiple videos found for rollout {rollout!r}: {unique}")


def make_detector(args: argparse.Namespace):
    """Build one backend and return a ``frame -> detections`` callable."""
    if args.backend == "yolo_world":
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:
            raise RuntimeError("YOLO-World needs: pip install ultralytics") from exc
        model = YOLOWorld(args.model)
        model.set_classes(["toy"])

        def detect(frame):
            result = model.predict(frame, conf=args.conf, iou=args.iou, max_det=args.max_det,
                                   device=args.device, verbose=False)[0]
            boxes = result.boxes if result.boxes is not None else []
            return [Detection(tuple(float(v) for v in item.xyxy[0].tolist()), float(item.conf[0]))
                    for item in boxes]

        return detect

    if args.backend == "grounding_dino":
        try:
            import cv2
            import torch
            from PIL import Image
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Grounding DINO needs: pip install torch transformers pillow opencv-python") from exc
        device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
        processor = AutoProcessor.from_pretrained(args.dino_model)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(args.dino_model).to(device).eval()

        def detect(frame):
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(images=image, text="toy.", return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            result = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=args.conf, text_threshold=args.conf,
                target_sizes=[image.size[::-1]],
            )[0]
            return [Detection(tuple(float(v) for v in box.tolist()), float(score))
                    for box, score in zip(result["boxes"], result["scores"])]

        return detect

    try:
        import cv2
        from PIL import Image
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as exc:
        raise RuntimeError("SAM 3 needs the official facebookresearch/sam3 installation") from exc
    processor = Sam3Processor(build_sam3_image_model())

    def detect(frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt="toy")
        return [Detection(tuple(float(v) for v in box.detach().cpu().tolist()), float(score))
                for box, score in zip(output["boxes"], output["scores"])
                if float(score) >= args.conf]

    return detect


def squared_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def assign_identities(
    detections: list[Detection], previous: dict[str, Detection]
) -> dict[str, Detection]:
    if not detections:
        return {}
    if not previous:
        # Physical identities are defined only once both instances are visible;
        # guessing the side of a lone first detection would be irreversible.
        if len(detections) < 2:
            return {}
        ordered = sorted(detections, key=lambda item: item.center[0])
        return dict(zip(SIDES, ordered))

    known_sides = [side for side in SIDES if side in previous]
    best_cost = float("inf")
    best: dict[str, Detection] = {}
    for count in range(1, min(len(known_sides), len(detections)) + 1):
        for sides in itertools.permutations(known_sides, count):
            for chosen in itertools.permutations(detections, count):
                cost = sum(squared_distance(previous[side].center, det.center) for side, det in zip(sides, chosen))
                if count > len(best) or (count == len(best) and cost < best_cost):
                    best_cost = cost
                    best = dict(zip(sides, chosen))

    unused = [item for item in detections if item not in best.values()]
    missing_sides = [side for side in SIDES if side not in best]
    for side, item in zip(missing_sides, sorted(unused, key=lambda det: det.center[0])):
        best[side] = item
    return best


def process_rollout(detector, video: Path, rollout: str, wanted: set[int], args) -> list[dict[str, object]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install the video dependency with: pip install opencv-python") from exc
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    last_wanted = max(wanted)
    previous: dict[str, Detection] = {}
    rows: list[dict[str, object]] = []
    frame_index = 0
    try:
        while frame_index <= last_wanted:
            ok, frame = capture.read()
            if not ok:
                break
            raw = sorted(detector(frame), key=lambda item: item.score, reverse=True)[:2]
            assigned = assign_identities(raw, previous)
            if assigned:
                # Retain the last observation for a temporarily missed identity.
                previous.update(assigned)
            if frame_index in wanted:
                for side in SIDES:
                    detection = assigned.get(side)
                    if detection is None:
                        continue
                    x1, y1, x2, y2 = detection.box
                    rows.append(
                        {
                            "rollout": rollout,
                            "frame": frame_index,
                            "side": side,
                            "x1": f"{x1:.2f}",
                            "y1": f"{y1:.2f}",
                            "x2": f"{x2:.2f}",
                            "y2": f"{y2:.2f}",
                            "score": f"{detection.score:.6f}",
                        }
                    )
            frame_index += 1
    finally:
        capture.release()
    missing = sorted(frame for frame in wanted if frame >= frame_index)
    if missing:
        raise RuntimeError(f"{video} ended at frame {frame_index - 1}; requested frames not found: {missing}")
    return rows


def main() -> None:
    args = parse_args()
    if not args.gt.is_file():
        raise FileNotFoundError(args.gt)
    if not args.video_dir.is_dir():
        raise NotADirectoryError(args.video_dir)
    sampled = load_sampled_frames(args.gt)
    detector = make_detector(args)
    rows = []
    for rollout, frames in sorted(sampled.items()):
        video = find_video(args.video_dir, rollout, args.video_suffix)
        print(f"Predicting {rollout}: {video} ({len(frames)} sampled frames)")
        rows.extend(process_rollout(detector, video, rollout, frames, args))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
