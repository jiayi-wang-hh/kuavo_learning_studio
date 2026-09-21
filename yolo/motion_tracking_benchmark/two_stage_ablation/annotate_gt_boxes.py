#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import cv2


SIDES = ["left_toy", "right_toy"]
CSV_FIELDS = [
    "rollout",
    "frame",
    "side",
    "x1",
    "y1",
    "x2",
    "y2",
    "visible",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manually annotate left/right toy bounding boxes."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/media/data/jiayi/dataset/toy_to_annotat"),
        help="Directory containing MP4 videos.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gt_boxes.csv"),
        help="Output GT CSV.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Annotate every N frames. Default: 10.",
    )

    return parser.parse_args()


def rollout_name(video_path: Path) -> str:
    """
    rollout21_head.mp4 -> rollout21
    rollout21.mp4      -> rollout21
    """
    name = video_path.stem

    for suffix in ["_head", "_left", "_right"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    return name


def load_existing(csv_path: Path):
    annotations = {}

    if not csv_path.exists():
        return annotations

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            key = (
                row["rollout"],
                int(row["frame"]),
                row["side"],
            )

            visible = int(row["visible"])

            if visible:
                box = (
                    int(float(row["x1"])),
                    int(float(row["y1"])),
                    int(float(row["x2"])),
                    int(float(row["y2"])),
                )
            else:
                box = None

            annotations[key] = {
                "visible": visible,
                "box": box,
            }

    return annotations


def save_annotations(csv_path: Path, annotations):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for (rollout, frame, side), ann in annotations.items():
        box = ann["box"]
        visible = ann["visible"]

        if visible and box is not None:
            x1, y1, x2, y2 = box
        else:
            x1 = y1 = x2 = y2 = ""

        rows.append(
            {
                "rollout": rollout,
                "frame": frame,
                "side": side,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "visible": visible,
            }
        )

    side_order = {
        "left_toy": 0,
        "right_toy": 1,
    }

    rows.sort(
        key=lambda r: (
            r["rollout"],
            int(r["frame"]),
            side_order.get(r["side"], 99),
        )
    )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def draw_existing(frame, rollout, frame_idx, annotations):
    canvas = frame.copy()

    for side in SIDES:
        key = (rollout, frame_idx, side)

        if key not in annotations:
            continue

        ann = annotations[key]

        if not ann["visible"] or ann["box"] is None:
            cv2.putText(
                canvas,
                f"{side}: NOT VISIBLE",
                (20, 70 if side == "left_toy" else 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )
            continue

        x1, y1, x2, y2 = ann["box"]

        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            canvas,
            side,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return canvas


def show_instruction(frame, rollout, frame_idx, total_frames, side=None):
    canvas = frame.copy()

    cv2.rectangle(
        canvas,
        (0, 0),
        (canvas.shape[1], 95),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        canvas,
        f"{rollout} | frame {frame_idx}/{total_frames - 1}",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    if side:
        cv2.putText(
            canvas,
            f"Annotating: {side}",
            (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

    cv2.putText(
        canvas,
        "ENTER/SPACE=draw | V=not visible | S=skip | Q=quit",
        (15, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
    )

    return canvas


def select_box(frame, rollout, frame_idx, total_frames, side):
    while True:
        display = show_instruction(
            frame,
            rollout,
            frame_idx,
            total_frames,
            side,
        )

        cv2.imshow("GT annotation", display)

        key = cv2.waitKey(0) & 0xFF

        # quit entire program
        if key in [ord("q"), 27]:
            return "quit", None

        # not visible
        if key == ord("v"):
            return "invisible", None

        # skip this object
        if key == ord("s"):
            return "skip", None

        # enter / space -> select ROI
        if key in [13, 32]:
            box = cv2.selectROI(
                f"Draw {side}",
                frame,
                fromCenter=False,
                showCrosshair=True,
            )

            cv2.destroyWindow(f"Draw {side}")

            x, y, w, h = box

            if w <= 1 or h <= 1:
                print("Invalid box. Try again.")
                continue

            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w)
            y2 = int(y + h)

            return "box", (x1, y1, x2, y2)


def annotate_frame(
    frame,
    rollout,
    frame_idx,
    total_frames,
    annotations,
    output_path,
):
    print()
    print("=" * 70)
    print(f"{rollout} | frame={frame_idx}")
    print("=" * 70)

    for side in SIDES:
        key = (rollout, frame_idx, side)

        # Existing annotation
        if key in annotations:
            ann = annotations[key]

            if ann["visible"]:
                print(f"Existing {side}: {ann['box']}")
            else:
                print(f"Existing {side}: NOT VISIBLE")

            preview = draw_existing(
                frame,
                rollout,
                frame_idx,
                annotations,
            )

            preview = show_instruction(
                preview,
                rollout,
                frame_idx,
                total_frames,
                side,
            )

            cv2.putText(
                preview,
                "Existing annotation: K=keep | R=redo",
                (15, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

            while True:
                cv2.imshow("GT annotation", preview)
                k = cv2.waitKey(0) & 0xFF

                if k == ord("k"):
                    break

                if k == ord("r"):
                    del annotations[key]
                    break

                if k in [ord("q"), 27]:
                    save_annotations(output_path, annotations)
                    return False

            if key in annotations:
                continue

        status, box = select_box(
            frame,
            rollout,
            frame_idx,
            total_frames,
            side,
        )

        if status == "quit":
            save_annotations(output_path, annotations)
            return False

        if status == "skip":
            continue

        if status == "invisible":
            annotations[key] = {
                "visible": 0,
                "box": None,
            }

            print(f"{side}: NOT VISIBLE")

        elif status == "box":
            annotations[key] = {
                "visible": 1,
                "box": box,
            }

            print(f"{side}: {box}")

        # Save immediately after every object
        save_annotations(output_path, annotations)

    return True


def main():
    args = parse_args()

    videos = sorted(args.input_dir.glob("*.mp4"))

    if not videos:
        raise FileNotFoundError(
            f"No MP4 files found in {args.input_dir}"
        )

    print(f"Found {len(videos)} videos:")
    for video in videos:
        print(f"  {video.name}")

    annotations = load_existing(args.output)

    if annotations:
        print(
            f"\nLoaded {len(annotations)} existing annotations "
            f"from {args.output}"
        )

    cv2.namedWindow(
        "GT annotation",
        cv2.WINDOW_NORMAL,
    )

    for video_index, video_path in enumerate(videos):
        rollout = rollout_name(video_path)

        print()
        print("#" * 70)
        print(
            f"[{video_index + 1}/{len(videos)}] "
            f"{video_path.name}"
        )
        print("#" * 70)

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"WARNING: cannot open {video_path}")
            continue

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"rollout: {rollout}")
        print(f"frames:  {total_frames}")
        print(f"fps:     {fps:.2f}")
        print(f"stride:  {args.stride}")

        sampled_frames = list(
            range(0, total_frames, args.stride)
        )

        for sample_index, frame_idx in enumerate(sampled_frames):
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_idx,
            )

            ok, frame = cap.read()

            if not ok:
                print(
                    f"WARNING: failed to read frame {frame_idx}"
                )
                continue

            print(
                f"[{sample_index + 1}/{len(sampled_frames)}] "
                f"frame={frame_idx}"
            )

            should_continue = annotate_frame(
                frame,
                rollout,
                frame_idx,
                total_frames,
                annotations,
                args.output,
            )

            if not should_continue:
                cap.release()
                cv2.destroyAllWindows()

                print()
                print("Annotation stopped.")
                print(f"Saved to: {args.output}")
                return

        cap.release()

    save_annotations(args.output, annotations)
    cv2.destroyAllWindows()

    print()
    print("=" * 70)
    print("Annotation complete")
    print(f"Saved to: {args.output}")
    print(f"Total object annotations: {len(annotations)}")
    print("=" * 70)


if __name__ == "__main__":
    main()