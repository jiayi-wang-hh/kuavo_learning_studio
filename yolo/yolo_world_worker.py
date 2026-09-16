#!/usr/bin/env python3
"""Line-delimited JSON YOLO-World worker for an external Conda environment.

Input:  {"image_path": "/shared/path/frame.jpg"}
Output: {"detections": [{"class": ..., "conf": ..., "box": [x1,y1,x2,y2]}]}
"""
from __future__ import annotations

import argparse
import json
import sys

import cv2
from ultralytics import YOLOWorld


CLASSES = ["toy", "basket", "robot gripper"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.03)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    model = YOLOWorld(args.model)
    model.set_classes(CLASSES)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            image = cv2.imread(request["image_path"])
            if image is None:
                raise ValueError("unable to read image")
            result = model.predict(image, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
            detections = []
            if result.boxes is not None:
                for det in result.boxes:
                    class_id = int(det.cls[0].item())
                    if 0 <= class_id < len(CLASSES):
                        detections.append({
                            "class": CLASSES[class_id],
                            "conf": float(det.conf[0].item()),
                            "box": [float(x) for x in det.xyxy[0].tolist()],
                        })
            print(json.dumps({"detections": detections}), flush=True)
        except Exception as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)


if __name__ == "__main__":
    main()
