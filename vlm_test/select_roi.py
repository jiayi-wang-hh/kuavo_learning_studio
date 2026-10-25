#!/usr/bin/env python3
"""Interactively annotate a fixed platform polygon and its robot/body-side edge.

Usage:
    python select_roi.py --image recovery_initial.jpg --output roi_config.json

Click platform corners in clockwise order starting anywhere (4 clicks). Then
click close to the platform edge facing the robot. Press S to save.
Keys: U undo, R reset, S save, Q/Esc quit.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def edge_distance(p, a, b):
    p, a, b = map(lambda x: np.asarray(x, dtype=np.float64), (p, a, b))
    delta = b - a
    t = np.clip(np.dot(p - a, delta) / max(float(np.dot(delta, delta)), 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * delta)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, help='RGB image to annotate')
    parser.add_argument('--output', default='roi_config.json')
    parser.add_argument('--max-width', type=int, default=1400)
    parser.add_argument('--max-height', type=int, default=900)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        parser.error(f'Cannot open image: {args.image}')
    h, w = img.shape[:2]
    scale = min(1.0, args.max_width / w, args.max_height / h)
    display_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    base = cv2.resize(img, display_size, interpolation=cv2.INTER_AREA)
    points = []  # original image coordinates, clicked in perimeter order
    body_edge = None  # index of edge i -> (i+1) % 4
    window = 'Platform polygon ROI'

    def mouse(event, x, y, flags, userdata):
        nonlocal body_edge
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        p = [int(np.clip(round(x / scale), 0, w - 1)),
             int(np.clip(round(y / scale), 0, h - 1))]
        if len(points) < 4:
            points.append(p)
        else:
            body_edge = min(range(4), key=lambda i: edge_distance(
                p, points[i], points[(i + 1) % 4]))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse)
    cv2.resizeWindow(window, *display_size)

    print('Click 4 platform corners in order around its perimeter (clockwise or counter-clockwise).')
    print('Then click on the edge closest to the robot/body (it becomes the FRONT edge).')
    print('U: undo | R: reset | S: save | Q/Esc: quit')

    while True:
        canvas = base.copy()
        screen_pts = [(round(x * scale), round(y * scale)) for x, y in points]
        for i, (x, y) in enumerate(screen_pts):
            cv2.circle(canvas, (x, y), 6, (0, 255, 255), -1)
            cv2.putText(canvas, str(i), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 3)
            cv2.putText(canvas, str(i), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
        for i in range(len(screen_pts) - 1):
            cv2.line(canvas, screen_pts[i], screen_pts[i + 1], (255, 0, 0), 3)
        if len(points) == 4:
            cv2.line(canvas, screen_pts[3], screen_pts[0], (255, 0, 0), 3)
            if body_edge is not None:
                a, b = screen_pts[body_edge], screen_pts[(body_edge + 1) % 4]
                cv2.line(canvas, a, b, (0, 0, 255), 5)
                mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
                cv2.putText(canvas, 'BODY SIDE', (mid[0], max(22, mid[1] - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
        if len(points) < 4:
            msg = f'Click platform corner {len(points)+1}/4 | U undo | R reset | Q quit'
        elif body_edge is None:
            msg = 'Click the body-side/front platform EDGE | U undo | R reset'
        else:
            msg = 'Blue: platform | Red: body-side edge | S save | U undo | R reset'
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (25, 25, 25), -1)
        cv2.putText(canvas, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    .57, (255, 255, 255), 2)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            print('Quit without saving')
            break
        if key == ord('r'):
            points.clear()
            body_edge = None
        if key == ord('u'):
            if body_edge is not None:
                body_edge = None
            elif points:
                points.pop()
        if key == ord('s'):
            if len(points) != 4 or body_edge is None:
                print('Select all 4 corners and body-side edge before saving')
                continue
            polygon = np.asarray(points, dtype=np.float32)
            area = float(abs(cv2.contourArea(polygon)))
            if area < 100:
                print('Polygon too small; reset and reselect')
                continue
            if not cv2.isContourConvex(polygon):
                print('Corners do not form a convex polygon in perimeter order; reset and reselect')
                continue
            result = {
                'image_width': w,
                'image_height': h,
                'platform_polygon': points,
                'body_side_edge_indices': [body_edge, (body_edge + 1) % 4],
                'body_side_edge': [points[body_edge], points[(body_edge + 1) % 4]],
                'coordinate_system': 'image_pixels_top_left_origin',
            }
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2), encoding='utf-8')
            preview_path = output.with_name(output.stem + '_preview.jpg')
            cv2.imwrite(str(preview_path), canvas)
            print('Saved ROI:', output)
            print('Saved preview:', preview_path)
            print(json.dumps(result, indent=2))
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
