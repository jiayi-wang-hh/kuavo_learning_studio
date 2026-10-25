#!/usr/bin/env python3
"""Export sampled head RGB and past-window tactile features from a ROS1 bag.

ROS Noetic, standard library + rosbag + OpenCV + numpy.
Uses ROS bag recording timestamps, NOT image header stamps (which may be zero).
"""
import argparse
import bisect
import csv
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rosbag

RGB_TOPIC = '/cam_h/color/image_raw/compressed'
TACTILE_TOPIC = '/dexhand/touch_state'


def force_sum(fingers, prefix):
    return float(sum(float(getattr(f, f'{prefix}{j}')) for f in fingers for j in (1, 2, 3)))


def tactile_values(msg):
    return {
        'left_normal': force_sum(msg.left_hand, 'normal_force'),
        'left_tangential': force_sum(msg.left_hand, 'tangential_force'),
        'right_normal': force_sum(msg.right_hand, 'normal_force'),
        'right_tangential': force_sum(msg.right_hand, 'tangential_force'),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bag', default='/tmp/normal.bag')
    p.add_argument('--output', default='/tmp/rgb_tactile_60s')
    p.add_argument('--duration', type=float, default=60.0)
    p.add_argument('--interval', type=float, default=0.5)
    p.add_argument('--window', type=float, default=0.3)
    p.add_argument('--max-gap-ms', type=float, default=100.0,
                   help='Maximum time since most recent tactile observation (bag time)')
    args = p.parse_args()
    if args.duration <= 0 or args.interval <= 0 or args.window <= 0:
        p.error('duration, interval and window must be positive')

    output = Path(args.output)
    images_dir = output / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    with rosbag.Bag(args.bag, 'r') as bag:
        info = bag.get_type_and_topic_info().topics
        for topic in (RGB_TOPIC, TACTILE_TOPIC):
            if topic not in info:
                raise RuntimeError(f'Missing bag topic: {topic}')
        start = bag.get_start_time()
        end = min(start + args.duration, bag.get_end_time())
        # Small, bounded-memory collection for requested interval only.
        tactile = []
        for topic, msg, stamp in bag.read_messages(topics=[TACTILE_TOPIC], start_time=None):
            t = stamp.to_sec()
            if t > end:
                break
            if t >= start - args.window:
                tactile.append((t, tactile_values(msg)))
        tactile.sort(key=lambda x: x[0])
        tactile_ts = [t for t, _ in tactile]
        print(f'Bag start: {start:.6f}  end: {end:.6f}; tactile messages: {len(tactile)}')

        fields = ['image', 'relative_time_s', 'rgb_bag_time_s', 'rgb_header_time_s',
                  'tactile_n', 'last_tactile_gap_ms', 'alignment_ok']
        for hand in ('left', 'right'):
            for kind in ('normal', 'tangential'):
                fields.extend([f'{hand}_{kind}_mean', f'{hand}_{kind}_max',
                               f'{hand}_{kind}_last'])

        csv_path = output / 'aligned_features.csv'
        saved = 0
        next_sample = start
        with csv_path.open('w', newline='', encoding='utf-8') as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            for _, msg, stamp in bag.read_messages(topics=[RGB_TOPIC]):
                t = stamp.to_sec()
                if t < start:
                    continue
                if t > end:
                    break
                if t + 1e-9 < next_sample:
                    continue
                next_sample = t + args.interval
                img_bytes = np.frombuffer(msg.data, dtype=np.uint8)
                image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if image is None:
                    print(f'WARNING: cannot decode RGB at {t:.6f}; skipping')
                    continue
                lo = bisect.bisect_left(tactile_ts, t - args.window)
                hi = bisect.bisect_right(tactile_ts, t)
                win = tactile[lo:hi]
                gap_ms = (t - win[-1][0]) * 1000.0 if win else math.nan
                ok = bool(win and gap_ms <= args.max_gap_ms)
                filename = f'frame_{saved:05d}.jpg'
                if not cv2.imwrite(str(images_dir / filename), image):
                    raise RuntimeError(f'Failed writing {filename}')
                row = {'image': f'images/{filename}',
                       'relative_time_s': round(t - start, 6),
                       'rgb_bag_time_s': round(t, 6),
                       'rgb_header_time_s': round(msg.header.stamp.to_sec(), 6),
                       'tactile_n': len(win),
                       'last_tactile_gap_ms': round(gap_ms, 3) if win else '',
                       'alignment_ok': int(ok)}
                for hand in ('left', 'right'):
                    for kind in ('normal', 'tangential'):
                        key = f'{hand}_{kind}'
                        vals = [entry[key] for _, entry in win]
                        row[f'{key}_mean'] = round(float(np.mean(vals)), 3) if vals else ''
                        row[f'{key}_max'] = round(max(vals), 3) if vals else ''
                        row[f'{key}_last'] = round(vals[-1], 3) if vals else ''
                writer.writerow(row)
                saved += 1
                if saved % 20 == 0:
                    print(f'Saved {saved} frames, latest t={t-start:.2f}s')
    print(f'Done: {saved} RGB frames; output: {output}')
    print(f'CSV: {csv_path}')
    print('NOTE: Bag timestamps provide approximate alignment, not verified sensor acquisition-time synchronization.')

if __name__ == "__main__":
    main()