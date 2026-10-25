#!/usr/bin/env python3
"""Offline Qwen2.5-VL scene reasoning on ROS-bag-aligned frames + ROI + tactile summaries.

This script never commands a robot. Requires prior extraction with extract_rgb_tactile.py.
Run in the Qwen/transformers environment, not the ROS Noetic container.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def load_roi(path):
    if not path:
        return None
    with open(path, encoding='utf8') as f:
        return json.load(f)


def overlay_roi(img, roi):
    if roi is None:
        return img
    h, w = img.shape[:2]
    if (w, h) != (roi['image_width'], roi['image_height']):
        raise ValueError(f'ROI expects {roi["image_width"]}x{roi["image_height"]} but image is {w}x{h}; do not scale blindly')
    out = img.copy()
    pts = np.array(roi['platform_polygon'], dtype=np.int32)
    cv2.polylines(out, [pts], True, (255, 0, 0), thickness=4)
    edge = np.array(roi['body_side_edge'], dtype=np.int32)
    cv2.line(out, tuple(edge[0]), tuple(edge[1]), (0, 0, 255), thickness=6)
    cv2.putText(out, 'BLUE: PLATFORM  RED: BODY-SIDE EDGE', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 3)
    cv2.putText(out, 'BLUE: PLATFORM  RED: BODY-SIDE EDGE', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 1)
    return out


def parse_json(text):
    text = re.sub(r'^\s*```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```\s*$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {'unparsed_output': text}


def tactile_summary(row):
    keys = ['alignment_ok', 'tactile_n', 'last_tactile_gap_ms',
            'left_normal_mean', 'left_normal_max', 'left_normal_last',
            'right_normal_mean', 'right_normal_max', 'right_normal_last',
            'left_tangential_mean', 'right_tangential_mean']
    return {k: row.get(k) for k in keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, help='Directory with images/ and aligned_features.csv')
    p.add_argument('--roi', required=True)
    p.add_argument('--model', default='/media/data/jiayi/hf_model/Qwen2.5-VL-7B-Instruct', help='Local Qwen2.5-VL-7B-Instruct directory or HF id')
    p.add_argument('--times', nargs='+', type=float, default=[0.5, 3.0, 7.0], help='Seconds relative to bag start')
    p.add_argument('--output', default='vlm_test/recovery_vlm_results.json')
    p.add_argument('--max-new-tokens', type=int, default=250)
    args = p.parse_args()

    folder = Path(args.dataset)
    rows = list(csv.DictReader((folder / 'aligned_features.csv').open(newline='', encoding='utf8')))
    if not rows:
        raise RuntimeError('No observations in CSV')
    roi = load_roi(args.roi)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map='auto').eval()
    processor = AutoProcessor.from_pretrained(args.model)
    results = []
    out_dir = Path(args.output).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for query_t in args.times:
        row = min(rows, key=lambda r: abs(float(r['relative_time_s']) - query_t))
        t = float(row['relative_time_s'])
        image_path = folder / row['image']
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        marked = overlay_roi(image, roi)
        annotated_path = out_dir / f'recovery_roi_{t:.2f}s.jpg'
        cv2.imwrite(str(annotated_path), marked)
        good_alignment = row.get('alignment_ok') == '1'
        sensors = tactile_summary(row) if good_alignment else {'alignment_ok': '0', 'notice': 'NO RELIABLE TIMED TACTILE MATCH'}
        prompt = (
            'You monitor a bimanual package sorting robot. Left side contains packages on the conveyor; '
            'center is the yellow sorting platform; right is the basket. '
            'The BLUE polygon marks the platform. The RED line marks its near edge, facing the robot. '
            'A package extending across the red line may need recovery, but packages held by a gripper '
            'must not be considered confirmed failures from position alone. '
            'Analyze ONLY the CURRENT IMAGE. Do not assume a past drop or future action. '
            'Tactile features below are raw aggregate force readings over the 300 ms before this image, '
            'not calibrated force in Newtons. They cannot establish which object is held or how many '
            'fingers contact it. Treat missing or stale measurements as unknown.\n'
            f'Image relative time: {t:.3f} s. Tactile summary: {json.dumps(sensors)}\n'
            'Output one JSON object with keys: '
            'package_visible_near_front_edge (true/false/"unknown"), '
            'package_extends_beyond_body_edge (true/false/"unknown"), '
            'gripper_holding_this_package (true/false/"unknown"), '
            'tactile_contact_evidence ("left"/"right"/"both"/"none"/"unknown"), '
            'recovery_candidate (true/false/"unknown"), '
            'recommended_skill ("normal_sort"/"recover_package"/"stop_and_verify"), '
            'visual_evidence (short string). '
            'When visual evidence or grasp status is uncertain, recommend stop_and_verify. '
            'Do not use a single image to claim recovery completed.'
        )
        rgb = Image.fromarray(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': rgb}, {'type': 'text', 'text': prompt}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors='pt').to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        tokens = generated[0][inputs['input_ids'].shape[1]:]
        raw = processor.decode(tokens, skip_special_tokens=True)
        result = {'time_requested_s': query_t, 'time_actual_s': t, 'image': str(image_path),
                  'roi_image': str(annotated_path), 'tactile': sensors,
                  'model_output': parse_json(raw), 'raw_output': raw}
        results.append(result)
        print(f'[{t:.2f}s] {raw}\n')
    with open(args.output, 'w', encoding='utf8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('Saved:', args.output)

if __name__ == '__main__':
    main()
