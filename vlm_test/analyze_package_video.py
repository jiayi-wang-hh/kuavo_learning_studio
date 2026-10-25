#!/usr/bin/env python3
"""Analyze a short package-sorting video using local Qwen2.5-VL and FFmpeg.

Requires: torch, transformers with Qwen2.5-VL support, Pillow, ffmpeg in PATH.
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# PROMPT = """You are observing a bimanual package-sorting robot from its head camera.
# Scene layout: conveyor with packages on the left, yellow sorting platform in the center, basket on the right.
# Expected NORMAL procedure:
# 1. Left arm moves one package from conveyor to the yellow platform.
# 2. If necessary, right arm flips the package.
# 3. Right arm places the package into the basket.
# 4. Repeat.

# Analyze ONLY what is visually supported by the sampled frames, in chronological order. This is a normal-demonstration clip, but do not assume every listed action actually occurs in it.
# Return ONLY a JSON object with these fields:
# {
#   "observed_phases": ["left_pick", "transfer_to_platform", "right_flip", "right_place"],
#   "package_locations_seen": ["conveyor", "platform", "basket"],
#   "package_outside_target_zone": "true|false|unknown",
#   "scene_summary": "short description",
#   "visual_evidence": ["evidence 1", "evidence 2"]
# }
# Use only phases and locations actually seen; arrays may be empty. Use 'unknown' when the body-side region is not visible or occluded. The terms true|false|unknown indicate one of three strings, not a literal combined value.
# """
PROMPT = """
You are inspecting the placement of packages in a robotic sorting workspace.

The image contains three areas:
- LEFT: A conveyor holding unsorted packages. Ignore these packages when checking for placement failures.
- CENTER: A yellow rectangular sorting platform.
- FRONT/BOTTOM: The body-side edge of the yellow platform, closest to the robot.

Focus only on the yellow platform and the area immediately in front of its near edge.

A package requires recovery if it extends significantly beyond the front edge of the yellow platform into the body-side area and is not being held by a gripper.

Answer the following based only on visible evidence:

1. Is there a package near the FRONT/BOTTOM edge of the yellow platform?
2. Does that package extend beyond the platform's front edge?
3. Is it currently held by a gripper?
4. Does its placement satisfy the recovery condition defined above?

Return JSON with:
- package_at_front_edge
- extends_beyond_front_edge
- held_by_gripper
- recovery_required
- visual_evidence

Use true, false, or "unknown". Do not consider packages on the left conveyor as failures.
"""


def extract_frames(video: Path, fps: float, max_frames: int, workdir: Path):
    pattern = workdir / "frame_%04d.jpg"
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-hwaccel", "none", "-i", str(video),
        "-vf", f"fps={fps},scale=672:-2",
        "-frames:v", str(max_frames), "-q:v", "3", str(pattern),
    ]
    subprocess.run(cmd, check=True)
    paths = sorted(workdir.glob("frame_*.jpg"))
    if not paths:
        raise RuntimeError("No frames extracted. Check the video path and FFmpeg decoder.")
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="/media/data/jiayi/dataset/recovery_clip.mp4", help="A short MP4 clip, ideally 10–30 seconds")
    p.add_argument("--model", default="/media/data/jiayi/hf_model/Qwen2.5-VL-7B-Instruct", help="Local Qwen2.5-VL-7B-Instruct directory or HF id")
    p.add_argument("--output", default="recovery_analysis.json")
    p.add_argument("--fps", type=float, default=0.5, help="Sampled frames per second")
    p.add_argument("--max-frames", type=int, default=3)
    p.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    args = p.parse_args()

    video = Path(args.video).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not in PATH")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this 7B demo")

    with tempfile.TemporaryDirectory(prefix="package_vlm_") as td:
        frame_paths = extract_frames(video, args.fps, args.max_frames, Path(td))
        print(f"Sampled {len(frame_paths)} frames", flush=True)
        images = []
        try:
            for path in frame_paths:
                im = Image.open(path)
                images.append(im.convert("RGB"))
                im.close()
            content = [{"type": "text", "text": PROMPT}]
            for i, im in enumerate(images):
                content.append({"type": "text", "text": f"Sample {i + 1}, ~{i / args.fps:.1f} seconds:"})
                content.append({"type": "image", "image": im})
            messages = [{"role": "user", "content": content}]
            processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model, torch_dtype=torch.bfloat16, device_map="auto"
            ).eval()
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device)
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=450, do_sample=False)
            completion = ids[:, inputs["input_ids"].shape[1]:]
            answer = processor.batch_decode(completion, skip_special_tokens=True)[0].strip()
            print("\n=== VLM analysis ===\n" + answer)
            out = {"video": str(video), "fps": args.fps,
                   "frames_sampled": len(frame_paths), "model_output": answer}
            Path(args.output).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).expanduser().write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nSaved to {args.output}")
        finally:
            for im in images:
                im.close()


if __name__ == "__main__":
    main()
