# Visual motion-tracking backend benchmark

This directory is an **offline, vision-only** comparison harness for the four
perception candidates in `report/10_MOTION_TRACKING`.  It deliberately does
not read robot state, infer task phases, or emit a robot PAUSE command.

Every run writes the same artifacts:

- `tracks.csv`: one row per target per frame (box, optional mask area, source,
  confidence, and latency);
- `annotated.mp4`: boxes/masks/track IDs overlaid on the input;
- `summary.json`: configuration, frame counts, source distribution and timing.

## Quick start

Run from the repository root.  A is the lightweight baseline and is the only
backend required for the first smoke test:

```bash
python yolo/motion_tracking_benchmark/run_benchmark.py \
  --backend yolo_csrt \
  --prompt 'toy' \
  --output-dir outputs/motion_tracking/a_yolo_csrt
```

The runner discovers every `*_head.mp4` rollout in `--input-dir`, and stores
each result under its rollout name. Use `--scene generic` only when the video
does not match the fixed toy/table setup.

Option B uses the fixed toy-table prompt and SAM 2.1 video propagation:

```bash
python yolo/motion_tracking_benchmark/run_benchmark.py \
  --backend yolo_sam \
  --prompt 'toy' \
  --output-dir outputs/motion_tracking/b_yolo_sam2 \
  --sam2-config /media/data/jiayi/sam2/configs/sam2.1/sam2.1_hiera_s.yaml \
  --sam2-checkpoint /media/data/jiayi/sam2/checkpoints/sam2.1_hiera_small.pt
```

## Backend mapping

| CLI backend | Report option | Method |
| --- | --- | --- |
| `yolo_csrt` | A | YOLO-World detection + OpenCV CSRT between detections |
| `yolo_sam` | B | YOLO-World box initialization + SAM 2.1 video propagation |
| `grounding_dino_sam` | C | Grounding DINO box initialization + SAM 2.1 video propagation |
| `sam3` | D | SAM 3 text-prompted video segmentation/tracking |

All runs use the same output schema so the results can later feed a
phase-aware trigger without changing its input format.

`yolo_csrt` directly reuses the temporal filtering/state-machine helpers from
`yolo/test_yolo_world_roi_filter.py`. Its detector vocabulary and defaults are
therefore exactly the established baseline: `toy`, `basket`, `robot gripper`,
with `DETECTED → TRACKED → PREDICTED → LOST` fallback behavior. The extra
`table` / `robot arm` context vocabulary belongs to B/C only.

Option B prompts YOLO-World with `toy, basket, robot gripper`, applies the
same verified Option-A geometry/side filters, then uses the first valid
`left_toy/right_toy` boxes as SAM 2.1 multi-object video prompts. SAM 2.1
propagation owns temporal identity and occlusion handling; it does not use
CSRT. Provide `--sam2-config` and `--sam2-checkpoint` from a SAM 2.1 checkout.
The two toy prompts may be initialized on different video frames; use
`--sam2-require-two-toys` only when a rollout must contain both toys.
During that same offline YOLO scan, baskets are temporally confirmed and
median-fused into persistent `basket_1/basket_2` static anchors.
Option C uses Grounding DINO with the same SAM 2.1 propagation path, but uses
its own `--dino-box-threshold`, `--dino-text-threshold`, strict class whitelist
and class-wise NMS before accepting any prompt box. Option D prompts SAM 3
only with `toy` and retains the two most confident initial SAM 3 object IDs
for the entire video.

For A/C, the detector prompt remains `robot gripper`. When two gripper
instances are first visible, the benchmark creates `left_gripper` and
`right_gripper` identities from their initial image-side order, then preserves
those IDs via `--gripper-association-iou` box association. They are instance
identities, not detector classes. Without robot state or calibrated camera
extrinsics, these names mean *initial image-side* rather than a guaranteed
robot-body left/right convention.

`table` is intentionally not detected by default. The fixed head-camera
workspace ROI is more stable for this setup. Baskets use `--basket-conf` for
low-threshold proposals, require `--basket-init-frames` temporal matches, then
become median-fused `STATIC_ANCHOR` boxes. After an anchor freezes it remains
available even if basket detection drops out.

## Known toy-table scene classes

The default `--scene toy_table` constrains the detector vocabulary to:

`toy`, `basket`, `table`, `robot arm`, `robot gripper`.

Only `toy` becomes a tracked motion target. On frames where two toys are
detected, the leftmost is recorded as `left_toy` and the other as `right_toy`;
this is more reliable than asking an open-vocabulary model to detect a
"left toy" directly. Basket/table/arm/gripper remain visible in the annotated
video and CSV as context detections, but never become motion tracks.

Option A 的 toy 会经基线 ROI、尺寸和 gripper 覆盖过滤。正常的 gripper 遮挡则由
`TRACKED → PREDICTED → LOST` 时序状态机处理，而不会直接删除已有 track。完整规则见
[OCCLUSION_AND_GEOMETRY.md](OCCLUSION_AND_GEOMETRY.md)。

Override the context vocabulary if needed:

```bash
python yolo/motion_tracking_benchmark/run_benchmark.py \
  --backend yolo_csrt --input-dir /media/data/jiayi/dataset/joy_success \
  --prompt 'toy' \
  --output-dir outputs/motion_tracking/toy_table
```

## Dependencies

Install only the backend(s) being compared.  The base dependencies are in
`requirements-base.txt`.

- A/B: `pip install ultralytics opencv-contrib-python`
- C: additionally `pip install transformers torch torchvision`
- D: install Meta's [`facebookresearch/sam3`](https://github.com/facebookresearch/sam3)
  following its checkpoint/access instructions.  The adapter uses the official
  `build_sam3_video_predictor()` request API.

`--stride N` processes every Nth video frame, useful for quick runs.  For a
fair timing comparison use `--stride 1`, the same video/prompt, device, and
output resolution for all candidates.

## Notes

- B/C run SAM on detector boxes; masks are used both for visualization and
  mask-IoU association.  They are deliberately offline prototypes, not an
  online controller.
- D processes a whole video session. It is expected to have materially higher
  memory use; start with a short clipped rollout.
- A requires the `cv2.TrackerCSRT_create` API, supplied by
  `opencv-contrib-python`.
