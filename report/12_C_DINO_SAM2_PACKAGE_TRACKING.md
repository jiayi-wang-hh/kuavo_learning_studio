# C: Grounding DINO + SAM 2.1 package tracking

This branch includes Option C as an offline visual tracking candidate for the
normal and recovery clips. The pipeline does not load a VLA model or use tactile
or robot-state inputs.

## Pipeline

1. Grounding DINO proposes separate blue mailer, pink envelope, yellow tray,
   robot arm, and robot gripper boxes.
2. Oversized and implausible-aspect object proposals are filtered before NMS.
   Package detections are associated over time and each distinct package is
   seeded at its first accepted detection, up to six instances.
3. Blue and pink package boxes use a color-conditioned HSV mask as the SAM 2.1
   seed when enough matching pixels exist. Other objects fall back to box seeds.
4. The yellow tray is matched over sampled frames and frozen as a median static
   anchor. Robot boxes are detected with a dedicated arm/gripper prompt. Package
   candidates mostly covered by an arm box are rejected; propagated masks are
   also clipped against the nearby arm boxes.
5. SAM 2.1 propagates each package mask through the remaining video frames.

## Current run

CUDA inference used device 1, Grounding DINO Tiny, and the SAM 2.1 Hiera Small
checkpoint. Package detection sampled every five frames; SAM 2.1 propagated at
the source video frame rate.

| Clip | First seed frame | Package seeds | Tray detections fused |
|---|---:|---:|---:|
| Normal | 5 (0.5 s) | 5 | 46 |
| Recovery | 0 (0 s) | 6 | 42 |

The first Normal frames before 0.5 s have only the tray anchor because no
package proposal passed the detection filters yet. The left arm is reduced by
color-seeded masks and arm-box clipping, but the arm detector is still a box
heuristic and can leave errors when the arm overlaps a similarly colored bag.
Grounding DINO can also confuse package color labels. Review the annotated
videos before using this as a robot control signal.

## Artifacts

Generated files are under
`outputs/motion_tracking/two_clip_comparison_multi_target/c_dino_sam2_colorseed_final/`:

- `normal/annotated.mp4`, `normal/tracks.csv`, `normal/summary.json`
- `recovery/annotated.mp4`, `recovery/tracks.csv`, `recovery/summary.json`
- `batch_summary.json`

The `outputs/` directory is git-ignored in this repository, so these local
videos are copied into the branch worktree but are not part of a normal Git
commit.
