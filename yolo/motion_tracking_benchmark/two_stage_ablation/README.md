# Two-stage initializer / tracker ablation

This experiment separates two failure sources that are confounded by an
end-to-end score:

```text
Layer 1: initializer
video -> {left_toy/right_toy initial boxes}

Layer 2: temporal tracker
the same initial boxes -> CSRT | SAM 2.1 | SAM 3 trajectories
```

## Protocol

1. Annotate the two toy boxes for sampled frames in `gt_boxes.csv`.
2. Run each initializer and export one row per predicted box in the common
   `initializer_predictions.csv` schema.
3. Run `evaluate_two_stage.py initializer` once per initializer.
4. Choose **one fixed seed manifest** for every tracker. It may be manually
   annotated or selected from an initializer, but it must not vary by tracker.
5. Export each tracker result in `tracker_predictions.csv` schema and run
   `evaluate_two_stage.py tracker` once per tracker.

The fair matrices are:

| Layer | Conditions | Main metrics |
| --- | --- | --- |
| Initializer | YOLO-World, Grounding DINO, SAM 3 text | recall@IoU, side recall, centre error, detection stability |
| Tracker | CSRT, SAM 2.1, SAM 3 | visible coverage, IoU, centre error, ID consistency, drift rate |

## CSV schemas

`gt_boxes.csv` and tracker predictions:

```csv
rollout,frame,side,x1,y1,x2,y2,visible
rollout21,0,left_toy,270,330,302,372,1
```

Initializer predictions add `score`:

```csv
rollout,frame,side,x1,y1,x2,y2,score
rollout21,0,left_toy,270,330,302,372,0.82
```

`side` must be `left_toy` or `right_toy`. Use the initial physical identity;
do not relabel a toy after it crosses the image centre.

## Commands

```bash
python yolo/motion_tracking_benchmark/two_stage_ablation/evaluate_two_stage.py \
  initializer --gt gt_boxes.csv --pred yolo_world_initializer.csv \
  --name yolo_world --output outputs/two_stage/yolo_world_initializer.json

python yolo/motion_tracking_benchmark/two_stage_ablation/evaluate_two_stage.py \
  tracker --gt gt_boxes.csv --pred sam2_tracks.csv \
  --name sam2 --output outputs/two_stage/sam2_tracker.json
```

The evaluator matches by `(rollout, frame, side)`. It intentionally does not
accept a tracker-specific initializer file: tracker runs must share a separate
seed manifest, recorded beside the experiment results.

## Common seed file and existing benchmark outputs

Build the seed file from GT (the default chooses the earliest frame where both
toys are visible for each rollout):

```bash
python yolo/motion_tracking_benchmark/two_stage_ablation/make_seed_manifest.py \
  --gt gt_boxes.csv --output seeds.csv
```

Use the exact `seeds.csv` in all CSRT, SAM 2.1, and SAM 3 runs.  The runner for
each tracker should record that file path in its output metadata; it must not
redetect or choose a tracker-specific initialization frame.

Current benchmark `tracks.csv` files with `left_toy` / `right_toy` labels can
be normalized before tracker evaluation:

```bash
python yolo/motion_tracking_benchmark/two_stage_ablation/normalize_tracks.py \
  --input-glob 'outputs/two_stage/sam2/**/tracks.csv' \
  --output outputs/two_stage/sam2_predictions.csv
```

For the initializer layer, evaluate predictions on exactly the same sampled
frames. A generic `toy` detection must first be assigned to the physical
`left_toy` / `right_toy` identity using the annotated seed frame; positional
left/right per frame is not a valid identity rule after objects move.
