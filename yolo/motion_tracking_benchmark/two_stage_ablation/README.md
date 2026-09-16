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
| Initializer | YOLO-World, Grounding DINO, SAM 3 text | recall@IoU, side recall, centre error, both-toy recall |
| Tracker | CSRT, SAM 2.1, SAM 3 | recall@IoU, IoU, centre error, both-toy recall |

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

## Initializer commands

```bash
python yolo/motion_tracking_benchmark/two_stage_ablation/generate_yolo_world_initializer.py \
  --backend yolo_world \
  --gt gt_boxes.csv \
  --video-dir /media/data/jiayi/dataset/toy_to_annotat \
  --output outputs/motion_tracking/yolo_world_initializer.csv

python yolo/motion_tracking_benchmark/two_stage_ablation/evaluate_two_stage.py \
  initializer --gt gt_boxes.csv \
  --pred outputs/motion_tracking/yolo_world_initializer.csv \
  --name yolo_world --output outputs/two_stage/yolo_world_initializer.json

python yolo/motion_tracking_benchmark/two_stage_ablation/generate_yolo_world_initializer.py \
  --backend grounding_dino \
  --gt gt_boxes.csv \
  --video-dir /media/data/jiayi/dataset/toy_to_annotat \
  --output outputs/motion_tracking/grounding_dino_initializer.csv

python yolo/motion_tracking_benchmark/two_stage_ablation/evaluate_two_stage.py \
  initializer --gt gt_boxes.csv \
  --pred outputs/motion_tracking/grounding_dino_initializer.csv \
  --name grounding_dino --output outputs/two_stage/grounding_dino_initializer.json

python yolo/motion_tracking_benchmark/two_stage_ablation/generate_yolo_world_initializer.py \
  --backend sam3 \
  --gt gt_boxes.csv \
  --video-dir /media/data/jiayi/dataset/toy_to_annotat \
  --output outputs/motion_tracking/sam3_text_initializer.csv

python yolo/motion_tracking_benchmark/two_stage_ablation/evaluate_two_stage.py \
  initializer --gt gt_boxes.csv \
  --pred outputs/motion_tracking/sam3_text_initializer.csv \
  --name sam3_text --output outputs/two_stage/sam3_text_initializer.json
```

Despite its historical filename, `generate_yolo_world_initializer.py` now
supports all three initializer backends through `--backend`. It uses the GT CSV
only as the list of visible annotated `(rollout, frame)` samples. GT coordinates
are not passed to a detector or used for identity assignment. Frames are scanned
sequentially so identities created from the initial image-side order remain
stable if the toys cross; only requested sample frames are written.

## Common tracker seeds and tracker commands

Create the single shared seed manifest first:

```bash
python yolo/motion_tracking_benchmark/two_stage_ablation/make_seed_manifest.py \
  --gt gt_boxes.csv --output outputs/two_stage/seeds.csv

python yolo/motion_tracking_benchmark/two_stage_ablation/run_seeded_tracker.py \
  --backend csrt \
  --seeds outputs/two_stage/seeds.csv \
  --video-dir /media/data/jiayi/dataset/toy_to_annotat \
  --output outputs/two_stage/csrt_predictions.csv

python yolo/motion_tracking_benchmark/two_stage_ablation/run_seeded_tracker.py \
  --backend sam2 \
  --seeds outputs/two_stage/seeds.csv \
  --video-dir /media/data/jiayi/dataset/toy_to_annotat \
  --sam2-config /media/data/jiayi/sam2/configs/sam2.1/sam2.1_hiera_s.yaml \
  --sam2-checkpoint /media/data/jiayi/sam2/checkpoints/sam2.1_hiera_small.pt \
  --output outputs/two_stage/sam2_predictions.csv

python yolo/motion_tracking_benchmark/two_stage_ablation/run_seeded_tracker.py \
  --backend sam3 \
  --seeds outputs/two_stage/seeds.csv \
  --video-dir /media/data/jiayi/dataset/toy_to_annotat \
  --output outputs/two_stage/sam3_predictions.csv
```

Every run writes canonical predictions directly and a sibling
`*.metadata.json` recording the exact seed file and resolved videos. Evaluate
each tracker with the same command shape:

```bash
python yolo/motion_tracking_benchmark/two_stage_ablation/evaluate_two_stage.py \
  tracker --gt gt_boxes.csv --pred outputs/two_stage/csrt_predictions.csv \
  --name csrt --output outputs/two_stage/csrt_tracker.json

python yolo/motion_tracking_benchmark/two_stage_ablation/evaluate_two_stage.py \
  tracker --gt gt_boxes.csv --pred outputs/two_stage/sam2_predictions.csv \
  --name sam2 --output outputs/two_stage/sam2_tracker.json

python yolo/motion_tracking_benchmark/two_stage_ablation/evaluate_two_stage.py \
  tracker --gt gt_boxes.csv --pred outputs/two_stage/sam3_predictions.csv \
  --name sam3 --output outputs/two_stage/sam3_tracker.json
```

The evaluator matches by `(rollout, frame, side)`. It intentionally does not
accept a tracker-specific initializer file: tracker runs must share a separate
seed manifest, recorded beside the experiment results.

## Existing benchmark outputs

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
