# N1.5 / N1.7 server-chain diagnostics

This branch provides two native N1.7 server paths:

- `groot`: the existing flexible adapter, kept unchanged as the baseline.
- `groot_explicit`: an exact Kuavo adapter that requires the checkpoint contract
  `head,wrist_left,wrist_right` and
  `left_arm,left_gripper,right_arm,right_gripper` with dimensions `7,1,7,1`.

Both paths use NVIDIA `Gr00tPolicy` and the processor saved with the checkpoint.
Neither path uses the LeRobot GR00T policy implementation. The explicit path is
intended to test whether the flexible adapter's truncation, zero-padding, key
heuristics, or camera fallback caused a deployment mismatch.

## 1. Capture one pre-action simulator observation

The simulator is optional. To load a frame directly from a dual-arm Kuavo
LeRobot dataset, skip this section and use the dataset command in section 3.

Set the following in `configs/deploy/deploy.yaml`:

```yaml
inference:
  capture_observation_path: outputs/diagnostics/sim_reset_observation.pt
  capture_observation_only: true
```

Run the normal simulator evaluation once. It will save the observation directly
after `env.reset()` and exit without sending a model action to the robot.

Use the same task prompt for every comparison. The captured payload contains the
resolved prompt so both servers receive identical language input.

## 2. Start comparison servers

Start the existing N1.5 server from its branch/worktree on port 5555. Start the
baseline N1.7 server from this branch on port 5556:

```bash
python kuavo_server/launch.py groot \
  --checkpoint /path/to/n1d7/checkpoint \
  --port 5556 \
  --strict
```

Optionally start the explicit N1.7 path on port 5557:

```bash
python kuavo_server/launch.py groot_explicit \
  --checkpoint /path/to/n1d7/checkpoint \
  --port 5557
```

The explicit server intentionally fails at startup or request time when modality
keys, dimensions, camera inputs, state shape, or finite-value checks do not match.
It never substitutes the head image for a missing wrist image.

## 3. Probe all servers with the exact same observation

### Read a dataset frame directly (no simulator required)

Run the diagnostic client in the N1.7 environment so the video decoder and
LeRobot loader dependencies are available:

```bash
uv run --project kuavo_model/external_models/gr00tn1d7 \
  python kuavo_server/diagnose.py \
  --dataset-path /path/to/lerobot_dataset \
  --auto-middle-safe-frame \
  --safe-state-threshold 0.85 \
  --server n15=localhost:5555 \
  --server n17=localhost:5556 \
  --server n17_explicit=localhost:5557 \
  --repeats 20 \
  --output outputs/diagnostics/n15_n17_dataset_probe.json
```

The dataset must contain the dual-arm Kuavo modality groups
`left_arm,left_gripper,right_arm,right_gripper` and the three video groups
`head,wrist_left,wrist_right`. The loader concatenates state in exact Kuavo
order and maps the videos to the raw deployment observation keys. Use
`--prompt "..."` only when you want to override the dataset task description.

The report records the resolved dataset path, episode, frame, and video backend
under `observation_source`, making the sampled frame reproducible. With
`--auto-middle-safe-frame`, the client scans from the middle of the dataset and
chooses the first frame whose percentile-normalized state stays within
`--safe-state-threshold` for every state dimension. If no frame meets the
threshold, it uses the best available frame and marks that under
`observation_source.auto_middle_safe_frame.selected_by`.

### Read a previously saved observation

```bash
python kuavo_server/diagnose.py \
  --observation outputs/diagnostics/sim_reset_observation.pt \
  --server n15=localhost:5555 \
  --server n17=localhost:5556 \
  --server n17_explicit=localhost:5557 \
  --repeats 20 \
  --output outputs/diagnostics/n15_n17_server_probe.json
```

For N1.7 servers the report includes:

- raw state, prompt, image hashes, shapes, and dtypes;
- state/action/video modality keys and split state values;
- normalization mode and checkpoint statistics for every state group;
- normalized values and the count clipped to `-1` or `1`;
- action representation/type/format settings;
- whether LoRA parameters were instantiated and nonzero;
- repeated-sampling standard deviation;
- first arm action minus current arm state;
- maximum adjacent step change inside the action chunk;
- raw model-action finiteness per group and the first non-finite indices;
- conversion errors without aborting the rest of the diagnostic report;
- valid/invalid repeat counts and an `invalid_model_output` status.

The N1.5 adapter does not expose the diagnostic endpoint, so the probe falls back
to repeated `select_action_chunk` calls and reports action-side metrics. The raw
observation in the top-level report is shared by all servers, preventing branch
configuration differences from contaminating the comparison.

## 4. Isolate and mitigate N1.7 NaN output

The VLM/action LoRA branch localized the observed final `action.left_arm` NaN
to an earlier mixed-precision failure in the Qwen3 visual backbone. The unsafe
combination is the default BF16 inference path, Flash Attention, and visual LoRA
layers. The action composer only exposes the already-invalid policy output; it
does not create the NaN.

This branch now reports the first non-finite backbone/action-head tensor and the
active dtypes under `precision`. Test the least invasive workaround first:

```bash
python kuavo_server/launch.py groot_explicit \
  --checkpoint /path/to/n1d7/checkpoint \
  --port 5557 \
  --use_fp16 \
  --disable_flash_attention
```

If the error still names `qwen3_backbone.visual`, add the visual FP32 fallback:

```bash
python kuavo_server/launch.py groot_explicit \
  --checkpoint /path/to/n1d7/checkpoint \
  --port 5557 \
  --use_fp16 \
  --disable_flash_attention \
  --visual_fp32
```

`--visual_fp32` restores visual checkpoint tensors through CPU FP32 and runs
visual attention linear layers through the safe FP32 path, so it is slower and
should be used only if FP16 plus eager attention is insufficient. If the first
failure is under `action_head` while all backbone tensors are finite, add
`--action_head_fp32` instead. Full `--use_fp32` is the final high-VRAM fallback;
do not combine it with `--use_fp16`.

After restarting the server, rerun `diagnose.py`. Check:

- `actions.status`: should become `ok`;
- `actions.errors[*].message`: identifies the first non-finite tensor;
- `precision`: confirms model, visual, and action-head dtypes;
- `processor.groups[*].normalized.finite`: must remain `true`.

## 5. Interpret the result

- Baseline and explicit N1.7 actions differ: the old adapter's heuristic mapping
  changed the model input or action output.
- Baseline and explicit actions match: the old adapter is unlikely to be the cause.
- N1.7 normalized state has many saturated elements while training observations do
  not: simulator state and checkpoint statistics/modality semantics are mismatched.
- `config_use_lora=true` but no LoRA parameter tensors are present: the inference
  checkout did not instantiate the LoRA checkpoint architecture correctly.
- Fixed-observation N1.7 variance is much larger than N1.5: investigate stochastic
  flow sampling and RTC/overlap.
- First action is far from the current state although each chunk is smooth: the
  visible robot jump is caused by chunk anchoring, not intra-chunk roughness.
- `actions.status=invalid_model_output`: the checkpoint returned NaN/Inf before
  the Kuavo action could be composed. Inspect `processor.groups` and
  `actions.raw_model_outputs` to distinguish bad normalized input from model output.

Repeat the same probe with an observation reconstructed from a training episode.
If training observations are normal but simulator observations saturate or have
large sampling variance, the failure is a deployment distribution mismatch rather
than checkpoint open-loop quality.
