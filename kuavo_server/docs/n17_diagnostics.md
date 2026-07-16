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
- maximum adjacent step change inside the action chunk.

The N1.5 adapter does not expose the diagnostic endpoint, so the probe falls back
to repeated `select_action_chunk` calls and reports action-side metrics. The raw
observation in the top-level report is shared by all servers, preventing branch
configuration differences from contaminating the comparison.

## 4. Interpret the result

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

Repeat the same probe with an observation reconstructed from a training episode.
If training observations are normal but simulator observations saturate or have
large sampling variance, the failure is a deployment distribution mismatch rather
than checkpoint open-loop quality.
