# Frozen LingBot progress features

For the current full-dataset manual workflow and output root
`/media/data/jiayi/outputs/lingbot_progress_pilot`, follow
[RUN_LINEAR_PROBE.md](RUN_LINEAR_PROBE.md). It provides ordered commands for
extracting all 141 episodes, training on 99 train episodes, comparing models on
21 validation episodes, and holding out 21 test episodes.
Paths below describe the earlier smoke/pilot artifacts; the trainer's
current defaults use the new output root. No experiment is started by reading
the guide or running CLI help.

Run from the repository root using the existing `lingbotvla` conda environment.
This standalone script never creates a robot/server adapter, starts ROS, or
calls `sample_actions` / `predict_velocity`. It reuses
`LingbotVLAServer.load_vla` (strict merged-weight loading) and `FeatureTransform`.
The loader's server class name does not start a network server when its loading
method is called. Model parameters are frozen and evaluated under inference mode.

```bash
/home/kuavo/miniforge3/envs/lingbotvla/bin/python -B progress_monitor/extract_lingbot_features.py --smoke --validate-only
/home/kuavo/miniforge3/envs/lingbotvla/bin/python -B progress_monitor/extract_lingbot_features.py --smoke
```

The smoke command selects only the first 20 valid frames of Episode 000.
CUDA and FlashAttention must work. There is deliberately no CPU fallback for the
custom vision encoder. The original checkpoint is read without changes; the
saved training YAML supplies architecture flags and the local Qwen directory
supplies config/tokenizer/image processing assets. No adapters are injected.

All three pooling modes are saved by default. Select a subset with
`--pooling all`, `--pooling vision`, or `--pooling language`.
The extracted tensor is the final normalized `outputs_embeds[0]` from
`QwenvlWithExpertModel.forward`, with no action suffix and no KV-cache retention.
Each output vector has dimension 2048. All-token pooling averages valid vision
and instruction tokens together; vision/language pooling restricts that mask.
The exact training image path is used: torchvision 224-square Resize, uint8
conversion in FeatureTransform, then Qwen image processing. Online use should
match this preprocessing and the dataset instruction.

Default outputs are outside Git:

```text
/media/data/jiayi/outputs/lingbot_progress_features/
  smoke_episode_000_20/train/episode_000.npz
  smoke_episode_000_20/report.json
  full/{train,val,test}/episode_NNN.npz
  full/report.json
```

Each NPZ stores `features_all`, `features_vision`, and `features_language` as
float16 `[T_episode,2048]` arrays (only requested modes), preserving every label
row. Unextracted rows are NaN; use `feature_valid & valid & ~overlap` for training.
It also stores phase IDs, timestamps, frame indices, cycle/progress labels,
episode index, original split, and extraction metadata. Cycle/progress labels
are alignment metadata, never model inputs. Normalization for a later probe
must be fitted using training episodes only.

Dataset rows are checked against label frame indices, timestamps, and states.
Only three RGB videos are decoded, using v3 metadata offsets and existing
LeRobot video decoding. No future actions are read. Zero dummy action buffers
satisfy FeatureTransform's interface and are discarded before model execution.
Errors fail explicitly; no random replacement of failed samples is allowed.
Existing output files are never overwritten.

Full extraction is opt-in by running without `--smoke`; it is not part of the
smoke test. `--episodes 0 1` limits extraction, and `--batch-size` defaults to 1.
Completed episode files survive an interrupted run; resume by explicitly
selecting remaining episodes. Video decoding currently happens per batch, so a
larger batch can reduce seek overhead. Benchmark before a full extraction.
The report records successful strict loading and peak allocated/reserved CUDA
memory. No commit, merge, push, training, or robot deployment is performed.

## Linear Probe pilot

The pilot uses 12 original training episodes and 4 original validation episodes,
selected from balanced, Flip-heavy, and No-Flip-heavy strata using cycle counts.
Within each stratum, episode IDs span the available range. Selection never reads
features or evaluation outcomes. No test episodes enter the selection or probes.

Use [RUN_LINEAR_PROBE.md](RUN_LINEAR_PROBE.md) for directly copyable commands.
It uses a separate output root, `/media/data/jiayi/outputs/lingbot_progress_pilot`,
and explicitly passes paths so it never mixes the earlier experiment artifacts
with a new run. The extractor retains its legacy output default; always use the
guide's explicit `--output` and `--run-name` arguments.

Use the environment path shown above; GPU extraction and writes under `/media`
require host execution permissions in Codex. Existing selections, episode files,
and probe output directories are protected against overwrites.

The trainer reads frozen NPZ features, verifies every selected valid frame and
the original split/provenance, and fits `torch.nn.Linear(2048,6)` independently
for each representation and loss weighting. All three modes × both cross-entropy
variants run by default. `--weighting weighted` or `unweighted` restricts it.
Weighted loss uses `N_train / (6 * class_count_train)`.

Feature mean/std come only from training frames. Near-constant dimensions use a
unit denominator. Parameters are trained with AdamW (lr 0.001, weight decay
0.0001, batch size 512), seed 42, deterministic operations, up to 50 epochs.
Early stopping uses validation macro-F1, patience 8, minimum improvement 0.0001.
The default probe device is CPU; this small model does not need the VLA or GPU.
The reported validation cross-entropy is always unweighted for comparability.

New results live under `/media/data/jiayi/outputs/lingbot_progress_pilot/probes/`.
Steps 6–9 of the guide use separate standard/weighted experiment directories:

- `summary.json` and `report.md`: all six runs, baseline, class metrics, confusion
  matrices, directional P2a/P3 errors, and per-episode validation results.
- `{pooling}_{weighting}/best.pt`: linear weights/bias, training-only mean/std,
  class order/weights, selected epoch, seed, episodes, and extraction provenance.
- Per-run `history.json`, `metrics.json`, and `validation_predictions.npz`.

At inference, apply the saved mean/std to the chosen pooled feature before the
linear layer. No VLA gradients, action denoising, smoothing, or future inputs are
used. The same validation episodes select epochs and compare the six variants;
these pilot scores are not an untouched test estimate.

`verify_features.py` checks all pooling arrays against the existing labels and
official splits. `evaluate_linear_probe.py` evaluates saved heads on CPU and
exports per-class metrics, predictions, CSV/optional PNG confusion matrices, and
a comparison with the training-majority baseline. Both are invoked with `-m`
from the repository root. `--scope full` admits the complete original splits;
training still reads only train/validation features, and test evaluation requires
an explicit later `--split test` invocation.
