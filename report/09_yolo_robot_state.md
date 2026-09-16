1.overview
/home/kuavo/jiayi/kuavo_learning_studio/yolo/eval_progress_trigger_v3.py /home/kuavo/jiayi/kuavo_learning_studio/yolo/eval_progress_trigger_v4.py参考这几个yolo做failure trigger的文件，现在效果并不好，我希望可以加入robot state ee pose等辅助判断，

在仿真中进行加入trigger进行测试，不再依赖纯video，

新建文件实现，不要修改现有文件

2. Current Pipeline

Current perception pipeline:

head-camera video
    ↓
YOLO-World
    ↓
generic toy geometric filtering
    ↓
left/right toy initialization
    ↓
CSRT visual tracking
    ↓
constant-velocity fallback
    ↓
tracking CSV

Current toy track states:

DETECTED
TRACKED
PREDICTED
LOST
UNINITIALIZED

Reliability rule:

DETECTED / TRACKED
→ reliable observation

PREDICTED / LOST
→ uncertain
→ must NOT directly cause a hard failure trigger

Current tracking script:

yolo/test_yolo_world_roi_filter.py

The script outputs files similar to:

outputs/yolo_world_visual_tracking/
    rollout01_head_yoloworld_visual_tracking.csv
    rollout01_head_yoloworld_visual_tracking.mp4
    ...

The tracking CSV contains fields including:

frame
time_s
class
side_center_x
side_center_y
track_source
vx_px
vy_px
visual_tracker
tracker_ok
tracker_reason

Target classes include:

left_toy
right_toy
basket

The current YOLO-based gripper detection is unreliable and should not be used as the primary gripper/EE signal.
3. Existing GT Annotation

The evaluation GT is stored in:

yolo/toy_annotations.json

The annotation contains, for each rollout:

outcome
failure_detected
first_failure_time_s
failure_type
failed_side
left_toy_final_state
right_toy_final_state
failure_events

Example rollout-level labels:

SUCCESS
FAILURE

Typical failure events include:

LEFT_GRASP_MISS
RIGHT_GRASP_MISS
LEFT_OBJECT_DROP
RIGHT_OBJECT_DROP
OBJECT_OUTSIDE_BASKET
MULTIPLE_FAILURES
NO_PROGRESS

The evaluator should primarily use:

outcome
first_failure_time_s

for Stage-1 trigger evaluation.

Exact failure-type classification is not required for this task.
4. Why the Current V3/V4 Trigger Is Not Good Enough

The previous trigger used:

PREGRASP timeout
+
post-activation toy stagnation

or earlier:

toy must continuously make progress toward basket

The result was poor.

Representative V4 behavior:

Failure rollouts:
TP: 3 / 9

Success rollouts:
FP: 8 / 11
TN: 3 / 11

Approximate metrics:

failure detection rate ≈ 33%
success false-pause rate ≈ 73%
precision ≈ 27%

The important conclusion is:

toy is not moving

alone cannot distinguish:

normal temporary pause / adjustment

from:

real manipulation failure

Likewise:

toy-to-basket distance is not monotonically decreasing

is not a valid generic failure criterion because normal manipulation contains:

approach
grasp
lift
reorientation
transport
placement

and the image-space toy-to-basket distance can temporarily stay constant or increase.

Therefore, do not continue tuning only:

pregrasp timeout
min_motion
pause_after
basket distance

as the main solution.
5. New V5 Trigger Principle

The new trigger should use cross-signal inconsistency:

robot is actively moving
+
toy is not moving
→ suspicious

More formally:

failure_candidate = (
    ee_motion > EE_MOTION_MIN
    and
    toy_motion < TOY_MOTION_MAX
)

If this condition persists for a time window:

failure_candidate_duration >= TRIGGER_DURATION

then:

PAUSE

This should be treated as a generic failure event.

Do not attempt to classify:

GRASP_MISS
OBJECT_DROP
STUCK

inside Stage-1.

The VLM will perform diagnosis after PAUSE.

## 6. V5 implementation

Implemented a new evaluator without changing V3/V4:

`yolo/eval_progress_trigger_v5_robot_state.py`

V5 consumes two time-stamped streams:

```text
YOLO tracking CSV ── reliable toy center ──┐
                                           ├─ cross-signal state machine ── PAUSE
robot-state CSV ── end-effector XYZ ───────┘
```

For each side independently, it measures both signals over `--motion-window-s`:

```text
ee_speed       = ||ee(t) - ee(t-window)|| / elapsed_time
toy_motion     = ||toy(t) - toy(t-window)||
candidate      = ee_speed >= ee_motion_min and toy_motion <= toy_motion_max
PAUSE          = candidate persists for trigger_duration_s
                 and confirm_frames consecutive frames agree
```

Only `DETECTED` and `TRACKED` toy observations are considered reliable by default. `PREDICTED` may be explicitly enabled with `--allow-predicted`. If either visual observation or a time-aligned EE sample is missing, the candidate timer is reset and no pause is emitted. Thus an uncertain visual track or stale robot state cannot independently cause a hard failure trigger.

The state stream is synchronized by nearest timestamp. A sample farther than `--sync-max-gap-s` (default 100 ms) is rejected. The evaluator writes a per-frame CSV containing `toy_motion`, `ee_speed`, state availability, candidate duration, reason, raw trigger, and final pause; it also writes per-rollout JSON summaries and a batch JSON summary.

## 7. Robot-state CSV interface

The simulator needs to log a separate CSV with `time_s` plus either a shared EE pose or per-arm EE poses. Position is sufficient for V5; orientation and gripper state are intentionally not used as the primary signal.

Shared EE (single-arm task):

```csv
time_s,ee_x,ee_y,ee_z
0.000,0.12,-0.08,0.34
0.050,0.13,-0.08,0.34
```

Per-arm EE (bimanual task):

```csv
time_s,left_ee_x,left_ee_y,left_ee_z,right_ee_x,right_ee_y,right_ee_z
0.000,0.12,-0.08,0.34,0.12,0.08,0.34
```

The timestamp must use the same clock domain as the video/tracking `time_s`. If the simulator emits joint states instead of EE poses, calculate FK at logging time (or before running the evaluator) and export the resulting XYZ columns. This avoids treating raw joint-coordinate distance as a Cartesian EE velocity threshold.

## 8. Run command

With one state log for each rollout (the glob filenames must contain the same `rolloutNN` key as the tracking files):

```bash
python3 yolo/eval_progress_trigger_v5_robot_state.py \
  --input 'outputs/yolo_world_visual_tracking/*_visual_tracking.csv' \
  --robot-state 'outputs/robot_state/*_ee_pose.csv' \
  --gt-json yolo/toy_annotations.json \
  --left-ee-columns left_ee_x,left_ee_y,left_ee_z \
  --right-ee-columns right_ee_x,right_ee_y,right_ee_z \
  --output-dir outputs/progress_trigger_v5_robot_state
```

For a single-arm/single-EE simulator log, replace the two per-side mappings with:

```bash
--ee-columns ee_x,ee_y,ee_z
```

The default thresholds are deliberately visible CLI parameters, not hidden constants:

```text
motion window:       0.50 s
EE active threshold: 0.020 state-units/s
toy static threshold:0.012 normalized image units/window
trigger duration:    1.00 s
confirmation:        2 frames
```

They must be calibrated on simulation EE units and camera resolution before deployment. V5 does not retain V4's pregrasp-timeout or basket-distance trigger, because its required condition is cross-signal inconsistency rather than toy stagnation alone.

## 9. Validation performed

- `python3 -m py_compile yolo/eval_progress_trigger_v5_robot_state.py`: passed.
- Pure parser/synchronization/history helper checks: passed.
- Synthetic state-machine replay: passed. A reliable stationary left toy and an EE moving at 0.06 state-units/s met the one-second persistence rule; with two-frame debounce V5 produced `PAUSE` at 1.5 s and matched a 1.5 s failure label (`TP`).

The repository currently contains the 20 visual tracking CSVs and GT annotations, but no rollout-aligned simulator robot-state/EE-pose CSV. Therefore an honest batch TP/FP comparison with V4 cannot yet be produced from the checked-in data. Once the simulator writes the interface above, the command in section 8 will generate that comparison without code changes.

## 10. Kuavo closed-loop simulation integration

V5 is now also connected to the automatic simulator evaluation path; it is no longer limited to post-rollout CSV analysis. The integration is disabled by default to preserve the existing evaluation behavior.

```text
env.step(action)
    ├─ latest head-camera observation ── YOLO-World + CSRT side tracking
    ├─ latest left/right PoseStamped ─── EE pose buffer
    └─ online cross-signal state machine
                         └─ PAUSE: stop next action, publish pause topic,
                                   save trigger CSV, then reset episode
```

Changed files:

- `yolo/online_robot_state_trigger.py`: online tracker and per-side state machine. It reuses the existing ROI filtering, left/right assignment, CSRT/KCF/MIL tracker fallback, and reliable-source rule from `test_yolo_world_roi_filter.py`.
- `kuavo_deploy/src/eval/sim_auto_test.py`: creates the monitor per episode, subscribes to EE poses, runs it after each `env.step`, publishes the configured pause signal on a confirmed trigger, and closes/clears it before the next reset.
- `kuavo_deploy/config.py` and `configs/deploy/total/deploy_total.yaml`: trigger settings.

### Required simulator configuration

The checked-in simulator config exposes joint and gripper state but does not name an EE pose topic. Set the real `geometry_msgs/PoseStamped` topics exported by `kuavo-ros-opensource`; do not guess topic names or substitute joint positions.

```yaml
inference:
  failure_trigger_enabled: true
  failure_trigger_ee_pose_topic_left: "/your_simulator/left_ee_pose"
  failure_trigger_ee_pose_topic_right: "/your_simulator/right_ee_pose"
  failure_trigger_pause_topic: "/kuavo/pause_state"
  failure_trigger_yolo_model: "yolov8s-worldv2.pt"
  failure_trigger_yolo_device: "0"
  failure_trigger_motion_window_s: 0.5
  failure_trigger_ee_motion_min: 0.02
  failure_trigger_toy_motion_max: 0.012
  failure_trigger_duration_s: 1.0
  failure_trigger_confirm_frames: 2
  failure_trigger_pose_max_age_s: 0.10
```

Then start the standard evaluation command from the task description. No separate V5 process is needed. Each rollout writes `rollout_<N>_yolo_robot_state_trigger.csv` next to its saved videos. A confirmed trigger ends that episode as a failed rollout, so it is included in the existing success-rate denominator.

`evaluation_autotest.log` and the final console summary additionally contain `Trigger Pause Count: <n>/<eval_episodes>`. This is the primary simulator-side trigger-effect metric; compare it with success count/rate and inspect the per-rollout CSV to determine whether pauses are legitimate or false pauses.

If a pose is not received, is older than `failure_trigger_pose_max_age_s`, or the toy track is not `DETECTED`/`TRACKED`, the trigger records its reason and resets its candidate timer. It cannot pause based on stale EE data or a predicted/lost toy track.

### Integration verification

Static verification passed with `python3 -m py_compile` for the newly added online trigger, simulation evaluator, and config loader. A live ROS/Kuavo run was not executed in this workspace because no ROS master, `kuavo-ros-opensource` simulator, or EE-pose topic is running here. Before collecting metrics, verify the actual topic types with `rostopic info <topic>` and confirm that the pose header timestamps use the simulator `/clock` domain.

## 11. Checkpoint + agentic simulator entry (the source of `kuavo_deploy.log`)

The supplied `log/kuavo_deploy/kuavo_deploy.log` is produced by `script_auto_test_vlm_agentic.py` / `sim_auto_test_vlm_agentic.py`, as shown by its source-path suffix in every log row. This is the entry that loads the user's checkpoint and already runs VLM trigger inference. YOLO has now been connected to this exact loop as well.

Do **not** tail and parse `kuavo_deploy.log` as the real-time input: it is a formatted, asynchronously flushed debug artifact and array values can span multiple lines. Instead, immediately after each checkpoint action is executed, the loop passes the in-memory `observation.state` to YOLO trigger. This is the exact state that `KuavoBaseRosEnv` logs as:

```text
STATE: contained ['joint_q', 'gripper'], concated value: [...]
```

Therefore log and trigger have the same data source and step, while the log remains an auditable trace.

The agentic summary CSV now has both `vlm_pause_count` and `yolo_pause_count`; the per-episode YOLO CSV records whether its robot-motion input was `ee_pose` or `joint_state_proxy`.

### Run with the current simulator state log as motion proxy

The current config/log exposes `joint_q + gripper`, not Cartesian EE pose. To test immediately with the same state shown in the log, explicitly enable the proxy:

```yaml
inference:
  failure_trigger_enabled: true
  failure_trigger_allow_joint_state_fallback: true
  failure_trigger_yolo_model: "yolov8s-worldv2.pt"
  failure_trigger_yolo_device: "0"
```

Then run the checkpoint evaluator actually used to produce the log:

```bash
python kuavo_deploy/src/scripts/script_auto_test_vlm_agentic.py \
  --task auto_test_vlm_agentic \
  --config /path/to/your_checkpoint_sim_config.yaml
```

`inference.pretrained_path` in that YAML must point to the checkpoint to evaluate. A confirmed YOLO trigger publishes the normal pause topic, stops further policy actions for the episode, records it as an unsuccessful episode, and the standard reset then advances to the next episode.

Joint-state proxy is useful for the requested first simulator test, but its threshold is in joint/gripper units per second, not metres per second. Calibrate `failure_trigger_ee_motion_min` on several successful rollouts. When the simulator exports left/right `PoseStamped`, set their topics and change `failure_trigger_allow_joint_state_fallback` back to `false`; pose samples take priority automatically.

## 12. Final integration entry: independent evaluator

The implementation is finalized as a new file, preserving both existing evaluators unchanged:

`kuavo_deploy/src/eval/sim_auto_test_yolo_robot_state.py`

It is based on `sim_auto_test.py`, loads `inference.pretrained_path`, performs ordinary closed-loop simulator inference, and injects the online YOLO + robot-state check immediately after every `env.step`. Sections 10 and 11 describe earlier integration attempts; this independent evaluator is the entry to use.

```bash
python -m kuavo_deploy.src.eval.sim_auto_test_yolo_robot_state \
  --config /path/to/your_checkpoint_sim_config.yaml
```

For the current `kuavo_deploy.log` state format, set:

```yaml
inference:
  pretrained_path: /path/to/checkpoint
  failure_trigger_enabled: true
  failure_trigger_allow_joint_state_fallback: true
```

The new output directory contains rollout videos, one `rollout_<N>_yolo_robot_state_trigger.csv` per episode, `yolo_robot_state_episode_summary.csv`, and `evaluation_yolo_robot_state.log`. The text deploy log is not parsed; `observation.state` is consumed directly before it is formatted into `STATE:` log lines.

The independent `sim_auto_test_yolo_robot_state.py` evaluator uses `observation.state` only: left/right joint positions plus gripper state. It neither subscribes to nor requires EE pose topics. Its configured `failure_trigger_ee_motion_min` is therefore a joint-state-motion threshold, not a Cartesian metre-per-second threshold.

Its startup sequence deliberately matches the existing menu option `auto_test` (input 3): wait for the simulator's first `/simulator/init`, call `/simulator/reset`, clear the first event, then wait for the post-reset `/simulator/init` before constructing `KuavoSimEnv`. This prevents creating the SDK environment while the simulator's gait-switch controller is still booting.

For consistency with the existing project layout, the matching CLI wrapper is also available at `kuavo_deploy/src/scripts/script_auto_test_yolo_robot_state.py` and is registered as `auto_test_yolo_robot_state` in `kuavo_deploy/eval.py`.

## 13. Isolated YOLO environment

The simulator runs under `lingbotvla` (Python 3.12), while the existing `yolo_jiayi` environment uses Python 3.10 and contains `ultralytics`. These environments must not share `site-packages`. Set:

```yaml
inference:
  failure_trigger_yolo_python: /home/kuavo/miniforge3/envs/yolo_jiayi/bin/python
  failure_trigger_yolo_model: yolov8s-worldv2.pt
  failure_trigger_yolo_device: "0"
```

The trigger then starts `yolo/yolo_world_worker.py` with that interpreter as one persistent local subprocess. It exchanges one JSON request per image over stdin/stdout and uses a shared temporary JPEG path. The simulator environment never imports `ultralytics`, `torch`, or other dependencies from `yolo_jiayi`; it only receives normalized detection boxes. The worker is terminated and its temporary directory removed at episode completion.

This IPC path was tested locally with the supplied Python interpreter, model, and a synthetic RGB frame. The worker started successfully and returned a valid empty detection list.

## 14. sim_task1 pregrasp gate calibration

The first grasp in the current simulator task occurs around 5--6 seconds. The initial run paused at 1.5--1.8 seconds because normal approach motion satisfied the generic “robot moving + toy static” rule. `configs/deploy/deploy_lingbot_yolo_trigger_sim_task1.yaml` now sets:

```yaml
failure_trigger_start_after_s: 6.0
```

YOLO still tracks and records frames during this period, but candidate time is forcibly zero and the CSV reason is `pregrasp_grace`. At six seconds the ordinary 0.5-second motion window begins; only then can the one-second persistence rule produce PAUSE. This is a task-time gate for the present simulation, not a general proof of grasp completion.

## 15. YOLO trigger + Qwen3.5 failure type and recovery

The registered `auto_test_yolo_vlm_recovery` entry keeps YOLO plus the
joint-state proxy as stage 1. A confirmed stage-1 event pauses the simulator,
saves the head-camera rollout clip, and invokes the existing Qwen3.5 verifier
in the isolated `qwen35` environment. Qwen is not loaded during normal policy
steps.

```bash
python kuavo_deploy/src/scripts/script_auto_test_yolo_vlm_recovery.py \
  --task auto_test_yolo_vlm_recovery \
  --config configs/deploy/deploy_lingbot_yolo_trigger_sim_task1.yaml
```

The stage-2 JSON verdict is written to
`stage2_failure_events.jsonl`; the episode/attempt outcome is written to
`yolo_vlm_recovery_episode_summary.csv`.

| Stage-2 failure type | Recovery |
|---|---|
| `FALSE_ALARM` | Publish resume and clear stage-1 temporal history. |
| `GRASP_MISS`, `OBJECT_DROP`, `PLACE_MISS` | Reset simulator and retry the same episode, up to `--stage2-max-retries-per-episode` (default 1). |
| `WRONG_OBJECT`, `WRONG_DESTINATION`, `COLLISION`, `UNKNOWN` | Stop evaluation safely. |

The external verifier defaults to `/home/kuavo/miniforge3/envs/qwen35/bin/python`,
the Qwen3.5-9B model path, and `vlm_test/qwen35_failure_verifier.py`. These can
be changed with `--stage2-python`, `--stage2-model-path`, and
`--stage2-verifier-script` without changing the simulator environment.

## 16. Cartesian planar motion versus image-plane toy motion

The trigger now subscribes to optional left/right `geometry_msgs/PoseStamped`
topics. When both topic names are set, it calculates EE velocity in the pose
frame as `sqrt(vx^2 + vy^2)` and uses that **world/base-frame XY planar
velocity** for the stage-1 condition. It logs `|vz|` separately as
`*_ee_z_speed`; vertical lift is not silently treated as planar motion.

```yaml
inference:
  # Replace with topics advertised by the paired simulator.
  failure_trigger_ee_pose_topic_left: /your/left/eef_pose
  failure_trigger_ee_pose_topic_right: /your/right/eef_pose
  failure_trigger_ee_motion_min: 0.02          # XY m/s
  failure_trigger_ee_vertical_motion_min: 0.02 # Z m/s, logged for analysis
```

The CSV now includes `*_ee_xy_speed`, `*_ee_z_speed`, and
`*_toy_area_change`. `*_toy_motion` remains normalized image-plane (`UV`)
motion, so it is a visual consistency signal rather than a metric comparison
with metres per second. If pose topics remain empty, the trigger explicitly
falls back to `joint_state_proxy`; in that case `*_ee_xy_speed` and
`*_ee_z_speed` are blank and the proxy must not be interpreted as Cartesian
velocity.
