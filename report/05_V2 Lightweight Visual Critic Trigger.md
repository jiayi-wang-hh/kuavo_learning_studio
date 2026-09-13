V2 Lightweight Visual Critic Trigger — Implementation Specification

1. Objective

Replace the current Qwen2.5-VL Stage-1 real-time trigger with a lightweight V2 Visual Critic Trigger.

The new V2 trigger should act as a fast visual execution monitor that decides when the current robot execution should be interrupted, while the existing Qwen Stage-2 remains responsible for why the failure happened and how to recover.

The intended separation is:

FAST PATH
Camera frame + current subtask
        ↓
Lightweight Visual Critic
        ↓
PROGRESSING / STALLED / FAILURE / SUCCESS / UNKNOWN
        ↓
STALLED / FAILURE
        ↓
PAUSE current policy

SLOW PATH
        ↓
Existing Qwen Stage-2
        ↓
Failure diagnosis
        ↓
Retry / recovery / replan

The first implementation should be inference-only / zero-shot, with no training code required yet. However, the software architecture must support replacing the initial model with a later task-specific fine-tuned checkpoint.

2. Background and Motivation

The current pipeline uses Qwen2.5-VL to inspect a short video window and return a trigger decision such as:

PAUSE
CONTINUE

The current logs show two major limitations:

The semantic judgment can be correct, but local guard logic may overwrite PAUSE back to CONTINUE.

Qwen inference is too slow for a real-time trigger. A result generated from an early source step may only return close to the end of the episode.

Example failure pattern:

source_step=30
observed_step=156
phase=PREGRASP
raw=PAUSE
final=CONTINUE
guard=PREGRASP_NO_COMPLETED_ATTEMPT

The new design should therefore:

remove slow large-VLM reasoning from the real-time trigger path;

remove natural-language keyword matching from trigger decisions;

replace free-form trigger outputs with a structured execution-state interface;

run asynchronously and never block VLA execution;

avoid stale observation queues;

keep Qwen only for event-triggered semantic diagnosis and recovery reasoning.

3. Reference Architecture

The design is inspired by the Critic in the Loop architecture:

System 1:
Fast VLA execution

System 3:
Lightweight visual Critic
- progress monitoring
- failure detection
- stagnation detection

System 2:
Large VLM reasoning
- invoked only when needed
- failure diagnosis
- replanning / recovery

For this V2 implementation, use a lightweight vision-language model such as:

Florence-2-base

or another approximately 0.2B-scale visual model with similar inference characteristics.

Important:

Do not assume zero-shot Florence automatically reaches 20 Hz.

Measure actual latency on the current machine.

Make critic frequency configurable.

The first version is only a software and inference proof-of-concept.

4. Scope

4.1 In Scope

Implement a new visual trigger that:

receives the latest robot camera image;

receives the current subtask text;

performs lightweight visual inference;

returns a structured execution-state result;

runs asynchronously;

allows only one active inference at a time;

keeps only the newest pending frame;

discards stale results;

applies temporal debounce for STALLED;

immediately triggers on confident visible FAILURE;

integrates with the existing agentic pipeline;

pauses the current policy when triggered;

invokes the existing Qwen Stage-2 afterward;

logs detailed timing and state information.

4.2 Out of Scope

Do not implement the following in this version:

Florence fine-tuning;

SAFE / VLA-latent failure detection;

recovery-policy learning;

changes to the existing Stage-2 diagnosis model;

action-space correctness fixes as part of the task trigger;

MPC / ROS / NaN errors as task-failure labels;

natural-language keyword-based safety guards;

long video-window inference.

5. Core Design Principle

The system must explicitly separate:

WHEN to interrupt

from:

WHY the failure happened
WHAT recovery should be executed

The Visual Critic answers only:

WHEN to interrupt

The existing Qwen Stage-2 answers:

WHY
WHAT NEXT

6. Required Critic States

Define the execution state as:

PROGRESSING
STALLED
FAILURE
SUCCESS
UNKNOWN

Recommended semantics:

PROGRESSING

There is visible evidence that the current subtask is advancing toward completion.

Examples:

end-effector is approaching the correct object;

object is moving with the gripper as intended;

placement motion is visibly progressing;

interaction direction appears consistent with the current subtask.

STALLED

There is no meaningful visible progress, but no explicit catastrophic failure is visible.

Examples:

robot remains stationary for repeated observations;

gripper keeps attempting without changing the scene;

object and robot interaction remain effectively unchanged;

repeated oscillation without task advancement.

FAILURE

A visible task-execution failure has occurred.

Examples:

missed grasp;

object dropped after grasp;

robot visibly interacts with the wrong object;

target is pushed in the wrong direction;

object is no longer recoverably held;

clearly incorrect physical interaction.

SUCCESS

The currently active subtask, not necessarily the whole episode, is visibly complete.

UNKNOWN

The image is insufficient or ambiguous.

Examples:

occlusion;

target not visible;

camera viewpoint insufficient;

model confidence too low.

7. Structured Output

Do not use PAUSE / CONTINUE as the raw model output.

Create a structured result type.

Recommended implementation:

from dataclasses import dataclass
from typing import Optional, Literal


CriticState = Literal[
    "PROGRESSING",
    "STALLED",
    "FAILURE",
    "SUCCESS",
    "UNKNOWN",
]

CriticConfidence = Literal[
    "LOW",
    "MEDIUM",
    "HIGH",
]


@dataclass
class CriticResult:
    state: CriticState
    progress_score: Optional[float]
    confidence: CriticConfidence
    reason: str

    source_step: int
    source_timestamp: float

    inference_ms: float
    result_age_ms: Optional[float] = None
    stale_discarded: bool = False

progress_score should be interpreted as:

0.0 = no visible task progress
1.0 = current subtask visibly complete

For zero-shot inference, progress_score may be None if the selected model cannot produce it reliably.

Do not fabricate a numeric progress score from an unreliable free-form answer.

8. Model Input

The first V2 version should use:

current latest image
+
current subtask text

Do not use the existing 3-second trigger video window.

Preferred input:

frame: np.ndarray | PIL.Image
subtask: str

Example:

Current subtask:
"Grasp the apple with the right gripper."

First implementation should prefer one informative view, such as:

head
or
front

Do not introduce multi-view inference unless a single view is proven insufficient.

Multi-view can be added later behind a configuration flag.

9. Prompt Design

The trigger prompt must be short.

Do not request chain-of-thought.

Do not request recovery suggestions.

Recommended prompt:

You are a lightweight robot execution critic.

Current subtask:
{subtask}

Inspect the current robot image and classify only the current execution state.

Allowed states:

PROGRESSING:
Visible progress toward completing the current subtask.

STALLED:
No meaningful visible progress, but no explicit failure is clearly visible.

FAILURE:
A visible execution failure has occurred, such as a missed grasp, dropped object,
wrong-object interaction, or clearly incorrect physical interaction.

SUCCESS:
The current subtask is visibly completed.

UNKNOWN:
The state cannot be determined reliably from the image.

Return JSON only:

{
  "state": "PROGRESSING|STALLED|FAILURE|SUCCESS|UNKNOWN",
  "progress_score": 0.0,
  "confidence": "LOW|MEDIUM|HIGH",
  "reason": "short phrase"
}

The prompt may be adapted to the exact model API, but the semantic meaning must remain unchanged.

10. Parsing Requirements

The trigger decision must never depend on English phrase matching.

Forbidden pattern:

if "stationary" in text:
    pause = True

Forbidden pattern:

if "no movement" in text:
    ...

Forbidden pattern:

if "not making visible progress" in text:
    ...

Required pattern:

result.state
result.confidence
result.progress_score

If JSON parsing fails:

state = "UNKNOWN"
confidence = "LOW"

Do not guess the state from prose fallback.

11. Remove the Existing PREGRASP Override From the V2 Path

The new V2 decision chain must not allow a rule such as:

PREGRASP_NO_COMPLETED_ATTEMPT

to force:

STALLED / FAILURE

back to:

CONTINUE

In particular:

PREGRASP

is still allowed to produce:

STALLED
FAILURE

A grasp attempt does not need to be completed before the monitor is allowed to interrupt.

The new Visual Critic is task-state driven, not phrase driven and not attempt-count driven.

12. Trigger Decision Logic

Recommended final-state logic:

if state == "FAILURE" and confidence in {"MEDIUM", "HIGH"}:
    trigger = True
    trigger_type = "FAILURE"

elif state == "STALLED" and stall_confirmed:
    trigger = True
    trigger_type = "STALLED"

elif state == "SUCCESS":
    trigger = False
    notify_subtask_success = True

else:
    trigger = False

13. Temporal Debounce

Do not pause on a single STALLED observation.

Use temporal confirmation.

Recommended configurable parameter:

stall_confirm_count = 3

Example:

step 40:
STALLED / HIGH
stall_count = 1

step 42:
STALLED / MEDIUM
stall_count = 2

step 44:
STALLED / HIGH
stall_count = 3
→ trigger

Reset the counter if the state becomes:

PROGRESSING
SUCCESS
FAILURE

Possible implementation:

if result.state == "STALLED" and result.confidence in {"MEDIUM", "HIGH"}:
    self.stall_counter += 1
else:
    self.stall_counter = 0

For explicit visible FAILURE, no multi-frame confirmation is required by default.

14. Optional Progress-Based Stagnation

If progress_score proves reasonably stable, support an optional progress-based stagnation check.

Track:

max_progress

Update:

if progress_score > max_progress + progress_epsilon:
    max_progress = progress_score
    no_progress_counter = 0
else:
    no_progress_counter += 1

Possible configuration:

progress_epsilon = 0.03
no_progress_confirm_count = 3

Do not enable this by default until progress_score quality has been measured.

15. Asynchronous Execution

The Critic must never block the robot control loop.

Required behavior:

robot execution
████████████████████████████

critic inference
    ████     ████     ████

Qwen Stage-2
                     ███████████
                     only after trigger

The Visual Critic runs asynchronously from VLA execution.

16. Single In-Flight Inference

Only one Critic inference may execute at a time.

Do not create an unbounded inference queue.

Required state:

self._running = False
self._latest_pending = None

Behavior:

frame A submitted
→ run A

while A is running:
frame B → pending = B
frame C → pending = C
frame D → pending = D

A finishes
→ process D

B and C are discarded

This implements:

single in-flight
+
latest-frame overwrite

17. Suggested Concurrency Pseudocode

def submit(frame, subtask, source_step, timestamp):
    item = CriticInput(
        frame=frame,
        subtask=subtask,
        source_step=source_step,
        timestamp=timestamp,
    )

    with self._lock:
        if self._running:
            self._latest_pending = item
            return

        self._running = True

    self._launch(item)


def _on_finished(result):
    self._publish_result(result)

    with self._lock:
        pending = self._latest_pending
        self._latest_pending = None

        if pending is None:
            self._running = False
            return

    self._launch(pending)

Use the repository's existing threading / async conventions if available.

Do not introduce a large new concurrency framework unnecessarily.

18. Stale Result Rejection

Each inference result must contain:

source_step
source_timestamp

When the result is consumed:

result_age = current_time - result.source_timestamp

If:

result_age > max_result_age

the result must be discarded.

Recommended configurable initial value:

max_result_age = 1.0 ~ 2.0 seconds

Do not hard-code the final value.

The exact threshold should be tuned after measuring zero-shot critic latency.

Example:

if result_age > cfg.max_result_age:
    result.stale_discarded = True
    return

A stale FAILURE must not pause the current robot state.

19. Critic Frequency

Do not assume full control-frequency inference is possible.

Add configuration:

critic_frequency_hz

Recommended initial value:

5 Hz

Then benchmark:

2 Hz
5 Hz
10 Hz
20 Hz

Use the fastest rate that does not cause backlog.

The architecture should support later increasing the rate after optimization or fine-tuning.

20. Separation Between Task Failure and System Errors

The new V2 Visual Critic should detect task execution state, not software errors.

The following are not task-failure labels:

action out of range
MPC exception
ROS exception
NaN
device error
tuple has no attribute name
camera read error
network exception

These belong to a separate:

System Health Monitor

Desired separation:

Task-state monitor
    ↓
Visual Critic
    ↓
PROGRESSING / STALLED / FAILURE / SUCCESS / UNKNOWN

versus:

System health
    ↓
ROS / MPC / action interface / NaN / device errors
    ↓
log / abort / debug

Do not count system implementation faults as Visual Critic detection success.

21. Suggested File Structure

Prefer new files over large rewrites of existing modules.

Recommended structure:

kuavo_deploy/
└── src/
    └── eval/
        └── failure_trigger/
            ├── __init__.py
            ├── base_trigger.py
            └── visual_critic_trigger.py

Optional config:

kuavo_deploy/
└── config/
    └── visual_critic.yaml

Adapt paths to the repository's real structure if needed.

Do not create duplicate package layers if the repository already has an equivalent module location.

22. Base Interface

Recommended abstraction:

from abc import ABC, abstractmethod
from typing import Optional


class FailureTrigger(ABC):

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def submit(
        self,
        frame,
        subtask: str,
        source_step: int,
        timestamp: float,
    ) -> None:
        ...

    @abstractmethod
    def get_latest_result(self) -> Optional[CriticResult]:
        ...

    @abstractmethod
    def shutdown(self) -> None:
        ...

This interface should make future replacement easy:

VisualCriticTrigger
LatentSafeTrigger

without rewriting the agentic loop.

23. Visual Critic Configuration

Recommended configuration fields:

visual_critic:
  enabled: true

  model_name_or_path: "microsoft/Florence-2-base"

  device: "cuda:0"
  dtype: "bfloat16"

  frequency_hz: 5.0

  max_result_age_s: 1.5

  stall_confirm_count: 3

  failure_min_confidence: "MEDIUM"
  stall_min_confidence: "MEDIUM"

  progress_enabled: false
  progress_epsilon: 0.03
  no_progress_confirm_count: 3

  view: "head"

  save_trigger_context: true
  pre_trigger_seconds: 1.5
  post_trigger_seconds: 0.5

Do not assume every model supports BF16 on every GPU.

Use an existing dtype/device helper if the repository has one.

24. Model Wrapper

Keep model-specific code isolated.

Example:

class FlorenceVisualCriticModel:

    def __init__(self, model_name_or_path, device, dtype):
        ...

    def infer(
        self,
        frame,
        subtask: str,
    ) -> dict:
        ...

The rest of the trigger pipeline must not depend on Florence-specific processor internals.

This is required so a later model can replace Florence without rewriting the scheduler.

25. Zero-Shot First Version

The first implementation should support:

Florence-2-base
zero-shot
inference only

The goal is not yet to prove best detection accuracy.

The goal is to verify:

image
→ lightweight critic
→ structured result
→ asynchronous trigger
→ pause
→ Stage-2

Later, the same wrapper should accept:

/path/to/finetuned_visual_critic_checkpoint

without changing agent-loop code.

26. Qwen Stage-2 Integration

Do not remove the existing Stage-2 failure reasoning path.

After trigger:

pause_current_policy()

Then prepare a short context for Stage-2.

Recommended context:

current subtask
trigger state
trigger confidence
critic reason
source step
current step
recent image/video context
recent robot execution context if already available

Example Stage-2 metadata:

{
  "trigger": {
    "state": "FAILURE",
    "confidence": "HIGH",
    "reason": "object appears to have slipped from gripper",
    "source_step": 51
  }
}

Stage-2 should continue to decide:

failure type
recoverable or not
retry
recovery action
replan
reset
abort

Do not move these responsibilities into the V2 Critic.

27. Trigger Context Buffer

Maintain a lightweight rolling frame buffer so Stage-2 can receive a short pre-trigger history.

Recommended:

1.0–1.5 seconds before trigger

After pause, optionally collect:

0.3–0.5 seconds after trigger

Do not use this longer context for the fast Critic itself.

It exists only for Stage-2 diagnosis.

28. Logging

Add explicit V2 logs.

Required fields:

source_step
observed_step

critic_state
progress_score
confidence
reason

inference_ms
result_age_ms

stale_discarded

stall_counter
stall_confirm_count

final_trigger
trigger_type

Recommended format:

[V2_CRITIC]
source_step=42
observed_step=44
state=STALLED
confidence=HIGH
progress=0.31
inference=78ms
age=95ms
stall_count=2/3
stale=False
trigger=False

Triggered example:

[V2_CRITIC]
source_step=45
observed_step=46
state=STALLED
confidence=HIGH
progress=0.30
inference=76ms
age=88ms
stall_count=3/3
stale=False
trigger=True
trigger_type=STALLED
action=PAUSE_AND_STAGE2

Failure example:

[V2_CRITIC]
source_step=53
observed_step=54
state=FAILURE
confidence=HIGH
progress=None
inference=81ms
age=90ms
stall_count=0/3
stale=False
trigger=True
trigger_type=FAILURE
action=PAUSE_AND_STAGE2

29. Metrics to Record

For each rollout, save at least:

episode id
task / subtask
success
number of critic calls
number of stale results
mean inference latency
median inference latency
p95 inference latency
number of STALLED predictions
number of FAILURE predictions
number of trigger events
trigger step
Stage-2 invocation count
pause count

If failure onset annotation exists, additionally record:

detection delay
false trigger count
missed failure count

Possible definition:

detection_delay =
trigger_time - annotated_failure_onset_time

Negative values mean early detection.

30. Evaluation Goals for the First V2 Prototype

The first V2 prototype should be considered successful if:

1. VLA control is not blocked by Critic inference.
2. Only one Critic inference can be active at once.
3. Old frames do not build up in a queue.
4. Stale results are never executed.
5. PREGRASP_NO_COMPLETED_ATTEMPT cannot override a V2 trigger.
6. STALLED can trigger after temporal confirmation.
7. Explicit FAILURE can trigger immediately.
8. Qwen Stage-2 is invoked after a trigger.
9. System errors are not counted as task failures.
10. Full trigger latency is logged.

Detection accuracy can be improved later through task-specific fine-tuning.

31. Important Implementation Constraints

Do not modify behavior unrelated to the trigger

Avoid broad refactors.

Keep:

existing VLA execution
existing simulator interface
existing Stage-2
existing recovery implementation
existing episode evaluation

unless a minimal adapter is required.

Prefer new files

Do not overwrite the current Qwen trigger implementation.

Keep it available as a baseline.

Desired future experiment:

--trigger qwen
--trigger visual_critic

No local rollout testing

Only modify and statically inspect the code unless explicitly instructed otherwise.

Do not launch simulator rollouts, GPU model downloads, robot execution, or expensive inference tests automatically.

32. CLI / Configuration Integration

Prefer a switch such as:

--failure-trigger qwen

and:

--failure-trigger visual_critic

or equivalent config:

failure_trigger:
  type: visual_critic

This is important for later ablation.

Do not delete the existing Qwen trigger path.

33. Recommended State Machine

                    ┌───────────────┐
                    │  PROGRESSING  │
                    └───────┬───────┘
                            │
                            │ keep executing
                            ↓

Camera ──→ Visual Critic ──→ STALLED
                            │
                            │ repeated N times
                            ↓
                          PAUSE
                            │
                            ↓
                         Stage-2

Camera ──→ Visual Critic ──→ FAILURE
                            │
                            │ immediate
                            ↓
                          PAUSE
                            │
                            ↓
                         Stage-2

Camera ──→ Visual Critic ──→ SUCCESS
                            │
                            ↓
                    notify subtask done

Camera ──→ Visual Critic ──→ UNKNOWN
                            │
                            ↓
                     continue observing

34. Future Fine-Tuning Compatibility

The architecture must anticipate a later V2 training phase.

Future training labels may include:

progress bin
anomaly token
subtask completion

Possible future output vocabulary:

<P00>
<P10>
<P20>
...
<P100>
<ACI>

or equivalent structured classes.

Do not hard-code the current zero-shot JSON prompt so deeply that replacing it with a fine-tuned checkpoint becomes difficult.

The following should remain stable across zero-shot and fine-tuned V2:

CriticResult
FailureTrigger interface
async scheduler
latest-frame overwrite
stale-result rejection
temporal debounce
agent-loop integration
logging
Stage-2 integration

Only the model wrapper should need to change.

35. Future V3 Compatibility

The V2 implementation should also make a later V3 trigger easy to add.

Future:

class LatentSafeTrigger(FailureTrigger):
    ...

V3 input will likely use:

VLA internal latent history

rather than images.

The agentic loop should not care whether the trigger source is:

Qwen
Florence
SAFE-style latent detector

It should consume only:

CriticResult

36. Final Intended Pipeline

                        ┌─────────────────────────┐
                        │        VLA Policy       │
                        │   continuous execution  │
                        └────────────┬────────────┘
                                     │
                                     │ actions
                                     ↓
                                  Robot
                                     │
                                     │ latest image
                                     ↓
                        ┌─────────────────────────┐
                        │   V2 Visual Critic      │
                        │                         │
                        │ PROGRESSING             │
                        │ STALLED                 │
                        │ FAILURE                 │
                        │ SUCCESS                 │
                        │ UNKNOWN                 │
                        └────────────┬────────────┘
                                     │
                        STALLED / FAILURE
                                     │
                                     ↓
                                  PAUSE
                                     │
                                     ↓
                        ┌─────────────────────────┐
                        │ Existing Qwen Stage-2   │
                        │                         │
                        │ diagnose failure        │
                        │ choose recovery         │
                        └────────────┬────────────┘
                                     │
                                     ↓
                         retry / recover / replan

37. One-Sentence Definition

V2 uses a lightweight visual Critic to continuously estimate task progress and detect visible execution anomalies, triggering the slower Qwen recovery agent only when intervention is actually needed.

38. Codex Deliverables

Please implement the following:

Inspect the current trigger and Stage-2 integration code.

Identify the minimal integration point for a new trigger backend.

Add a reusable FailureTrigger abstraction if one does not already exist.

Add VisualCriticTrigger.

Add a lightweight model wrapper, initially supporting Florence-2-base zero-shot inference.

Add structured CriticResult.

Add asynchronous single-in-flight inference.

Add latest-frame overwrite.

Add stale-result rejection.

Add temporal debounce for STALLED.

Add immediate confident FAILURE trigger.

Remove PREGRASP_NO_COMPLETED_ATTEMPT from the V2 decision path.

Keep system/control faults separate from task-failure detection.

Connect V2 trigger events to the existing pause + Qwen Stage-2 path.

Add configuration flags for all thresholds and model settings.

Add detailed V2 logs.

Keep the existing Qwen trigger implementation available as a baseline.

Do not launch local simulation or GPU-heavy tests; only modify code and perform lightweight static validation unless explicitly requested.

Before changing code, first inspect the repository and summarize:

- current Stage-1 trigger entry point
- current pause mechanism
- current Stage-2 invocation point
- files that need to be added
- files that minimally need to be modified

Then implement the V2 architecture with the smallest practical changes to the existing pipeline.