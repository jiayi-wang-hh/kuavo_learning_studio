## Failure Trigger：Robot-State + Multi-View Object Tracking

### 核心思路

使用 **Head Camera + Left/Right Wrist Camera + Robot State** 判断 target 的实际运动是否符合当前机器人动作的预期。

> **Robot State：告诉我现在应该发生什么**
> 
> 
> **Vision：告诉我实际上发生了什么**
> 
> **Expected ≠ Observed → Candidate Failure**
> 

### Pipeline

```
VLA Action / Robot State
(gripper, EE pose/motion)
          │
          ↓
   Task Phase Estimation
          │
          ↓
APPROACH → GRASP → LIFT → TRANSPORT → PLACE → RELEASE
          │
          ↓
 Phase-aware Camera Selection
          │
          ├─ APPROACH        → HEAD
          ├─ GRASP / LIFT    → HEAD + WRIST
          ├─ TRANSPORT       → WRIST dominant
          ├─ PLACE / RELEASE → HEAD + WRIST
          └─ POST-RELEASE    → HEAD
          │
          ↓
 Object Detection / Segmentation / Tracking
          │
          ↓
 Multi-view Target State
(VISIBLE / HELD / DROPPED / RELEASED / LOST / UNCERTAIN)
          │
          ↓
 Expected vs. Observed Motion
          │
          ↓
 Temporal Consistency Check
          │
          ↓
   Candidate Failure Trigger
          │
          ↓
      VLM Verifier
          │
     ┌────┴────┐
   NORMAL    FAILURE
               ↓
          Interrupt VLA
               ↓
            Recovery
```

### Robot State 作为 Phase/Event Trigger

- `Gripper OPEN → CLOSED` → **GRASP attempt** → 启动 Head + Wrist 验证
- `Gripper CLOSED + EE upward` → **LIFT** → 检查 target 是否随 gripper 运动
- `Gripper CLOSED + EE moving` → **TRANSPORT** → Wrist 检查 target 是否保持 held
- `Gripper CLOSED → OPEN` → **RELEASE** → Head 接管，检查 target 最终位置
- EE approaching placement region → **PLACE** → Head + Wrist 联合判断

**注意：`gripper closed ≠ grasp success`，robot state 只表示“机器人尝试做什么”，成功与否仍由视觉验证。**

### Multi-view 分工

- *Head Camera：**全局 target location、gripper-target relation、placement region、release 后 target 最终位置。
- *Wrist Camera：**主要负责 grasp / lift / transport。成功抓取后 target 与 wrist/gripper 应近似相对静止，因此非常适合判断 **HELD / GRASP_MISS / DROP**。

Target 在 Head 中消失**不能直接认为 failure**：

```
Head LOST
   ↓
可能是正常 gripper occlusion
   ↓
Check Wrist
   ├─ target attached → HELD / NORMAL
   └─ target absent   → UNCERTAIN / Candidate Failure
```

### 典型 Failure

```
GRASP_MISS:
gripper closes
+ EE lifts
+ target does NOT follow gripper
→ Trigger

DROP:
previous state = HELD
+ gripper remains closed
+ target suddenly moves relative to wrist/gripper
→ Trigger

PLACE_FAILURE:
gripper opens
+ target released
+ Head detects target outside target region
→ Trigger
```

### Perception Backend

先不绑定具体模型，作为可替换模块：

```
Option A: YOLO-World + CSRT        ← lightweight baseline
Option B: YOLO-World + SAM         ← detection + segmentation/tracking
Option C: Grounding DINO + SAM     ← grounding + segmentation
Option D: SAM 3                    ← unified segmentation/tracking candidate
```

**最终方法定位：**

> **Robot-State-Conditioned + Phase-Aware + Multi-View Object Tracking Failure Trigger**
> 

相比单纯 `YOLO → motion threshold → trigger`，核心不是“物体动没动”，而是：

当前 robot action 下，target 是否按照预期运动\boxed{\text{当前 robot action 下，target 是否按照预期运动}}

这也是后面最值得作为 trigger 主线继续做的部分。