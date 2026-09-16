YOLO + ByteTrack Fast Trigger：第一版检测测试

目标

先不要做最终 PAUSE / CONTINUE。这一步只验证一件事：YOLO + tracker 能不能稳定看到并持续跟踪 toy / basket / gripper。

如果这一层都不稳定，就不值得继续写 trigger 规则；如果 tracking 稳定，再加入 grasp-miss / drop / placement-failure 逻辑。

1. 安装

建议在你现有实验环境里单独安装，不需要动 Qwen 环境：

pip install ultralytics opencv-python

测试：

python -c "from ultralytics import YOLO; print('ultralytics ok')"

2. 先跑一个 head video

脚本：yolo_fast_trigger_test.py

python yolo_fast_trigger_test.py \
  --video /media/data/jiayi/dataset/joy_videos/rollout01_head.mp4 \
  --model yolo11n.pt \
  --device 0

输出：

outputs/yolo_trigger_test/
├── rollout01_head_tracked.mp4
└── rollout01_head_tracks.csv

其中：

*_tracked.mp4：画好 bbox 和 track id 的视频

*_tracks.csv：每一帧的 bbox、中心点、类别、track id

CSV 里最重要的是：

frame
track_id
class_name
cx, cy
x1, y1, x2, y2
confidence

3. 先看什么

打开：

mpv outputs/yolo_trigger_test/rollout01_head_tracked.mp4

如果服务器没 GUI，就把视频拉到 Mac 看。

你先人工检查：

toy 是否能被检测；

basket 是否能被检测；

同一个 toy 在连续帧里 track id 是否稳定；

toy 被抓起来后 bbox 是否真的跟着 gripper 移动；

toy 掉落时 track 是否还能持续；

速度是否足够实时。

4. 一个重要现实问题

标准 yolo11n.pt 是通用数据集预训练模型，大概率没有 robot gripper，也未必认识你实验里的 toy / basket。

所以这一步主要是 pipeline sanity check。

如果效果是：

basket 偶尔检测到
custom toy 基本检测不到
gripper 检测不到

这不是 YOLO 方案失败，而是类别不匹配。

下一步有两条路：

方案 A：自定义 YOLO（更推荐最终使用）

标注少量图像：

toy
basket
gripper

或者为了双臂直接标：

left_toy
right_toy
left_basket
right_basket
left_gripper
right_gripper

第一版只需要验证可行性，不必一开始标很多数据。

方案 B：YOLO-World（更适合快速 zero-shot 试验）

可以用文本类别尝试：

toy
basket
robot gripper

如果 zero-shot 已经足够稳定，就可以减少标注工作；如果不稳定，再训练 custom YOLO。

5. Tracking 稳定后再做 Fast Trigger

第一版只需要三个 trigger。

Grasp miss

逻辑：

gripper_near_toy_before = True

gripper_moved_up = gripper_dy < -GRIPPER_UP_THRESHOLD

toy_moved_up = toy_dy < -TOY_UP_THRESHOLD

if gripper_near_toy_before and gripper_moved_up and not toy_moved_up:
    pause = True

更稳一点可以比较 toy-gripper 距离：

if toy_follows_gripper_for_N_frames:
    grasp_success = True
else:
    possible_grasp_miss = True

Unexpected drop

if was_following_gripper and suddenly_far_from_gripper:
    possible_drop = True

如果同时 toy 向下移动，会更可靠：

if was_following_gripper \
   and toy_gripper_distance > DROP_DISTANCE \
   and toy_vertical_velocity > DOWNWARD_THRESHOLD:
    pause = True

Placement failure

如果 basket bbox 已经有了，可以先用最简单规则：

inside = (
    basket_x1 < toy_cx < basket_x2
    and basket_y1 < toy_cy < basket_y2
)

if placement_finished and not inside:
    pause = True

实际论文版本可以以后换成 segmentation / mask overlap，而不是 bbox center。

6. 最终建议结构

camera @ 10-30 Hz
      ↓
YOLO
      ↓
ByteTrack
      ↓
object / basket trajectories
      +
robot joint / gripper / controller state
      ↓
Hybrid Fast Trigger
      ↓
NORMAL ───────────────→ VLA continues
      |
      └── EVENT → PAUSE
                    ↓
                Qwen Stage-2
                    ↓
          failure diagnosis
                    ↓
       retry / recover / reset

YOLO 不应该负责回答“为什么失败”。

YOLO/Tracker 只负责给 fast trigger 提供稳定的低层运动事实；Qwen 留在 Stage-2 做语义理解和 recovery reasoning。

7. 你现在建议的实验顺序

rollout01_head.mp4 跑 YOLO11n + ByteTrack。

看 annotated video。

如果 toy / gripper 检不出来，不调 trigger，直接测试 YOLO-World 或标 50–200 张图训练 custom YOLO。

如果 detection + tracking 已经稳定，再实现 grasp_miss。

用你已有的 toy_annotations.json 对比 trigger time。

最后再把它接入真机 agentic pipeline。

第一阶段最重要的指标不是 success rate，而是：

detection recall
track continuity
ID switches
per-frame latency / FPS
toy-gripper relative trajectory quality