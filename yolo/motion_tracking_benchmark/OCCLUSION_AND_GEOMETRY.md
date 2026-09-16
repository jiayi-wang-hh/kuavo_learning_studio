# Option A：Toy 几何过滤与 gripper 遮挡处理

Option A (`yolo_csrt`) 现在直接复用
`yolo/test_yolo_world_roi_filter.py` 的过滤、side track 和时序状态机。这里记录的是
实际生效的基线规则，而不是另一套相似但不同的实现。

## Detector vocabulary

Option A 固定使用 `toy`、`basket`、`robot gripper`。`table` 和 `robot arm` 不属于
这个已验证基线的 detector vocabulary；它们仍可在 B/C 的开放词表配置中使用。

## 新 toy 检测的几何过滤

每个 `toy` detector box 会依次检查：面积、最大宽度、最大高度、中心点 ROI、底边
位置，以及它相对于 gripper 的覆盖率。默认参数与原脚本完全一致，例如：

- `--toy-max-area-ratio 0.07`
- `--roi-xmin 0.05 --roi-xmax 0.95 --roi-ymin 0.30 --roi-ymax 0.88`
- `--gripper-overlap-reject 0.55`

被拒绝的候选会以 `source=REJECTED` 写入 benchmark 的 `tracks.csv`，具体原因在
`rejection_reason` 中。

## 正常的 gripper 遮挡

gripper 遮挡 toy 时，YOLO 检测框可能消失或被几何过滤。这不会立即删除既有 toy
轨迹：原基线的优先级是：

`DETECTED → TRACKED → PREDICTED → LOST`

1. detector 没有可关联 toy 时，先更新 CSRT/KCF/MIL visual tracker；
2. visual tracker 失败后，以指数衰减的常速度模型短时预测，默认最多
   `--max-missing 5` 帧；
3. 之后才输出 `LOST`。

重要的是，gripper overlap rejection **只应用于 YOLO 的新检测**，不应用于 visual
tracker。因此 tracker 可以在真实抓取重叠期间维持 toy 身份。

## 双目标与 temporal association

基线固定维护 `left_toy` 和 `right_toy` 两条 side track。新 detection 先按左右 ROI
进入候选集合，再根据上一状态的速度预测，以 `--assoc-max-dist`（默认 0.20，归一化
距离）选择最一致的候选。每侧每帧最多接纳一个候选；其余候选标记
`not_selected_by_side_track`。这就是 Option A 的 two-target cap。

## Tracker 安全检查

CSRT 更新还必须通过：出画面、中心跳变、面积过度扩大/缩小的检查；不通过会转入
prediction fallback，而不会把漂移框当作可靠 toy 轨迹。
