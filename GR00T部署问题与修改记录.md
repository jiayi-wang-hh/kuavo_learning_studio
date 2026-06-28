# GR00T 部署问题与修改记录

## 1. Execution Step 设置

### 可能存在的问题

当前 GR00T 模型预测的 action chunk 长度为 16：

```python
delta_indices = list(range(16))
```

Learning Studio 中 `execution_horizon` 默认也设置为 16。这意味着模型观察一次环境并预测 16 步动作后，会连续执行完整的 16 步动作，期间不会根据最新图像和机器人状态重新推理。

当控制频率为 10 Hz 时，完整执行 16 步相当于约 1.6 秒的开环控制。在抓取、接近目标等需要频繁视觉修正的阶段，可能出现横向偏移、越过目标或无法及时纠正动作的问题。

需要区分：

- `action_horizon`：模型一次预测的 action chunk 长度，本项目为 16。
- `execution_horizon`：从预测 chunk 中实际执行多少步后重新观察和规划。

官方 Isaac GR00T N1.7 仿真 rollout 默认使用 `n_action_steps = 8`。官方代码也给出了预测 16 步、执行 8 步的 receding-horizon 示例。因此，保留 16 步预测 chunk、将实际执行步数改为 8，更接近官方部署方式。

### 建议修改

修改：

```text
configs/server/launcher.yaml
```

将：

```yaml
groot:
  args:
    execution_horizon: 16
```

改为：

```yaml
groot:
  args:
    execution_horizon: 8
```

也可以在启动时临时覆盖：

```bash
python kuavo_server/launch.py groot \
  --checkpoint /path/to/checkpoint \
  --execution_horizon 8 \
  --strict
```

### 对照实验

建议分别测试：

| execution_horizon | 10 Hz 下的重规划周期 | 用途 |
|---:|---:|---|
| 4 | 0.4 秒 | 更频繁地根据视觉反馈修正 |
| 8 | 0.8 秒 | 官方 N1.7 仿真默认，建议优先测试 |
| 16 | 1.6 秒 | 完整 action chunk 开环执行 |

测试时保持 checkpoint、任务提示词、相机输入、初始状态和随机种子一致，只修改 `execution_horizon`，对比目标接近轨迹和任务成功率。

### 判断标准

- 如果 8 步或 4 步明显改善目标接近和抓取轨迹，说明完整执行 16 步导致反馈修正不及时。
- 如果 4、8、16 步的行为基本一致，说明 `execution_horizon` 不是主要原因，应继续检查模型离线预测、数据与部署状态顺序、相机输入和归一化统计。

### 修改状态

- [ ] 修改 `execution_horizon`
- [ ] 完成 4/8/16 步对照实验
- [ ] 记录轨迹、成功率和结论

---

## 2. Simulator Head Init 设置

### 是否影响部署

有影响。

在 simulator 环境每次执行 `env.reset()` 时，部署代码都会调用：

```python
self.robot.control_head(self.head_init[0], self.head_init[1])
```

因此，`head_init` 会直接改变机器人头部姿态和头部相机视角。GR00T 使用 `head`、`wrist_left`、`wrist_right` 三路图像进行推理，如果头部相机视角与训练数据不一致，就会产生视觉域偏移，可能影响目标定位、接近方向和抓取动作。

### 可能存在的问题

原始 `kuavo_data_challenge` simulator 配置和 Learning Studio 部署文档使用：

```yaml
head_init: [0, 0.209]
```

其中 `0.209 rad` 约等于 `12°`。

但 Learning Studio 当前在加载 sim 配置时强制设置：

```python
env_cfg["head_init"] = [0, 12]
```

随后将该值直接传给 `control_head()`。如果 SDK 接口使用弧度，那么这里传入的是 `12 rad`，而不是预期的 `0.209 rad`，头部可能被转到错误位置或被关节限位裁剪，导致部署图像与训练图像明显不一致。

此外，该值是在配置加载代码中直接赋值的，因此即使在 `configs/deploy/deploy.yaml` 中手动填写 `head_init`，也会被覆盖。

### 建议修改

修改：

```text
kuavo_deploy/config.py
```

将：

```python
env_cfg["head_init"] = [0, 12]
```

改为：

```python
env_cfg.setdefault("head_init", [0, 0.209])
```

这样默认值与旧 simulator 部署保持一致，同时允许用户在 `deploy.yaml` 中显式覆盖。

如果希望暂时保持强制默认值，也至少应改为：

```python
env_cfg["head_init"] = [0, 0.209]
```

### 验证方法

1. 分别使用 `[0, 12]` 和 `[0, 0.209]` 重置 simulator。
2. 保存 `observation.images.head_cam_h` 的第一帧。
3. 与训练数据中相同初始阶段的 `head` 图像对比相机俯仰角和目标位置。
4. 在 SDK 文档或运行日志中确认 `control_head()` 的单位。
5. 保持 checkpoint、seed、prompt 和 execution horizon 不变，对比任务轨迹及成功率。

### 判断标准

- 如果 `[0, 0.209]` 的头部图像明显更接近训练数据，并改善目标接近轨迹，则错误的 head init 是部署问题之一。
- 如果两种设置得到完全相同的实际头部姿态，需要检查 SDK 是否自动进行角度转换或限位裁剪。
- 即使 head init 修复后模型仍存在动作噪声，也不能排除该问题；head init 影响视觉输入，而模型离线预测噪声属于另一条问题链。

### 修改状态

- [ ] 确认 `control_head()` 参数单位
- [ ] 修正 sim 默认 `head_init`
- [ ] 保存并对比 simulator 与训练数据头部图像
- [ ] 对比修改前后的任务成功率

---

## 3. 

### 可能存在的问题



### 建议修改



### 验证结果



---

## 4. 

### 可能存在的问题



### 建议修改



### 验证结果
