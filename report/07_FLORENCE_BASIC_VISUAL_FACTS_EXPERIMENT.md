# Florence-2 基础机器人视觉事实小实验

## 实验问题

这个实验只回答 Florence-2-base 能否从单帧识别以下可见事实：

1. gripper 是否可见；
2. 指定目标物体是否可见；
3. 目标是否明显被 gripper 抓住；
4. 目标位于画面左、中、右哪一区域；
5. 目标在桌面、空中或容器内。

不测试 `STALLED`、运动进展、抓取后是否持续跟随等必须依赖时间的信息。

## 为什么需要两条 probe

每张图执行两次推理：

- 原生 `<MORE_DETAILED_CAPTION>`：检查 Florence 是否在 caption 中看见了 gripper、目标、接触关系和位置。
- 自由 JSON prompt：检查它是否能直接遵守当前 V2 所需的结构化分类接口。

两者必须分开解释。如果原生 caption 描述正确但 JSON 无效，说明视觉表征可能可用，但 Florence-2-base 不适合作为零样本自由 JSON trigger。如果两者都无法识别基础事实，则不应继续用该 checkpoint 做 Stage-1。

## 最小数据设计

建议从已有仿真录像人工截取 24 张互不重复的清晰单帧，每个条件 4 张：

| 条件 | 数量 | 关键事实 |
|---|---:|---|
| gripper 接近目标、目标仍在桌面 | 4 | NOT_HELD / ON_SURFACE |
| gripper 闭合但抓空 | 4 | NOT_HELD / ON_SURFACE |
| 目标明确夹在 gripper 中并离桌 | 4 | HELD / IN_AIR |
| 目标已在容器中 | 4 | NOT_HELD 或 AMBIGUOUS / IN_CONTAINER |
| 目标被遮挡或接触关系不清 | 4 | AMBIGUOUS / UNKNOWN |
| 干扰物存在、指定目标在另一位置 | 4 | 测试 target grounding 和 LEFT/CENTER/RIGHT |

至少覆盖左右两只 gripper、画面左中右、不同目标位置。不要连续抽取几乎相同的视频帧，否则结果会虚高。

`HELD` 标签应非常保守：单帧中只有夹持接触和支撑关系都清楚时才标 HELD；仅仅二维重叠应标 `AMBIGUOUS`。

## 准备文件

复制 `vlm_test/florence_basic_visual_facts_manifest.example.json`，将图片放在 manifest 相对路径下，然后填写人工 ground truth。推荐目录：

```text
vlm_test/florence_facts/
├── manifest.json
└── frames/
    ├── held_center_01.jpg
    └── ...
```

## 运行命令

使用 Hugging Face 模型名：

```bash
python vlm_test/florence_basic_visual_facts_probe.py \
  --manifest vlm_test/florence_facts/manifest.json \
  --model microsoft/Florence-2-base \
  --output vlm_test/florence_facts/results.jsonl \
  --device cuda:0 \
  --dtype bfloat16
```

若权重已经下载到本地，推荐直接指定本地目录，避免运行时联网：

```bash
python vlm_test/florence_basic_visual_facts_probe.py \
  --manifest vlm_test/florence_facts/manifest.json \
  --model /path/to/Florence-2-base \
  --output vlm_test/florence_facts/results.jsonl \
  --device cuda:0 \
  --dtype bfloat16
```

不支持 BF16 的设备改用 `--dtype float16`；CPU 小规模检查可使用 `--device cpu --dtype float32`，但会较慢。

## 如何判定结果

先看以下三个层级，不要只看一个总准确率：

1. `free_json_schema_valid`：24 张中有多少张产生完整合法 JSON。建议至少 95%。
2. 基础事实准确率：对每个字段分别人工/脚本比较 ground truth，重点报告 HELD precision、HELD recall、NOT_HELD recall。
3. `native_detailed_caption`：人工检查 caption 是否明确提到目标、gripper、支撑/夹持关系；不要因 JSON 失败而忽略这部分。

建议的最低继续条件：

- gripper visible accuracy ≥ 90%；
- target visible accuracy ≥ 90%；
- object position accuracy ≥ 85%；
- HELD precision ≥ 90%，避免把二维重叠误判成抓住；
- NOT_HELD recall ≥ 85%；
- JSON schema validity ≥ 95%。

如果 caption 能力达标但 JSON validity 很低，结论应是“Florence 视觉特征可能有用，但自由 JSON 调用方式不合适”，而不是“Florence 完全看不懂机器人场景”。

## 输出

脚本逐图写入 JSONL，包含 ground truth、原生 caption、自由 JSON 原文、解析结果、schema validity 和两次推理延迟。它不会修改 V2 trigger，也不会自动启动 simulator。
