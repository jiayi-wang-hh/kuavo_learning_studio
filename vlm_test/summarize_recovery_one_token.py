"""Build a compact comparison from completed one-token benchmark runs."""
import json
from pathlib import Path

root = Path('outputs/qwen_one_token_20261108')
runs = [json.loads(p.read_text()) for p in sorted(root.glob('Qwen*.json'))]
lines = ['# Recovery / normal 单 token 实测', '',
         'GPU1 RTX PRO 6000 Blackwell；batch=1；Transformers + SDPA；关闭思考；贪心生成恰好 1 token。',
         '输出 0=normal、1=recovery；recovery 本身是 2 token。未强制限制候选词，其他输出计 invalid。',
         '同一提示词、同一图像；每种输入预热2次，再逐窗口测3轮。延迟为 CUDA 同步后的 prefill + 单 token 生成时间，排除加载、预处理和磁盘读取。',
         '参考标签：7 normal、2 recovery，共9个端点，来自两段视频，非独立盲标测试集。三轮重复不计作27个独立样本。',
         'recovery_17 是恢复终点敏感样本；此处按包裹回到平台内部记 normal，即使仍被手持。', '',
         '| 模型 | 精度 | 输入 | 正确 | Recovery precision | Recovery recall | 无效 | 中位延迟 ms | P95 ms | 参数占用 GiB |',
         '|---|---|---|---|---|---|---|---|---|---|']
for run in runs:
    for group, s in run.get('summary', {}).items():
        precision = s['recovery_precision']
        precision = 'N/A' if precision is None else f'{precision:.1%}'
        lines.append(f"| {run['model']} | {run['precision']} | {group} | {s['correct']}/{s['n']} | {precision} | {s['recovery_recall']:.1%} | {s['invalid']} | {s['median_seconds']*1000:.1f} | {s['p95_seconds']*1000:.1f} | {run['memory_footprint_bytes']/2**30:.2f} |")
lines += ['', 'final=最后单帧；original=原始时间窗口，最多3帧。',
          '速度受推理后端、输入长度和GPU0并行任务影响；MoE的激活参数量小不保证此后端的单 token 延迟最低。',
          'NF4 只量化受支持的 Linear；35B MoE 的三维专家参数在此 bitsandbytes 集成中不量化，不能把它当作完整4bit模型。', '',
          '## 逐端点预测（第一轮）', '', '| 模型 | 精度 | 输入 | 窗口 | 标签 | 预测 | 原始输出 |', '|---|---|---|---|---|---|---|']
for run in runs:
    for r in run['rows']:
        if r['repeat'] == 0:
            lines.append(f"| {run['model']} | {run['precision']} | {r['group']} | {r['id']} | {r['truth']} | {r['prediction']} | {r['raw']!r} |")
root.mkdir(parents=True, exist_ok=True)
(root / 'comparison.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:25]))
