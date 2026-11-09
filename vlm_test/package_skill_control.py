"""Controlled offline skill-label experiment; never writes baseline files or robot state.

Prepare and lock labels before inference:
  python vlm_test/package_skill_control.py prepare
Run each model once (9B before 27B), then summarize.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/qwen_skill_control_20261028'
BASE = ROOT / 'outputs/qwen_recovery_nf4'
FRAMES = ROOT / 'outputs/qwen_recovery_bf16/frames'
MODELS = ['Qwen3.5-9B', 'Qwen3.5-27B']

STATE_RULES = """Observe the current state of a Kuavo bimanual package-sorting task. Images are chronological. Classify the LAST image, using earlier images only as context.
Select recovery when a package is displaced at the near/front edge of the yellow sorting platform on the side facing the robot and needs to be brought back onto the platform. This near-edge misplaced state is the START of recovery: it is not necessary to see the package falling or to prove that it reached the floor. A package extending down beyond this near edge can be partly outside the bottom of the camera image. Recovery can already be in progress; a hand approaching, touching or picking up the misplaced package does not by itself make the state normal.
Select normal for ordinary left-side parcel supply, normal pick/flip/transfer on or above the yellow platform, and parcels in the right basket. Once the displaced package is brought back onto/over the platform, with its footprint inside the platform, classify that restored spatial state as normal even if a hand is still holding it. Do not treat every package outside a target zone or a normally carried package as recovery. There may be 0-6 parcels; flipping is not mandatory.
"""
SKILL_OUTPUT = 'Return ONLY {"skill":"normal"} or {"skill":"recovery"}. No explanation, reasoning, markdown or other fields.'
BOOL_OUTPUT = 'Return ONLY {"recovery_required":false} for normal or {"recovery_required":true} for recovery. No explanation, reasoning, markdown or other fields.'
DESCRIPTION = """Describe only the visible workspace and parcel positions in these chronological images. Focus on the LAST image and the parcel nearest the robot-facing/front edge of the yellow platform. Where are its upper and lower parts relative to the platform top and near edge? Is it wholly on the platform, extending over the near edge, held above it, or impossible to tell? Is any part cut off by the image boundary or hidden by a hand? Distinguish visible contact/support from inference. Describe any actual downward fall only if the supplied images show it; otherwise say that a fall is not shown. Do not use the clip name or assume a failure. Answer briefly in plain text."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write_new(path: Path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def prepare(prompt_config=None, labels_csv=None):
    custom = read(Path(prompt_config)) if prompt_config else None
    reviewed_labels = None
    if custom:
        if not labels_csv:
            raise ValueError('A new task definition requires --labels-csv with reviewed per-endpoint labels; do not reuse labels by video name')
        with Path(labels_csv).open() as f:
            reviewed_labels = {r['id']: r for r in csv.DictReader(f)}
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        raise FileExistsError('Manifest already locked; use existing experiment, do not overwrite labels')
    base = read(BASE / (MODELS[0] + '.json'))
    old_prompt = base['prompt']
    assert read(BASE / (MODELS[1] + '.json'))['prompt'] == old_prompt
    windows = []
    for w in base['windows']:
        clip = w['clip']
        end = w['frame_times_s'][-1]
        original = [str(ROOT / p) for p in w['frames']]
        end_index = int(Path(original[-1]).stem.split('_')[-1]) - 1
        all_paths = sorted(FRAMES.glob(clip + '_*.jpg'))
        label = 'recovery' if clip == 'recovery' and end <= 11 else 'normal'
        reference = {
            ('recovery', 5): 'User confirmed that the near/front-edge parcel state is the recovery start. Parcel extends down beyond near edge; lower part is cropped; no fall event is shown.',
            ('recovery', 11): 'Parcel remains at robot-facing near edge; left hand touches/manipulates it. In-progress recovery stays recovery under the state-label task; contact is partly occluded.',
            ('recovery', 17): 'Parcel has been lifted back over/onto yellow platform; footprint is inside the platform. Hand still holds it; support/release not confirmed. Normal here means restored spatial state, not completed skill.',
            ('recovery', 19): 'Parcel footprint is inside the platform; left hand remains close/in contact. Normal denotes restored spatial state, not confirmed release.',
        }.get((clip, end), 'Normal task clip: supply, on-platform manipulation, or rightward normal transfer; no displaced parcel at body-side near edge.')
        groups = {'original': original,
                  'final': [original[-1]],
                  'short': [str(p) for p in all_paths[max(0, end_index - 1):end_index + 1]],
                  'long': [str(p) for p in all_paths[max(0, end_index - 4):end_index + 1]]}
        index = (int(Path(original[0]).stem.split('_')[-1]) - 1)
        start_s = 2 * index
        stop_s = min(start_s + 6, 25 if clip == 'normal' else 20)
        windows.append(dict(id=f'{clip}_{end:02d}', clip=clip,
                            original_interval_s=[start_s, stop_s], final_time_approx_s=end,
                            ground_truth=label, ground_truth_source='user-defined start + manual spatial-state reference; not independent blind annotation',
                            reference_evidence=reference,
                            end_of_recovery_boundary_sensitive=clip == 'recovery' and end == 17,
                            lower_support_visible=False if clip == 'recovery' and end <= 11 else None,
                            groups=groups))
    if reviewed_labels is not None:
        if set(reviewed_labels) != {w['id'] for w in windows}:
            raise ValueError('Reviewed labels must cover exactly the same 9 endpoint IDs')
        for w in windows:
            row = reviewed_labels[w['id']]
            if row['ground_truth'] not in ['normal', 'recovery'] or not row.get('reference_evidence', '').strip():
                raise ValueError('Each reviewed endpoint requires normal/recovery and reference_evidence')
            w.update(ground_truth=row['ground_truth'], reference_evidence=row['reference_evidence'],
                     ground_truth_source='Explicit per-endpoint review under ' + custom['definition_version'])
    protected = [ROOT / 'vlm_test/benchmark_package_recovery.py']
    for directory in [ROOT / 'outputs/qwen_recovery_bf16', BASE]:
        protected += [p for p in directory.rglob('*') if p.is_file()]
    if custom:
        prior = ROOT / 'outputs/qwen_skill_control_20261028'
        protected += [p for p in prior.rglob('*') if p.is_file()]
    file_hashes = {str(p): digest(p) for p in sorted(protected)}
    inputs = {str(p): digest(p) for p in sorted(FRAMES.glob('*.jpg'))}
    for clip in ['normal', 'recovery']:
        p = Path('/media/data/jiayi/dataset') / (clip + '_clip.mp4')
        inputs[str(p)] = digest(p)
    manifest = dict(
        branch='lingbot_vlm_vla (actual checkout name; no branch named lingbot-vlm-vla exists)',
        models=MODELS, baseline_prompt=old_prompt, skill_prompt=STATE_RULES + SKILL_OUTPUT,
        state_boolean_prompt=STATE_RULES + BOOL_OUTPUT, description_prompt=DESCRIPTION,
        settings=dict(precision='nf4', quant_type='nf4', double_quant=True,
                      compute_dtype='bfloat16', torch_dtype='bfloat16', attention='sdpa',
                      do_sample=False, max_new_tokens=24, enable_thinking=False,
                      sampling_fps=0.5, jpeg_size=[640, 362], use_same_processor_defaults=True,
                      context_memory=False, physical_gpu=1, logical_device='cuda:0',
                      gpu_note='GPU0 is occupied. GPU1 is same RTX PRO 6000 Blackwell model; old results used GPU0. Do not infer precise speed changes across runs.',
                      description_max_new_tokens=192),
        variants=dict(original='Exact same 9 windows and JPEG bytes as previous NF4 baseline; <=3 frames.',
                      final='Only the final image at the identical endpoint.',
                      short='Last <=2 samples, ~2s first-to-last span, same endpoint and 0.5fps.',
                      long='Last <=5 samples, ~8s first-to-last span, same endpoint and 0.5fps; context can cross original interval start.'),
        label_definition='Recovery starts at the near-edge displaced state per user. Recovery stays positive during near-edge grasping. Restored footprint inside/over platform is normal; no requirement to show completed release. Labels are endpoint labels, not clip labels.',
        label_limitations='Partial cropping prevents proof of lower support and no actual fall is visible. Reference labels follow user task semantics, not a physical-drop detector. Restored-but-held boundary at ~17s is sensitivity-tested separately.',
        windows=windows, protected_file_sha256=file_hashes, input_sha256=inputs,
    )
    if custom:
        for key in ['skill_prompt', 'state_boolean_prompt', 'label_definition', 'label_limitations']:
            manifest[key] = custom[key]
        manifest['definition_version'] = custom['definition_version']
        manifest['prompt_config_sha256'] = digest(Path(prompt_config))
        manifest['reviewed_labels_sha256'] = digest(Path(labels_csv))
    write_new(OUT / 'manifest.json', manifest)
    write_new(OUT / 'prompts.json', {k: manifest[k] for k in ['baseline_prompt', 'skill_prompt', 'state_boolean_prompt', 'description_prompt']})
    with (OUT / 'ground_truth.csv').open('x') as f:
        fields = ['id', 'clip', 'original_interval_s', 'final_time_approx_s', 'ground_truth', 'end_of_recovery_boundary_sensitive', 'reference_evidence']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for w in windows:
            writer.writerow({k: w[k] for k in fields})
    print('Locked labels, prompts and hashes for 9 endpoints before inference.')


def verify(manifest):
    for key in ['protected_file_sha256', 'input_sha256']:
        changed = [p for p, h in manifest[key].items() if digest(Path(p)) != h]
        if changed:
            raise RuntimeError(f'Protected/input files changed: {changed}')


def parse_output(raw: str, mode: str):
    try:
        # Preserve raw text and record formatting compliance separately.
        value = json.loads(raw[raw.index('{'):raw.rindex('}') + 1])
        strict = json.loads(raw.strip()) == value
    except (ValueError, json.JSONDecodeError):
        try:
            value = json.loads(raw[raw.index('{'):raw.rindex('}') + 1])
        except (ValueError, json.JSONDecodeError):
            return None, False
        strict = False
    if mode == 'skill':
        valid = isinstance(value, dict) and set(value) == {'skill'} and value['skill'] in ['normal', 'recovery']
        return (value['skill'] if valid else None), strict and valid
    valid = isinstance(value, dict) and set(value) == {'recovery_required'} and type(value['recovery_required']) is bool
    return (('recovery' if value['recovery_required'] else 'normal') if valid else None), strict and valid


def run(model_name: str):
    manifest = read(OUT / 'manifest.json')
    verify(manifest)
    if model_name == MODELS[1]:
        previous = OUT / (MODELS[0] + '.json')
        if not previous.exists() or not read(previous).get('complete'):
            raise RuntimeError('Finish 9B experiment first')
    dest = OUT / (model_name + '.json')
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite {dest}')
    import torch
    import transformers
    from PIL import Image
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16-capable CUDA GPU required')
    model_path = '/media/data/jiayi/hf_model/' + model_name
    processor = transformers.AutoProcessor.from_pretrained(model_path, local_files_only=True)
    t = time.perf_counter()
    model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map='cuda:0', local_files_only=True,
        attn_implementation='sdpa', quantization_config=transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    q = model.config.quantization_config
    result = dict(model=model_name, manifest_sha256=digest(OUT / 'manifest.json'),
                  versions={'torch': torch.__version__, 'transformers': transformers.__version__},
                  gpu=torch.cuda.get_device_name(0), quantization_config=q.to_dict() if hasattr(q, 'to_dict') else q,
                  load_seconds=time.perf_counter() - t, memory_footprint_bytes=model.get_memory_footprint(),
                  requests=[], descriptions=[], complete=False)
    write_new(dest, result)

    def save():
        temp = dest.with_suffix('.tmp')
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        temp.replace(dest)

    def infer(prompt, paths, limit=24):
        content = [{'type': 'text', 'text': prompt}]
        images = []
        times = []
        for path in paths:
            timestamp = (int(Path(path).stem.split('_')[-1]) - 1) * 2 + 1
            times.append(timestamp)
            with Image.open(path) as im:
                images.append(im.convert('RGB'))
            content.extend([{'type': 'text', 'text': f'Frame near t={timestamp}s'},
                            {'type': 'image', 'image': images[-1]}])
        text = processor.apply_chat_template([{'role': 'user', 'content': content}],
                                            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=[text], images=images, return_tensors='pt', padding=True).to('cuda:0')
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(**inputs, do_sample=False, max_new_tokens=limit)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        answer_ids = ids[:, inputs['input_ids'].shape[1]:]
        raw = processor.batch_decode(answer_ids, skip_special_tokens=True)[0]
        return dict(raw=raw, generation_seconds=elapsed, frame_times_approx_s=times,
                    frame_paths=paths, generated_tokens=answer_ids.shape[1], input_tokens=inputs['input_ids'].shape[1])

    arms = [('old_replay', 'original', 'baseline_prompt', 'boolean')]
    arms += [('skill', v, 'skill_prompt', 'skill') for v in ['original', 'final', 'short', 'long']]
    arms += [('state_boolean', 'original', 'state_boolean_prompt', 'boolean')]
    for arm, variant, prompt_key, parser_mode in arms:
        for w in manifest['windows']:
            row = dict(arm=arm, variant=variant, window_id=w['id'], clip=w['clip'],
                       interval_s=w['original_interval_s'], ground_truth=w['ground_truth'],
                       **infer(manifest[prompt_key], w['groups'][variant]))
            row['prediction'], row['strict_json'] = parse_output(row['raw'], parser_mode)
            result['requests'].append(row)
            save()
            print(json.dumps(row, ensure_ascii=False), flush=True)
    # Required fallback: if primary binary skill prompts all return normal, inspect all windows openly.
    primary = [r for r in result['requests'] if r['arm'] == 'skill' and r['variant'] == 'original']
    all_normal = all(r['prediction'] == 'normal' for r in primary)
    result['all_primary_skill_predictions_normal'] = all_normal
    diagnostic_windows = manifest['windows'] if all_normal else [w for w in manifest['windows'] if w['id'] in ['normal_05', 'recovery_05', 'recovery_11']]
    for w in diagnostic_windows:
        row = dict(window_id=w['id'], variant='original', reason='all-normal fallback' if all_normal else 'visual sanity check',
                   **infer(manifest['description_prompt'], w['groups']['original'], limit=192))
        result['descriptions'].append(row)
        save()
        print('DESCRIPTION ' + json.dumps(row, ensure_ascii=False), flush=True)
    # Compare initial abnormal endpoint with single-image description without its earlier context.
    w = next(w for w in manifest['windows'] if w['id'] == 'recovery_05')
    row = dict(window_id=w['id'], variant='final', reason='single-frame diagnostic',
               **infer(manifest['description_prompt'], w['groups']['final'], limit=192))
    result['descriptions'].append(row)
    result['complete'] = True
    verify(manifest)
    save()
    print('COMPLETE ' + model_name, flush=True)


def metrics(rows):
    labels = ['normal', 'recovery']
    matrix = [[sum(r['ground_truth'] == a and r['prediction'] == b for r in rows) for b in labels] for a in labels]
    per_class = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        actual = sum(r['ground_truth'] == label for r in rows)
        predicted = sum(r['prediction'] == label for r in rows)
        # Undefined precision with no predictions is null, not an invented 0/1.
        per_class[label] = dict(precision=tp / predicted if predicted else None,
                                recall=tp / actual if actual else None, support=actual)
    return dict(class_order=labels, confusion_matrix_rows_true_columns_predicted=matrix,
                per_class=per_class, invalid=sum(r['prediction'] is None for r in rows),
                strict_json_count=sum(r['strict_json'] for r in rows), total=len(rows))


def summarize():
    manifest = read(OUT / 'manifest.json')
    verify(manifest)
    results = [read(OUT / (m + '.json')) for m in MODELS if (OUT / (m + '.json')).exists()]
    all_rows = [dict(model=d['model'], **r) for d in results for r in d['requests']]
    table = {}
    for d in results:
        for arm, variant in sorted({(r['arm'], r['variant']) for r in d['requests']}):
            rows = [r for r in d['requests'] if r['arm'] == arm and r['variant'] == variant]
            value = metrics(rows)
            value['generation_median_seconds'] = statistics.median(r['generation_seconds'] for r in rows)
            value['without_recovery_17_boundary'] = metrics([r for r in rows if r['window_id'] != 'recovery_17'])
            value['alternative_keep_recovery_while_still_held'] = metrics([
                dict(r, ground_truth='recovery') if r['window_id'] in ['recovery_17', 'recovery_19'] else r
                for r in rows])
            table[f"{d['model']}/{arm}/{variant}"] = value
    (OUT / 'metrics.json').write_text(json.dumps(table, ensure_ascii=False, indent=2))
    fields = ['model', 'arm', 'variant', 'window_id', 'clip', 'interval_s', 'frame_times_approx_s', 'ground_truth', 'prediction', 'strict_json', 'generation_seconds', 'raw']
    with (OUT / 'predictions.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row[k] for k in fields})
    def percentage(v):
        return 'N/A' if v is None else f'{100 * v:.1f}%'
    lines = ['# Kuavo Package Sorting：离线受控对照', '',
             '实际分支：`lingbot_vlm_vla`。旧 Prompt、旧结果、原始视频与抽帧文件 SHA256 已核对未变化。', '',
             '## 标签和视频事实', '',
             manifest['label_definition'],
             manifest['label_limitations'],
             '标签为推理前固定的人工参考，非独立盲标。metrics.json 另含排除17秒点和将仍被手持的17/19秒也记作recovery的敏感性分析，不替换主标签。', '',
             '| 窗口 | 原视频时间段 s | 原始采样近似时刻 s | 人工真值 | 依据 |', '|---|---|---|---|---|']
    for w in manifest['windows']:
        times = [(int(Path(p).stem.split('_')[-1]) - 1) * 2 + 1 for p in w['groups']['original']]
        lines.append(f"| {w['id']} | {w['original_interval_s']} | {times} | {w['ground_truth']} | {w['reference_evidence']} |")
    lines += ['', '## 控制变量', '',
              '主对照直接复用原 9 个窗口的 JPEG 字节，640×362、0.5fps、最多3帧。最终单帧/短窗口≤2帧/长窗口≤5帧共享同一最终帧；改变的只有上下文。开头不足长度时截断，可能与原窗口重复，不当作额外独立样本。',
              '两模型使用此前 NF4 配置：double quant、BF16 compute、SDPA、do_sample=false、max_new_tokens=24、enable_thinking=false。GPU0 正被占用，使用同型号 GPU1（逻辑cuda:0）；不据此作精确速度优劣结论。开放描述是独立诊断，生成上限192，不参与分类指标。',
              'old_replay 重放原 recovery_required Prompt；skill 是新的场景状态语义；state_boolean 与 skill 场景规则逐字相同，仅输出字段/值不同，用于分离标签格式影响与语义影响。旧结果全部保留。', '',
              '## Precision / Recall', '',
              f"每组9个端点，normal={sum(w['ground_truth'] == 'normal' for w in manifest['windows'])}、recovery={sum(w['ground_truth'] == 'recovery' for w in manifest['windows'])}。两类分别按 one-vs-rest 计算；无该类预测时 Precision 为 N/A。混淆矩阵顺序为 [normal,recovery]，行=真值，列=预测。无效输出单列统计，不悄悄改成 normal；Recall 分母含所有真值。窗口来自同两条视频且有重叠，不代表独立测试集。", '',
              '| 模型/Prompt/输入 | Normal P | Normal R | Recovery P | Recovery R | 混淆矩阵 | 无效 | 严格JSON | 中位生成s |', '|---|---|---|---|---|---|---|---|---|']
    for key, v in table.items():
        n, r = v['per_class']['normal'], v['per_class']['recovery']
        lines.append(f"| {key} | {percentage(n['precision'])} | {percentage(n['recall'])} | {percentage(r['precision'])} | {percentage(r['recall'])} | {v['confusion_matrix_rows_true_columns_predicted']} | {v['invalid']} | {v['strict_json_count']}/{v['total']} | {v['generation_median_seconds']:.3f} |")
    lines += ['', '## 每个端点的 skill 预测', '', '| 模型 | 视频窗口 | 时间段s | 真值 | 原3帧 | 最终单帧 | 短≤2帧 | 长≤5帧 |', '|---|---|---|---|---|---|---|---|']
    for d in results:
        for w in manifest['windows']:
            values = []
            for variant in ['original', 'final', 'short', 'long']:
                r = next((r for r in d['requests'] if r['window_id'] == w['id'] and r['arm'] == 'skill' and r['variant'] == variant), None)
                values.append(r['prediction'] or 'INVALID' if r else 'pending')
            lines.append(f"| {d['model']} | {w['id']} | {w['original_interval_s']} | {w['ground_truth']} | " + ' | '.join(values) + ' |')
    lines += ['', '## 开放式视觉诊断（模型原文，不等于事实确认）', '']
    for d in results:
        for row in d['descriptions']:
            lines += [f"### {d['model']} / {row['window_id']} / {row['variant']}", '', row['raw'], '']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(table, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'summarize', 'verify'])
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--prompt-config', type=Path)
    parser.add_argument('--labels-csv', type=Path)
    args = parser.parse_args()
    if args.output_dir:
        OUT = args.output_dir.resolve()
    if args.action == 'prepare':
        prepare(args.prompt_config, args.labels_csv)
    elif args.action == 'run':
        if not args.model:
            parser.error('--model required for run')
        run(args.model)
    elif args.action == 'verify':
        verify(read(OUT / 'manifest.json'))
        print('All protected files and input bytes unchanged.')
    else:
        summarize()
