"""Summarize the saved offline monitor results without assigning ground truth."""
import csv
import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='outputs/qwen_recovery_bf16')
    args = parser.parse_args()
    root = Path(args.output_dir)
    records = []
    for path in sorted(root.glob('Qwen*.json')):
        data = json.loads(path.read_text())
        for w in data['windows']:
            parsed = w.get('parsed')
            valid = (isinstance(parsed, dict) and set(parsed) == {'recovery_required'}
                     and (type(parsed['recovery_required']) is bool or parsed['recovery_required'] == 'unknown'))
            records.append(dict(model=data['model'], clip=w['clip'],
                                frame_times_s=','.join(map(str, w['frame_times_s'])),
                                recovery_required=str(parsed.get('recovery_required')) if valid else 'INVALID',
                                generation_seconds=w['generation_seconds'], valid=valid))
    if not records:
        return
    with (root / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    lines = ['# Qwen recovery monitor comparison', '',
             'Same prompt and sampled images; output contains only recovery_required.',
             'Times cover synchronized GPU generation (prefill + decode), excluding loading and preprocessing.',
             'These two clips have no independent temporal ground-truth annotations; counts are predictions, not accuracy.', '',
             '| Model | Normal true/false/unknown | Recovery true/false/unknown | Invalid | Median generation (s) |',
             '|---|---|---|---|---|']
    for model in sorted({r['model'] for r in records}):
        rows = [r for r in records if r['model'] == model]
        counts = []
        for clip in ['normal', 'recovery']:
            counts.append('/'.join(str(sum(r['clip'] == clip and r['recovery_required'] == value for r in rows))
                                   for value in ['True', 'False', 'unknown']))
        lines.append(f"| {model} | {counts[0]} | {counts[1]} | {sum(not r['valid'] for r in rows)} | {statistics.median(r['generation_seconds'] for r in rows):.3f} |")
    lines += ['', '## Window predictions', '', '| Model | Clip | Approximate frame times (s) | Recovery | Generation (s) |', '|---|---|---|---|---|']
    for r in records:
        lines.append(f"| {r['model']} | {r['clip']} | {r['frame_times_s']} | {r['recovery_required']} | {r['generation_seconds']:.3f} |")
    (root / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:14]))


if __name__ == '__main__':
    main()
