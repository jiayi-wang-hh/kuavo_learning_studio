"""Compare local models on frozen spatial-state labels with one output token."""
import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--precision', choices=['bf16', 'nf4', 'fp8'], default='bf16')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output-dir', default='outputs/qwen_one_token_20261108')
    args = p.parse_args()
    import torch
    import transformers
    from PIL import Image
    torch.set_num_threads(8)
    manifest_path = Path('outputs/qwen_skill_control_20261028/manifest.json')
    manifest = json.loads(manifest_path.read_text())
    prompt = manifest['skill_prompt'].split('Return ONLY')[0] + (
        'Return exactly ONE token: 0 for normal, 1 for recovery. '
        'Do not explain. Do not output any other text.'
    )
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f'{args.model}_{args.precision}.json'
    if dest.exists():
        raise FileExistsError(dest)
    model_path = '/media/data/jiayi/hf_model/' + args.model
    processor = transformers.AutoProcessor.from_pretrained(model_path, local_files_only=True)
    label_ids = {s: processor.tokenizer.encode(s, add_special_tokens=False) for s in ['0', '1']}
    assert all(len(ids) == 1 for ids in label_ids.values()), label_ids
    kwargs = {}
    if args.precision == 'nf4':
        kwargs['quantization_config'] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    elif args.precision == 'fp8':
        kwargs['quantization_config'] = transformers.FineGrainedFP8Config()
    cls = (transformers.Qwen3_5MoeForConditionalGeneration if '35B' in args.model
           else transformers.Qwen3_5ForConditionalGeneration)
    started = time.perf_counter()
    model = cls.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16,
                                device_map='cuda:0', attn_implementation='sdpa', **kwargs).eval()
    result = dict(model=args.model, precision=args.precision, load_seconds=time.perf_counter()-started,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, transformers=transformers.__version__,
                  prompt=prompt, label_ids=label_ids, generation='unconstrained greedy, max_new_tokens=1, thinking disabled',
                  reference=str(manifest_path), reference_limit='9 endpoints from 2 clips; manual, not blind labels; recovery_17 boundary sensitive',
                  memory_footprint_bytes=model.get_memory_footprint(),
                  quantized_modules=sum('4bit' in type(m).__name__.lower() for m in model.modules()), rows=[])

    def generate(inputs):
        with torch.inference_mode():
            return model.generate(**inputs, do_sample=False, max_new_tokens=1)

    for group in ['final', 'original']:
        prepared = []
        for w in manifest['windows']:
            images = [Image.open(f).convert('RGB') for f in w['groups'][group]]
            content = [{'type': 'image', 'image': im} for im in images]
            content.append({'type': 'text', 'text': prompt})
            text = processor.apply_chat_template([{'role': 'user', 'content': content}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
            start = time.perf_counter()
            inputs = processor(text=[text], images=images, return_tensors='pt').to('cuda:0')
            torch.cuda.synchronize()
            preprocessing = time.perf_counter()-start
            prepared.append((w, inputs, preprocessing))
        for _ in range(2):
            generate(prepared[0][1])
        torch.cuda.synchronize()
        for repeat in range(args.repeats):
            for w, inputs, preprocessing in prepared:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                ids = generate(inputs)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
                token_ids = ids[0, inputs['input_ids'].shape[1]:].tolist()
                raw = processor.tokenizer.decode(token_ids, skip_special_tokens=False)
                prediction = {'0': 'normal', '1': 'recovery'}.get(raw.strip(), 'invalid')
                row = dict(id=w['id'], group=group, repeat=repeat, truth=w['ground_truth'],
                           prediction=prediction, raw=raw, token_ids=token_ids,
                           input_tokens=inputs['input_ids'].shape[1], generation_seconds=elapsed,
                           preprocessing_seconds=preprocessing, peak_allocated_bytes=torch.cuda.max_memory_allocated())
                result['rows'].append(row)
                dest.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                print(json.dumps(dict(model=args.model, precision=args.precision, **row)), flush=True)
        del prepared
    result['summary'] = {}
    for group in ['final', 'original']:
        rows = [r for r in result['rows'] if r['group'] == group]
        # Accuracy uses one prediction per endpoint; repeats measure speed and stability.
        first = [r for r in rows if r['repeat'] == 0]
        tp = sum(r['truth'] == r['prediction'] == 'recovery' for r in first)
        fp = sum(r['truth'] != 'recovery' and r['prediction'] == 'recovery' for r in first)
        positive = sum(r['truth'] == 'recovery' for r in first)
        times = sorted(r['generation_seconds'] for r in rows)
        result['summary'][group] = dict(
            correct=sum(r['truth'] == r['prediction'] for r in first), n=len(first),
            recovery_precision=tp/(tp+fp) if tp+fp else None, recovery_recall=tp/positive,
            invalid=sum(r['prediction'] == 'invalid' for r in first),
            median_seconds=statistics.median(times), p95_seconds=times[min(len(times)-1, int(len(times)*.95))],
            stable=all(len({r['prediction'] for r in rows if r['id'] == w['id']}) == 1 for w in manifest['windows']))
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
