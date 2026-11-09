"""Offline, causal BF16 comparison of local Qwen recovery monitors."""
import argparse
import json
import time
from pathlib import Path

PROMPT = """Observe a bimanual package-sorting robot. Images are chronological and end at the current observation time. Judge ONLY the CURRENT final state, using earlier frames for motion context.
Normal task: pick parcels from the left supply area, place them on the yellow platform, manipulate/reorient/flip when needed, then transfer to the right basket. Do not assume every parcel needs flipping. There may be 0-6 parcels. Parcels waiting on the left, in the destination basket, being carried by either gripper, or undergoing normal manipulation are not failures.
Specific recovery condition: a parcel has slipped off the near/front edge of the elevated yellow platform toward the robot's body, is supported in the lower body-side region rather than on the platform, and is not currently being held or already actively lifted back by a gripper. This is not necessarily the floor. Do not classify every parcel outside a target zone, slight overhang, or normal in-flight transfer as requiring recovery. Check actual platform support, package position, grip and temporal motion; do not infer grasp merely from proximity.
Recovery skill: left arm picks that displaced parcel from the lower body-side region and returns it onto the yellow platform. If recovery is visibly in progress, do not request a new recovery. If visibility is insufficient use "unknown".
Return ONLY one JSON object with one field: {"recovery_required": true} or {"recovery_required": false} or {"recovery_required": "unknown"}. Do not output explanations, reasoning, markdown or other fields.
"""

MODELS = {
    'Qwen2.5-VL-3B-Instruct': 'Qwen2_5_VLForConditionalGeneration',
    'Qwen2.5-VL-7B-Instruct': 'Qwen2_5_VLForConditionalGeneration',
    'Qwen3-VL-4B-Instruct': 'Qwen3VLForConditionalGeneration',
    'Qwen3.5-4B': 'Qwen3_5ForConditionalGeneration',
    'Qwen3.5-9B': 'Qwen3_5ForConditionalGeneration',
    'Qwen3.5-27B': 'Qwen3_5ForConditionalGeneration',
    'Qwen3.5-35B-A3B': 'Qwen3_5MoeForConditionalGeneration',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--output-dir', default='outputs/qwen_recovery_bf16')
    p.add_argument('--frames-dir', default='outputs/qwen_recovery_bf16/frames')
    p.add_argument('--precision', choices=['bf16', 'nf4'], default='bf16')
    args = p.parse_args()
    import torch
    import transformers
    from PIL import Image
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU supporting BF16 is required')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = '/media/data/jiayi/hf_model/' + args.model
    processor = transformers.AutoProcessor.from_pretrained(path, local_files_only=True)
    load_kwargs = {}
    if args.precision == 'nf4':
        load_kwargs['quantization_config'] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    model = getattr(transformers, MODELS[args.model]).from_pretrained(
        path, dtype=torch.bfloat16, device_map='cuda:0',
        attn_implementation='sdpa', local_files_only=True,
        **load_kwargs,
    ).eval()
    quant_config = getattr(model.config, 'quantization_config', None)
    if hasattr(quant_config, 'to_dict'):
        quant_config = quant_config.to_dict()
    result = dict(model=args.model, dtype=str(next(model.parameters()).dtype),
                  precision=args.precision, quantization_config=quant_config,
                  quantized_module_count=sum('4bit' in type(module).__name__.lower() for module in model.modules()),
                  memory_footprint_bytes=model.get_memory_footprint(),
                  transformers=transformers.__version__, torch=torch.__version__,
                  prompt=PROMPT, sampling='FFmpeg fps=0.5, width=640; disjoint causal windows of up to 3 frames', windows=[])
    dest = out / (args.model + '.json')
    for clip in ['normal', 'recovery']:
        paths = sorted(Path(args.frames_dir).glob(clip + '_*.jpg'))
        if not paths:
            raise FileNotFoundError(f'No sampled frames for {clip} in {args.frames_dir}')
        for start in range(0, len(paths), 3):
            selected = paths[start:start + 3]
            content = [{'type':'text', 'text':PROMPT}]
            for i, frame in enumerate(selected, start):
                content.extend([{'type':'text','text':f'Frame near t={i*2+1}s'},
                                {'type':'image','image':Image.open(frame).convert('RGB')}])
            messages = [{'role':'user', 'content':content}]
            kw = {'enable_thinking':False} if '3.5' in args.model else {}
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
            images = [v['image'] for v in content if v['type']=='image']
            inputs = processor(text=[text], images=images, return_tensors='pt', padding=True).to('cuda:0')
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.inference_mode():
                ids = model.generate(**inputs, do_sample=False, max_new_tokens=24)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-t
            raw = processor.batch_decode(ids[:,inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            try:
                parsed = json.loads(raw[raw.index('{'):raw.rindex('}')+1])
            except (ValueError, json.JSONDecodeError):
                parsed = None
            row = dict(clip=clip, frame_times_s=[i*2+1 for i in range(start,start+len(selected))], frames=[str(f) for f in selected], generation_seconds=elapsed, raw=raw, parsed=parsed)
            result['windows'].append(row)
            dest.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(json.dumps(dict(model=args.model, **row), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
