import json
from copy import deepcopy
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Literal
from collections import defaultdict
import numpy as np
import torch
import torch.distributed as dist
import wandb
from PIL import Image
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from lingbotvla.checkpoint import build_checkpointer, ckpt_to_state_dict
from lingbotvla.data import (
    VLADataCollatorWithPacking,
    build_dataloader,
)
from lingbotvla.data.vla_data import VLADataset
from lingbotvla.distributed.offloading import build_activation_offloading_context
from lingbotvla.distributed.parallel_state import get_parallel_state, init_parallel_state
from lingbotvla.distributed.torch_parallelize import build_parallelize_model
from lingbotvla.models import build_foundation_model, build_processor, save_model_assets, save_model_weights, build_tokenizer
from lingbotvla.optim import build_lr_scheduler, build_optimizer
from lingbotvla.utils import helper
from lingbotvla.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from lingbotvla.utils.dist_utils import all_reduce
from lingbotvla.utils.lora_utils import add_lora_to_model, count_trainable_parameters, mark_only_lora_and_modules_trainable

from lingbotvla.models.vla.vision_models.module_utils import build_depth_model, get_depth_target, log_depth



logger = helper.create_logger(__name__)

def get_param_groups(model: "torch.nn.Module", default_lr: float, vit_lr: float):
    vit_params, other_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "visual" in name:
                vit_params.append(param)
            else:
                other_params.append(param)

    return [{"params": vit_params, "lr": vit_lr}, {"params": other_params, "lr": default_lr}]


def prune_checkpoints(checkpoint_dir: str, max_to_keep: int):
    if max_to_keep <= 0 or not os.path.isdir(checkpoint_dir):
        return

    pattern = re.compile(r"global_step_(\d+)")
    checkpoints = []
    for dirname in os.listdir(checkpoint_dir):
        match = pattern.fullmatch(dirname)
        if match:
            checkpoints.append((int(match.group(1)), os.path.join(checkpoint_dir, dirname)))
    checkpoints.sort(key=lambda item: item[0], reverse=True)

    for step, path in checkpoints[max_to_keep:]:
        shutil.rmtree(path, ignore_errors=True)
        logger.info_rank0(f"Pruned old checkpoint global_step_{step}: {path}")


@dataclass
class Arguments:

    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


def main():
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    torch.cuda.set_device(f"cuda:{args.train.local_rank}")
    dist.init_process_group(backend="nccl")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare model")

    config_kwargs = {**vars(args.model), **vars(args.train)}
    if args.train.enable_expert_vision and not args.model.post_training:
        assert args.train.expert_vision_path is not None, "expert_vision_path is required when enable_expert_vision is True!!!"
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        init_device=args.train.init_device,
        freeze_vision_encoder=args.train.freeze_vision_encoder,
        tokenizer_max_length=args.train.tokenizer_max_length,
        vocab_size=args.model.vocab_size,
        use_lm_head=args.model.use_lm_head,
        force_use_huggingface=args.model.force_use_huggingface,
        config_kwargs=config_kwargs,
    )
    use_depth_align = True if args.train.align_params != {} else False
    depth_model_type = None
    if use_depth_align:
        assert args.model.moge_path is not None and args.model.morgbd_path is not None, 'Depth models need to be loaded when uing LingBot-VLA-Depth!'
        args.train.align_params['visual_dir'] = os.path.join(args.train.output_dir, 'images')
        args.train.align_params['depth']['moge_path'] = args.model.moge_path
        args.train.align_params['depth']['morgbd_path'] = args.model.morgbd_path
        depth_model_type = args.train.align_params['depth']['model_type']
        moge_model, morgbd_model = build_depth_model(args.train.align_params)
        if args.train.use_compile:
            moge_model = torch.compile(moge_model)
            morgbd_model = torch.compile(morgbd_model)
        os.makedirs(args.train.align_params['visual_dir'], exist_ok=True)

    if args.train.use_lora:
        injected_count, injected_names = add_lora_to_model(
            model,
            lora_rank=args.train.lora_rank,
            lora_alpha=args.train.lora_alpha,
            lora_dropout=args.train.lora_dropout,
            lora_target_modules=args.train.lora_target_modules,
            lora_target_scope=args.train.lora_target_scope,
        )
        if injected_count == 0:
            raise ValueError(
                "LoRA is enabled, but no Linear layers matched "
                f"scope={args.train.lora_target_scope!r} and modules={args.train.lora_target_modules!r}."
            )
        trainable_names = mark_only_lora_and_modules_trainable(
            model,
            trainable_module_patterns=args.train.lora_trainable_modules,
        )
        trainable_params, total_params = count_trainable_parameters(model)
        preview = ", ".join(injected_names[:8])
        if injected_count > 8:
            preview += ", ..."
        logger.info_rank0(
            f"LoRA injected into {injected_count} Linear layers "
            f"(rank={args.train.lora_rank}, alpha={args.train.lora_alpha}, dropout={args.train.lora_dropout})."
        )
        logger.info_rank0(f"LoRA target preview: {preview}")
        logger.info_rank0(
            f"Trainable parameters: {trainable_params:,}/{total_params:,} "
            f"({trainable_params / total_params:.4%}); trainable tensors: {len(trainable_names)}"
        )
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    logger.info_rank0("Prepare data")
    processor = build_processor(args.model.tokenizer_path) # if use build_processor,  tokenizer is processor.tokenizer

    if args.train.rmpad:
        raise ValueError("Qwen2-VL does not support rmpad. Use `rmpad_with_pos_ids` instead.")

    data_collate_fn = []
    if args.data.datasets_type == 'vla':
        data_collate_fn.append(VLADataCollatorWithPacking())
    else:
        raise NotImplementedError(f"Unsupported dataset type: {args.data.datasets_type}.")

    if args.data.dataloader_type == "native":
        if args.data.datasets_type == 'vla':
            logger.info_rank0("Start building VLA dataset")
            args.data.chunk_size = args.train.chunk_size
            image_processor=processor.image_processor if 'qwen' in args.model.tokenizer_path.lower() else None

            train_dataset = VLADataset(repo_id=args.data.train_path, data_name =args.data.data_name, robot_config_root=args.data.robot_config_root, config=model.config, tokenizer=processor.tokenizer, data_config=args.data, image_processor=image_processor,use_depth_align=use_depth_align)
            
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, len(train_dataset))

        train_dataloader = build_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            collate_fn=data_collate_fn,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train.train_steps,
            rmpad=args.train.rmpad,
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor if args.data.num_workers > 0 else None,
            persistent_workers=args.data.persistent_workers,
        )
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")

    fsdp_kwargs = {}
    if args.train.freeze_vit:
        model.visual.requires_grad_(False)
        if args.train.data_parallel_mode == "fsdp1":
            fsdp_kwargs["use_orig_params"] = True


    model = build_parallelize_model(
        model,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_fp32=args.train.enable_fp32,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        init_device=args.train.init_device,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        fsdp_kwargs=fsdp_kwargs,
        basic_modules=model._no_split_modules if args.train.module_fsdp_enable else None,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        fsdp_llm_blocks=False,
    )
    
    if args.train.use_compile:
        model = torch.compile(model)


    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=False,
        optimizer_type=args.train.optimizer,
        post_training=args.model.post_training,
    )
    total_train_steps = args.train.train_steps * args.train.num_train_epochs
    if args.train.max_steps is not None:
        total_train_steps = min(total_train_steps, args.train.max_steps)
    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=total_train_steps,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        log_dir=f"{args.train.output_dir}/runs/"
        writer = SummaryWriter(log_dir=log_dir)
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
            )

        if args.train.enable_profiling:
            profiler = helper.create_profiler(
                start_step=args.train.profile_start_step,
                end_step=args.train.profile_end_step,
                trace_dir=args.train.profile_trace_dir,
                record_shapes=args.train.profile_record_shapes,
                profile_memory=args.train.profile_profile_memory,
                with_stack=args.train.profile_with_stack,
            )
            profiler.start()

        model_assets = [model_config, processor]
        save_model_assets(args.train.model_assets_dir, model_assets)

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
    )

    load_checkpoint_path = None
    candidates = []
    if args.train.load_checkpoint_path or args.train.enable_resume:
        if args.train.load_checkpoint_path:
            load_checkpoint_path = args.train.load_checkpoint_path
            candidates = [load_checkpoint_path]
        elif args.train.enable_resume:
            checkpoint_dir = f'{args.train.output_dir}/checkpoints'
            if os.path.exists(checkpoint_dir):
                pattern = re.compile(r"global_step_(\d+)")
                tmp = []
                for dirname in os.listdir(checkpoint_dir):
                    match = pattern.fullmatch(dirname)
                    if match:
                        step = int(match.group(1))
                        tmp.append((step, os.path.join(checkpoint_dir, dirname)))
                tmp.sort(key=lambda x: x[0], reverse=True)
                candidates = [p for _, p in tmp]
            if candidates:
                load_checkpoint_path = candidates[0]
            else:
                logger.info_rank0(f"No checkpoints in {args.train.output_dir} now!")
    if candidates:
        last_err = None
        loaded = False
        for cp in candidates:
            state = {"model": model, "optimizer": optimizer, "extra_state": {}}  # cannot be None
            try:
                Checkpointer.load(cp, state)
                global_step = state["extra_state"]["global_step"]
                start_epoch = global_step // args.train.train_steps
                start_step = global_step % args.train.train_steps
                lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
                if start_step > 0 and args.train.resume_dataloader_state:
                    train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
                environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
                torch.set_rng_state(state["extra_state"]["torch_rng_state"])
                if start_step == 0:  # resume at the end of epoch
                    iter(train_dataloader)  # clear resume state and prefetch data
                dist.barrier()
                logger.info_rank0(f"Load distributed checkpoint from {cp} successfully!")
                loaded = True
                break
            except Exception as e:
                last_err = e
                logger.info_rank0(f"Failed to load checkpoint {cp}: {repr(e)}. Trying older one...")
                continue
        if not loaded:
            logger.info_rank0("Starting training from scratch. No valid checkpoint could be loaded.")
    else:
        logger.info_rank0("Starting training from scratch.")

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    
    # create the path in advance to save loss log
    if args.train.global_rank == 0:
        os.makedirs(args.train.save_checkpoint_path, exist_ok=True)
    reached_max_steps = False
    profile_enabled = args.train.profile_training
    if profile_enabled:
        if args.train.profile_log_interval <= 0:
            raise ValueError("profile_log_interval must be positive")
        if args.train.profile_warmup_steps < 0:
            raise ValueError("profile_warmup_steps must be non-negative")
        logger.info_rank0(
            f"Lightweight training profiling enabled: warmup_steps={args.train.profile_warmup_steps}, "
            f"log_interval={args.train.profile_log_interval}. CUDA is synchronized only at report boundaries."
        )
    profile_data_time = 0.0
    profile_steps = 0
    profile_forward_events = []
    profile_backward_events = []
    profile_optimizer_events = []
    profile_interval_start = None
    max_steps_driven = (args.train.max_steps is not None and
                        args.train.max_steps < args.train.train_steps * args.train.num_train_epochs)
    if max_steps_driven:
        data_loader_tqdm = trange(
            args.train.max_steps,
            bar_format="Step: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
            initial=global_step,
            disable=args.train.local_rank != 0,
        )
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        if not max_steps_driven:
            data_loader_tqdm = trange(
                args.train.train_steps,
                desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
                total=args.train.train_steps,
                initial=start_step,
                disable=args.train.local_rank != 0,
            )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, args.train.train_steps):
            global_step += 1
            if (
                profile_enabled
                and global_step > args.train.profile_warmup_steps
                and profile_interval_start is None
            ):
                torch.cuda.synchronize()
                profile_interval_start = time.perf_counter()
            data_start_time = time.perf_counter()
            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            total_loss = 0
            total_vla_loss = 0
            total_depth_loss = 0
            depth_targets = None
            depth_preds = None
            start_time = time.time()
            if profile_enabled and global_step > args.train.profile_warmup_steps:
                profile_data_time += time.perf_counter() - data_start_time
                profile_steps += 1
            for micro_batch in micro_batches:
                dataset_names = micro_batch.pop('rep_id', None)
                environ_meter.add(micro_batch)

                micro_batch = {
                    k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in micro_batch.items()
                }
                
                if use_depth_align:
                    with torch.no_grad():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            pil_images = micro_batch.pop('pil_images', None)
                            depth_targets, cls_token = get_depth_target(depth_model_type, (moge_model, morgbd_model), pil_images)

                if profile_enabled and global_step > args.train.profile_warmup_steps:
                    forward_start = torch.cuda.Event(enable_timing=True)
                    forward_end = torch.cuda.Event(enable_timing=True)
                    forward_start.record()
                with model_fwd_context:
                    # torch.cuda.synchronize()
                    loss, vla_loss, depth_loss, loss_log, depth_preds = model(**micro_batch, vlm_causal = args.train.vlm_causal, depth_targets=depth_targets)
                    # torch.cuda.synchronize()

                    loss = loss / len(micro_batches)
                    vla_loss = vla_loss / len(micro_batches)
                    depth_loss = depth_loss / len(micro_batches)
                if profile_enabled and global_step > args.train.profile_warmup_steps:
                    forward_end.record()
                    profile_forward_events.append((forward_start, forward_end))

                if profile_enabled and global_step > args.train.profile_warmup_steps:
                    backward_start = torch.cuda.Event(enable_timing=True)
                    backward_end = torch.cuda.Event(enable_timing=True)
                    backward_start.record()
                with model_bwd_context:
                    loss.backward()
                if profile_enabled and global_step > args.train.profile_warmup_steps:
                    backward_end.record()
                    profile_backward_events.append((backward_start, backward_end))

                total_loss += loss.item()
                total_vla_loss += vla_loss.item()
                if not (isinstance(depth_loss, int) or isinstance(depth_loss, float)):
                    total_depth_loss += depth_loss.item()
                del micro_batch
            if profile_enabled and global_step > args.train.profile_warmup_steps:
                optimizer_start = torch.cuda.Event(enable_timing=True)
                optimizer_end = torch.cuda.Event(enable_timing=True)
                optimizer_start.record()
            if global_step > args.train.stable_train_steps:
                max_grad_norm = args.train.decayed_max_grad_norm
            else:
                max_grad_norm = args.train.max_grad_norm
            if args.train.data_parallel_mode == "fsdp1":
                grad_norm = model.clip_grad_norm_(max_grad_norm).item()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, foreach=True)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if profile_enabled and global_step > args.train.profile_warmup_steps:
                optimizer_end.record()
                profile_optimizer_events.append((optimizer_start, optimizer_end))
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            total_loss, total_vla_loss, total_depth_loss, grad_norm = all_reduce((total_loss, total_vla_loss, total_depth_loss, grad_norm), group=get_parallel_state().fsdp_group)
            
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            data_loader_tqdm.update()
            logger.info_rank0(
                f"Step {global_step}/{args.train.train_steps}, "
                f"Epoch {epoch+1}, "
                f"Loss {total_loss:.4f}, "
                f"VLA_Loss {total_vla_loss:.4f}, "
                f"Depth_Loss {total_depth_loss:.4f}, "
                f"GradNorm {grad_norm:.4f}, "
                f"LR {lr:.2e}, "
                f"StepTime {delta_time:.3f}s, "
            )

            if profile_enabled and profile_steps >= args.train.profile_log_interval:
                torch.cuda.synchronize()
                profile_interval_time = time.perf_counter() - profile_interval_start
                forward_time = sum(start.elapsed_time(end) for start, end in profile_forward_events) / 1000.0
                backward_time = sum(start.elapsed_time(end) for start, end in profile_backward_events) / 1000.0
                optimizer_time = sum(start.elapsed_time(end) for start, end in profile_optimizer_events) / 1000.0
                profile_totals = torch.tensor(
                    [profile_data_time, forward_time, backward_time, optimizer_time, profile_interval_time],
                    device=torch.cuda.current_device(),
                    dtype=torch.float64,
                )
                dist.all_reduce(profile_totals, op=dist.ReduceOp.MAX)
                profile_data_time, forward_time, backward_time, optimizer_time, profile_interval_time = (
                    profile_totals.cpu().tolist()
                )
                samples = profile_steps * args.train.global_batch_size
                logger.info_rank0(
                    "PERF "
                    f"steps={profile_steps}, data_time={profile_data_time / profile_steps:.4f}s/step, "
                    f"forward_time={forward_time / profile_steps:.4f}s/step, "
                    f"backward_time={backward_time / profile_steps:.4f}s/step, "
                    f"optimizer_time={optimizer_time / profile_steps:.4f}s/step, "
                    f"step_time={profile_interval_time / profile_steps:.4f}s, "
                    f"samples/s={samples / profile_interval_time:.3f}"
                )
                profile_data_time = 0.0
                profile_steps = 0
                profile_forward_events.clear()
                profile_backward_events.clear()
                profile_optimizer_events.clear()
                profile_interval_start = None


            if args.train.global_rank == 0:
                writer.add_scalar("training/loss", total_loss, global_step)
                writer.add_scalar("training/vla_loss", total_vla_loss, global_step)
                writer.add_scalar("training/depth_loss", total_depth_loss, global_step)
                writer.add_scalar("training/grad_norm", grad_norm, global_step)
                writer.add_scalar("training/lr", lr, global_step)
                writer.add_scalar("steptime", delta_time, global_step)
                # we only log the last mini batch if grad acc is activated
                if dataset_names is not None and 'batch_mean_losses' in loss_log:
                    batch_mean_losses = loss_log['batch_mean_losses']  # shape (B,)
                    if hasattr(batch_mean_losses, "detach"):
                        batch_mean_losses = batch_mean_losses.detach().cpu()

                    group_losses = defaultdict(list)
                    for name, loss_value in zip(dataset_names, batch_mean_losses):
                        group_losses[name].append(loss_value.item() if hasattr(loss_value, "item") else float(loss_value))

                    for name, values in group_losses.items():
                        mean_loss = sum(values) / len(values)
                        writer.add_scalar(f"detailed_loss/{name}", mean_loss, global_step)

                if args.train.enable_profiling and global_step <= args.train.profile_end_step:
                    profiler.step()
                    if global_step == args.train.profile_end_step:
                        profiler.stop()
                        helper.upload_trace(
                            args.train.wandb_project, args.train.wandb_name, args.train.profile_trace_dir
                        )

                loss_record = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": total_loss,
                    "grad_norm": grad_norm,
                    "lr": lr,
                    "step_time": delta_time
                }
                if args.train.use_wandb:
                    wandb.log(loss_record, step=global_step)
                loss_file_path = os.path.join(args.train.save_checkpoint_path, "loss.jsonl")
                try:
                    with open(loss_file_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(loss_record, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.info_rank0(f"⚠️ Failed to write loss.jsonl: {e}")


            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")

                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
                if args.train.global_rank == 0:
                    prune_checkpoints(args.train.save_checkpoint_path, args.train.max_checkpoints_to_keep)
                    if args.train.save_hf_weights and save_checkpoint_path is not None:
                        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
                        model_state_dict = ckpt_to_state_dict(
                            save_checkpoint_path=save_checkpoint_path,
                            output_dir=args.train.output_dir,
                            ckpt_manager=args.train.ckpt_manager,
                        )
                        if args.train.enable_fp32:
                            save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets, save_dtype=torch.float32)
                        else:
                            save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
                        save_args(args, hf_weights_path)
                        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

            # global max_steps early stop
            if args.train.max_steps is not None and global_step >= args.train.max_steps:
                logger.info_rank0(f"Reached max_steps={args.train.max_steps}, stopping training.")
                reached_max_steps = True
                break

        if not max_steps_driven:
            data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if reached_max_steps:
            # save checkpoint at max_steps if not already saved by save_steps
            already_saved = args.train.save_steps and global_step % args.train.save_steps == 0
            if not already_saved:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
                if args.train.global_rank == 0:
                    prune_checkpoints(args.train.save_checkpoint_path, args.train.max_checkpoints_to_keep)
                    if args.train.save_hf_weights:
                        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
                        model_state_dict = ckpt_to_state_dict(
                            save_checkpoint_path=save_checkpoint_path,
                            output_dir=args.train.output_dir,
                            ckpt_manager=args.train.ckpt_manager,
                        )
                        if args.train.enable_fp32:
                            save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets, save_dtype=torch.float32)
                        else:
                            save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
                        save_args(args, hf_weights_path)
                        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")
            break
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
            if args.train.global_rank == 0:
                prune_checkpoints(args.train.save_checkpoint_path, args.train.max_checkpoints_to_keep)
                if args.train.save_hf_weights and save_checkpoint_path is not None:
                    hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
                    model_state_dict = ckpt_to_state_dict(
                        save_checkpoint_path=save_checkpoint_path,
                        output_dir=args.train.output_dir,
                        ckpt_manager=args.train.ckpt_manager,
                    )
                    if args.train.enable_fp32:
                        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets, save_dtype=torch.float32)
                    else:
                        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
                    save_args(args, hf_weights_path)
                    logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    if max_steps_driven:
        data_loader_tqdm.close()
    torch.cuda.synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    # save model in huggingface's format
    if args.train.global_rank == 0:
        if args.train.save_hf_weights and save_checkpoint_path is not None:
            hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
            model_state_dict = ckpt_to_state_dict(
                save_checkpoint_path=save_checkpoint_path,
                output_dir=args.train.output_dir,
                ckpt_manager=args.train.ckpt_manager,
            )
            if args.train.enable_fp32:
                save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets, save_dtype=torch.float32)
            else:
                save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
            save_args(args, hf_weights_path)
            logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")
    if args.train.global_rank == 0 and args.train.use_wandb:
        wandb.finish()


    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
