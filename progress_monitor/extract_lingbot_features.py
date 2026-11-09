"""Frozen LingBot prefix extraction. No robot interfaces or action generation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LINGBOT = REPO / "kuavo_model/external_models/lingbot-vla"
CHECKPOINT = Path("/media/data/jiayi/outputs/kuavo_lingbotvla/task3_pacakge_lora/global_step_66480/hf_ckpt")
DATASET = Path("/media/data/training_data/task3_package_pick_place_pack_flip_lerobot/lerobot")
LABELS = Path("/media/data/jiayi/dataset/progress_training_v1_5")
OUTPUT = Path("/media/data/jiayi/outputs/lingbot_progress_features")
QWEN = Path("/media/data/jiayi/hf_model/Qwen2.5-VL-3B-Instruct")


def pool_prefix(hidden, pad, vision_tokens, mode):
    """Pool final normalized hidden states; padding never contributes."""
    import torch

    mask = pad.bool().clone()
    if mode == "vision":
        mask[:, vision_tokens:] = False
    elif mode == "language":
        mask[:, :vision_tokens] = False
    elif mode != "all":
        raise ValueError(mode)
    if not torch.all(mask.any(dim=1)):
        raise ValueError(f"No valid tokens for {mode} pooling")
    return hidden.float().masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1, keepdim=True)


def load_labels(root):
    splits = {}
    for split in ("train", "val", "test"):
        for value in (root / f"{split}_episodes.txt").read_text().split():
            ep = int(value)
            if ep in splits:
                raise ValueError(f"Episode {ep} occurs in multiple splits")
            splits[ep] = split
    return splits


def read_episode(dataset_root, labels_root, ep):
    import pyarrow.parquet as pq

    with np.load(labels_root / "episodes" / f"episode_{ep:03d}.npz") as archive:
        labels = {k: archive[k].copy() for k in archive.files}
    info = json.loads((dataset_root / "meta/info.json").read_text())
    metadata = pq.read_table(dataset_root / "meta/episodes", filters=[("episode_index", "=", ep)]).to_pylist()
    if len(metadata) != 1:
        raise ValueError(f"Expected one metadata row for episode {ep}")
    meta = metadata[0]
    data_path = dataset_root / info["data_path"].format(
        chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"])
    rows = pq.read_table(data_path, columns=["episode_index", "frame_index", "timestamp", "observation.state", "task_index"],
                         filters=[("episode_index", "=", ep)]).to_pydict()
    order = np.argsort(rows["frame_index"])
    rows = {k: np.asarray(v)[order] for k, v in rows.items()}
    np.testing.assert_array_equal(rows["frame_index"], labels["frame_index"])
    np.testing.assert_allclose(rows["timestamp"], labels["timestamp"], atol=1e-4)
    np.testing.assert_allclose(rows["observation.state"], labels["state"], atol=1e-6)
    tasks = pq.read_table(dataset_root / "meta/tasks.parquet").to_pylist()
    task_map = {r["task_index"]: r["__index_level_0__"] for r in tasks}
    valid = labels["valid"] & ~labels["overlap"] & (labels["phase"] >= 0) & (labels["phase"] < 6)
    return labels, rows, meta, info, task_map, np.flatnonzero(valid)


def load_frozen(args):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: frozen model loading and GPU smoke extraction cannot run")
    sys.path.insert(0, str(LINGBOT))
    os.environ["QWEN25_PATH"] = str(args.qwen)
    from deploy.lingbot_vla_policy import LingbotVLAServer
    from lingbotvla.data.vla_data.utils import FeatureTransform

    # Reuse the deployment loader only, without constructing a server/adapter.
    holder = SimpleNamespace(use_compile=False, robot_norm_path=str(LINGBOT / "assets/norm_stats/kuavo_package.json"))
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        policy = LingbotVLAServer.load_vla(holder, str(args.checkpoint))
    finally:
        torch.set_default_dtype(previous_dtype)
    policy = policy.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    transform = FeatureTransform(str(LINGBOT / "configs/robot_configs/kuavo.yaml"), holder.data_config,
                                 holder.language_tokenizer, holder.processor.image_processor,
                                 chunk_size=holder.config.chunk_size, norm_stats_path=holder.robot_norm_path)
    return policy, transform, holder


def extract_batch(policy, batch, modes):
    import torch
    from lingbotvla.models.vla.pi0.utils import make_att_2d_masks

    with torch.inference_mode():
        m = policy.model
        # Upstream caches vision grid metadata, including the image batch size.
        # Rebuild it for short final batches or a change in extraction batch size.
        vision = m.qwenvl_with_expert
        grid_shape = tuple(batch["images"].shape)
        if getattr(vision, "_progress_grid_shape", None) != grid_shape:
            for name in ("rotary_pos_emb", "window_index", "cu_window_seqlens", "cu_seqlens"):
                setattr(vision, name, None)
            vision._progress_grid_shape = grid_shape
        prefix, pad, blocks = m.embed_prefix(batch["images"], batch["img_masks"],
                                             batch["lang_tokens"], batch["lang_masks"], vlm_causal=False)
        outputs, _ = m.qwenvl_with_expert.forward(
            attention_mask=make_att_2d_masks(pad, blocks), position_ids=pad.long().cumsum(1) - 1,
            past_key_values=None, inputs_embeds=[prefix, None], use_cache=False,
            fill_kv_cache=False, norm_qkv=False)
        hidden = outputs[0]
        vision_tokens = prefix.shape[1] - batch["lang_tokens"].shape[1]
        return {mode: pool_prefix(hidden, pad, vision_tokens, mode).cpu().numpy().astype(np.float16)
                for mode in modes}


def run(args):
    import torch

    started = time.perf_counter()
    torch.set_num_threads(args.cpu_threads)
    splits = load_labels(args.labels)
    episodes = [0] if args.smoke else (args.episodes if args.episodes is not None else sorted(splits))
    selection = None
    if args.selection:
        if args.smoke or args.episodes:
            raise ValueError("selection cannot be combined with smoke/episodes")
        selection = json.loads(args.selection.read_text())
        episodes = selection["train"] + selection["val"]
        for split in ("train", "val"):
            if any(splits.get(ep) != split for ep in selection[split]):
                raise ValueError("Selection does not match original train/val splits")
    if len(set(episodes)) != len(episodes):
        raise ValueError("Duplicate episodes")
    prepared = []
    for ep in episodes:
        if ep not in splits:
            raise ValueError(f"Episode {ep} absent from label splits")
        data = read_episode(args.dataset, args.labels, ep)
        selected = data[-1][:20] if args.smoke else data[-1]
        prepared.append((ep, data, selected))
        print(json.dumps({"episode": ep, "split": splits[ep], "selected_valid_frames": len(selected),
                          "label_alignment": "passed"}), flush=True)
    if args.validate_only:
        return
    if args.output.resolve().is_relative_to(REPO.resolve()):
        raise ValueError("Feature outputs must be outside the Git repository")
    load_started = time.perf_counter()
    policy, transform, holder = load_frozen(args)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    from lerobot.datasets.video_utils import decode_video_frames
    from torchvision.transforms.v2 import Resize

    resize = Resize((224, 224))
    report = {"checkpoint": str(args.checkpoint.resolve()), "qwen": str(args.qwen.resolve()),
              "pooling": args.pooling, "dimension": 2048, "dtype": "float16", "smoke": args.smoke,
              "loading": "strict=True passed", "episodes": [], "torch_version": torch.__version__,
              "batch_size": args.batch_size, "cpu_threads": args.cpu_threads,
              "model_load_seconds": load_seconds, "gpu": torch.cuda.get_device_name(),
              "selection": selection}
    torch.cuda.reset_peak_memory_stats()
    extraction_started = time.perf_counter()
    run_name = "smoke_episode_000_20" if args.smoke else args.run_name
    for ep, data, selected in prepared:
        episode_started = time.perf_counter()
        labels, rows, meta, info, tasks, _ = data
        target = args.output / run_name / splits[ep]
        target.mkdir(parents=True, exist_ok=True)
        destination = target / f"episode_{ep:03d}.npz"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        arrays = {f"features_{mode}": np.full((len(labels["phase"]), 2048), np.nan, np.float16)
                  for mode in args.pooling}
        feature_valid = np.zeros(len(labels["phase"]), dtype=bool)
        for start in range(0, len(selected), args.batch_size):
            indices = selected[start:start + args.batch_size]
            cameras = {}
            for key in transform.feature_config.images:
                prefix = f"videos/{key}"
                video = args.dataset / info["video_path"].format(video_key=key,
                    chunk_index=meta[f"{prefix}/chunk_index"], file_index=meta[f"{prefix}/file_index"])
                timestamps = (rows["timestamp"][indices] + meta[f"{prefix}/from_timestamp"]).tolist()
                cameras[key] = resize(decode_video_frames(video, timestamps, tolerance_s=1 / info["fps"] - 1e-4,
                                                         backend="torchcodec"))
            items = []
            for j, i in enumerate(indices):
                observation = {k: v[j] for k, v in cameras.items()}
                observation.update({"observation.state": torch.tensor(rows["observation.state"][i], dtype=torch.float32),
                                    "task": tasks[int(rows["task_index"][i])],
                                    "action": torch.zeros(holder.config.chunk_size, 16),
                                    "action_is_pad": torch.zeros(holder.config.chunk_size, dtype=torch.bool)})
                # Dummy actions satisfy existing preprocessing only; never enter the model.
                items.append(transform.apply(observation))
            batch = {k: torch.stack([item[k] for item in items]).cuda()
                     for k in ("images", "img_masks", "lang_tokens", "lang_masks")}
            batch["images"] = batch["images"].to(torch.bfloat16)
            features = extract_batch(policy, batch, args.pooling)
            for mode, values in features.items():
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite extracted features")
                arrays[f"features_{mode}"][indices] = values
            feature_valid[indices] = True
            if start % (args.batch_size * 25) == 0:
                print(json.dumps({"episode": ep, "done": int(feature_valid.sum()), "total": len(selected)}), flush=True)
        metadata = dict(report, episode=ep, split=splits[ep], camera_order=transform.feature_config.images,
                        task_instructions=list(tasks.values()), preprocessing="VLADataset Resize + FeatureTransform.apply")
        payload = {k: labels[k] for k in ("frame_index", "timestamp", "phase", "valid", "overlap", "cycle", "progress")}
        # Exclusive creation avoids replacing existing caches.
        with destination.open("xb") as stream:
            np.savez_compressed(stream, **arrays, **payload, feature_valid=feature_valid,
                                episode_index=np.int64(ep), split=np.asarray(splits[ep]), metadata_json=json.dumps(metadata))
        report["episodes"].append({"episode": ep, "frames": int(feature_valid.sum()), "path": str(destination),
                                   "seconds": time.perf_counter() - episode_started})
        print(json.dumps(report["episodes"][-1]), flush=True)
    report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    torch.cuda.synchronize()
    report["extraction_seconds"] = time.perf_counter() - extraction_started
    report["total_seconds"] = time.perf_counter() - started
    report["observations"] = sum(item["frames"] for item in report["episodes"])
    report["observations_per_second"] = report["observations"] / report["extraction_seconds"]
    report["end_to_end_observations_per_second"] = report["observations"] / report["total_seconds"]
    report_path = args.output / run_name / "report.json"
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("checkpoint", CHECKPOINT), ("qwen", QWEN), ("dataset", DATASET),
                          ("labels", LABELS), ("output", OUTPUT)):
        parser.add_argument(f"--{name}", type=Path, default=default)
    parser.add_argument("--pooling", nargs="+", choices=("all", "vision", "language"), default=["all", "vision", "language"])
    parser.add_argument("--episodes", nargs="+", type=int)
    parser.add_argument("--selection", type=Path, help="Pilot JSON containing original train/val episode IDs")
    parser.add_argument("--run-name", default="full", help="Output subdirectory; pilot runs should use pilot_v1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="Episode 000, first 20 valid frames only")
    parser.add_argument("--validate-only", action="store_true", help="Check labels/splits/data without loading model")
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("batch-size and cpu-threads must be positive")
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("run-name must be a single directory name")
    run(args)


if __name__ == "__main__":
    main()
