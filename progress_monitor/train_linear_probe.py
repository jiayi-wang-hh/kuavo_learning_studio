"""Train six-class linear probes on frozen, episode-separated LingBot features."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

CLASSES = ("P1", "P2", "P2a", "P2b", "P3", "P4")
DEFAULT_LABELS = Path("/media/data/jiayi/dataset/progress_training_v1_5")
DEFAULT_ROOT = Path("/media/data/jiayi/outputs/lingbot_progress_pilot")


def validate_selection(selection, labels_root, scope="pilot"):
    """Validate original episode membership; test IDs never enter a training set."""
    official = {s: set(map(int, (labels_root / f"{s}_episodes.txt").read_text().split()))
                for s in ("train", "val", "test")}
    if any(official[a] & official[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Original episode splits overlap")
    for split in official:
        ids = selection.get(split, [])
        if not isinstance(ids, list) or any(type(ep) is not int for ep in ids):
            raise ValueError(f"{split} must contain integer episode IDs")
        if len(set(ids)) != len(ids) or not set(ids) <= official[split]:
            raise ValueError(f"Invalid or duplicate {split} episodes")
        if scope == "full" and set(ids) != official[split]:
            raise ValueError(f"Full selection must contain every original {split} episode")
    if not selection.get("train") or not selection.get("val"):
        raise ValueError("Both training and validation episodes are required")
    if scope == "pilot" and (len(selection["train"]) != 12 or len(selection["val"]) != 4 or selection.get("test")):
        raise ValueError("Pilot selection requires 12 train, 4 val, and no test episodes")


def feature_rows(archive, label, pooling):
    """Explicit feature whitelist: supervision arrays are alignment checks only."""
    if pooling not in ("all", "vision", "language"):
        raise ValueError("Unknown pooling representation")
    for name in ("phase", "frame_index", "timestamp", "valid", "overlap"):
        np.testing.assert_array_equal(archive[name], label[name])
    valid = label["valid"] & ~label["overlap"] & (label["phase"] >= 0) & (label["phase"] < 6)
    np.testing.assert_array_equal(archive["feature_valid"], valid, err_msg="Requires every valid labeled frame")
    features = archive[f"features_{pooling}"]
    if features.shape != (len(valid), 2048) or features.dtype != np.float16:
        raise ValueError("Expected float16 features [T_episode,2048]")
    x = features[valid].astype(np.float32)
    if not np.isfinite(x).all() or not np.isnan(features[~valid]).all():
        raise ValueError("Expected finite extracted rows and NaN unextracted rows")
    return x, valid


def classification_metrics(y, predicted):
    y, predicted = np.asarray(y), np.asarray(predicted)
    if y.shape != predicted.shape or not len(y):
        raise ValueError("Expected equally sized, nonempty label/prediction vectors")
    if ((y < 0) | (y >= 6) | (predicted < 0) | (predicted >= 6)).any():
        raise ValueError("Labels must be in [0,6)")
    matrix = np.bincount(6 * y.astype(int) + predicted.astype(int), minlength=36).reshape(6, 6)
    tp = matrix.diagonal().astype(float)
    support = matrix.sum(1)
    precision = np.divide(tp, matrix.sum(0), out=np.zeros(6), where=matrix.sum(0) != 0)
    recall = np.divide(tp, support, out=np.zeros(6), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(6), where=(precision + recall) != 0)
    return {"accuracy": float(tp.sum() / len(y)), "macro_f1": float(f1.mean()),
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                                   "f1": float(f1[i]), "support": int(support[i])}
                          for i, name in enumerate(CLASSES)},
            "confusion_matrix": matrix.tolist(), "confusion_order": list(CLASSES),
            "p2a_vs_p3": {"P2a_to_P3_count": int(matrix[2, 4]), "P3_to_P2a_count": int(matrix[4, 2]),
                          "P2a_to_P3_rate": float(matrix[2, 4] / max(1, support[2])),
                          "P3_to_P2a_rate": float(matrix[4, 2] / max(1, support[4]))}}


def training_normalization(x):
    # float64 accumulation reduces cancellation in nearly constant dimensions.
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1., std).astype(np.float32)
    return mean, std


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def load_split(root, labels_root, ids, split, pooling):
    official = set(map(int, (labels_root / f"{split}_episodes.txt").read_text().split()))
    if len(set(ids)) != len(ids) or not set(ids) <= official:
        raise ValueError(f"Invalid or duplicate {split} episodes")
    xs, ys, eps, frames = [], [], [], []
    provenance = None
    for ep in ids:
        with np.load(root / split / f"episode_{ep:03d}.npz") as f, np.load(labels_root / "episodes" / f"episode_{ep:03d}.npz") as label:
            if int(f["episode_index"]) != ep or str(f["split"]) != split:
                raise ValueError("Feature episode/split metadata mismatch")
            x, valid = feature_rows(f, label, pooling)
            meta = json.loads(str(f["metadata_json"]))
            current = {k: meta[k] for k in ("checkpoint", "qwen", "camera_order", "preprocessing", "task_instructions")}
            if provenance is not None and current != provenance:
                raise ValueError("Mixed feature extraction provenance")
            provenance = current
            xs.append(x)
            ys.append(label["phase"][valid].astype(np.int64))
            eps.append(np.full(valid.sum(), ep, dtype=np.int64))
            frames.append(label["frame_index"][valid])
    if not xs:
        raise ValueError(f"Empty {split} selection")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(eps), np.concatenate(frames), provenance


@torch.inference_mode()
def evaluate(model, x, y, batch_size):
    logits = torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])
    loss = nn.functional.cross_entropy(logits, y).item()
    probabilities = logits.softmax(-1).cpu().numpy()
    predicted = probabilities.argmax(1)
    return dict(classification_metrics(y.cpu().numpy(), predicted), cross_entropy=loss), predicted, probabilities


def train_one(x_train, y_train, x_val, y_val, config, directory, normalization, provenance, val_episode, val_frame):
    seed_everything(config["seed"])
    directory.mkdir(parents=True, exist_ok=False)
    model = nn.Linear(2048, 6).to(x_train.device)
    counts = torch.bincount(y_train, minlength=6)
    if not torch.all(counts > 0):
        raise ValueError("All six classes must occur in training")
    weights = len(y_train) / (6 * counts.float()) if config["weighting"] == "weighted" else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    generator = torch.Generator(device="cpu").manual_seed(config["seed"])
    history, best_score, best_epoch, stale = [], -1., 0, 0
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator).to(x_train.device)
        for batch in order.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        train_metrics, _, _ = evaluate(model, x_train, y_train, config["batch_size"])
        val_metrics, _, _ = evaluate(model, x_val, y_val, config["batch_size"])
        history.append({"epoch": epoch, "train_loss": train_metrics["cross_entropy"],
                        "train_accuracy": train_metrics["accuracy"], "train_macro_f1": train_metrics["macro_f1"],
                        "val_loss": val_metrics["cross_entropy"], "val_accuracy": val_metrics["accuracy"],
                        "val_macro_f1": val_metrics["macro_f1"]})
        if best_epoch == 0 or val_metrics["macro_f1"] > best_score + config["min_delta"]:
            best_score, best_epoch, stale = val_metrics["macro_f1"], epoch, 0
            torch.save({"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "mean": torch.from_numpy(normalization[0]), "std": torch.from_numpy(normalization[1]),
                        "config": config, "classes": CLASSES, "epoch": epoch,
                        "class_weights": None if weights is None else weights.cpu(),
                        "training_class_counts": counts.cpu(), "provenance": provenance}, directory / "best.pt")
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(dict(run=directory.name, **history[-1])), flush=True)
        if stale >= config["patience"]:
            break
    checkpoint = torch.load(directory / "best.pt", map_location=x_train.device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    metrics, predicted, probabilities = evaluate(model, x_val, y_val, config["batch_size"])
    train_metrics, _, _ = evaluate(model, x_train, y_train, config["batch_size"])
    actual = y_val.cpu().numpy()
    metrics["per_episode"] = {str(ep): classification_metrics(actual[val_episode == ep], predicted[val_episode == ep])
                              for ep in np.unique(val_episode)}
    metrics["mean_episode_accuracy"] = float(np.mean([m["accuracy"] for m in metrics["per_episode"].values()]))
    metrics["mean_episode_macro_f1"] = float(np.mean([m["macro_f1"] for m in metrics["per_episode"].values()]))
    result = dict(config=config, best_epoch=best_epoch, epochs_run=len(history), validation=metrics,
                  training=train_metrics, seconds=time.perf_counter() - started)
    (directory / "metrics.json").write_text(json.dumps(result, indent=2))
    (directory / "history.json").write_text(json.dumps(history, indent=2))
    np.savez_compressed(directory / "validation_predictions.npz", episode_index=val_episode, frame_index=val_frame,
                        phase=actual, predicted=predicted, probabilities=probabilities)
    return result


def write_markdown(summary, output):
    percent = lambda value: f"{100 * value:.2f}%"
    lines = [f"# LingBot Linear Probe {summary.get('scope', 'pilot')}", "",
             f"Train episodes: {summary['selection']['train']}",
             f"Validation episodes: {summary['selection']['val']}",
             f"Frames: {summary['training_frames']} train, {summary['validation_frames']} validation. No test episodes.", "",
             f"All scores below are validation estimates. The same {len(summary['selection']['val'])} validation episodes select epochs and compare models.",
             "Labels are automatic weak labels; neighboring frames are correlated. No temporal smoothing is used.", "",
             "Episode separation is enforced for the probe. The VLA training configuration does not document an exclusion of these episodes; "
             "do not interpret this as evaluation on data unseen by VLA fine-tuning.", "",
             "| Representation / loss | Best epoch | Epochs run | Accuracy | Macro-F1 | P2a→P3 | P3→P2a |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    baseline = summary["majority_baseline"]
    lines.append(f"| Majority ({baseline['majority_class']}) | — | — | {percent(baseline['accuracy'])} | {percent(baseline['macro_f1'])} | "
                 f"{percent(baseline['p2a_vs_p3']['P2a_to_P3_rate'])} | {percent(baseline['p2a_vs_p3']['P3_to_P2a_rate'])} |")
    for name, run in summary["runs"].items():
        m = run["validation"]
        lines.append(f"| {name} | {run['best_epoch']} | {run['epochs_run']} | {percent(m['accuracy'])} | {percent(m['macro_f1'])} | "
                     f"{percent(m['p2a_vs_p3']['P2a_to_P3_rate'])} | {percent(m['p2a_vs_p3']['P3_to_P2a_rate'])} |")
    for name, m in [("Majority baseline", baseline)] + [(k, v["validation"]) for k, v in summary["runs"].items()]:
        lines.extend(["", f"## {name}", "", "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"])
        for label, row in m["per_class"].items():
            lines.append(f"| {label} | {percent(row['precision'])} | {percent(row['recall'])} | {percent(row['f1'])} | {row['support']} |")
        lines.extend(["", "Confusion matrix: rows = true class, columns = prediction.", "",
                      "| True / predicted | " + " | ".join(CLASSES) + " |", "|---|---:|---:|---:|---:|---:|---:|"])
        for label, row in zip(CLASSES, m["confusion_matrix"]):
            lines.append("| " + label + " | " + " | ".join(map(str, row)) + " |")
        if "per_episode" in m:
            lines.extend(["", "| Validation episode | Accuracy | Macro-F1 |", "|---|---:|---:|"])
            for ep, row in m["per_episode"].items():
                lines.append(f"| {ep} | {percent(row['accuracy'])} | {percent(row['macro_f1'])} |")
    extraction = summary.get("extraction")
    if extraction:
        lines.extend(["", "## Extraction", "",
                      f"{extraction['observations']} observations; total {extraction['total_seconds']:.2f} seconds "
                      f"including loading, extraction {extraction['extraction_seconds']:.2f} seconds.",
                      f"Throughput: {extraction['observations_per_second']:.2f} observations/s during extraction; "
                      f"{extraction['end_to_end_observations_per_second']:.2f} observations/s end to end.",
                      f"Peak allocated/reserved GPU memory: {extraction['gpu_peak_allocated_bytes'] / 2**30:.2f} / "
                      f"{extraction['gpu_peak_reserved_bytes'] / 2**30:.2f} GiB."])
    (output / "report.md").write_text("\n".join(lines) + "\n")


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.cpu_threads)
    if not 1 <= args.epochs <= 50 or min(args.batch_size, args.patience, args.cpu_threads) < 1:
        raise ValueError("Use 1–50 epochs and positive batch size/patience")
    if args.lr <= 0 or args.weight_decay < 0 or args.min_delta < 0:
        raise ValueError("Require positive lr and nonnegative weight-decay/min-delta")
    if len(set(args.pooling)) != len(args.pooling):
        raise ValueError("Duplicate pooling modes")
    selection = json.loads(args.selection.read_text())
    validate_selection(selection, args.labels, args.scope)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    summary = {"selection": selection, "scope": args.scope, "classes": CLASSES, "runs": {}, "seed": args.seed,
               "device": args.device, "torch_version": torch.__version__,
               "selection_metric": f"validation macro-F1; early stopping uses the same {len(selection['val'])} validation episodes",
               "note": "Validation estimates; no test data used. Frame metrics are temporally correlated."}
    if (args.features / "report.json").exists():
        summary["extraction"] = json.loads((args.features / "report.json").read_text())
    for pooling in args.pooling:
        train = load_split(args.features, args.labels, selection["train"], "train", pooling)
        val = load_split(args.features, args.labels, selection["val"], "val", pooling)
        if train[-1] != val[-1]:
            raise ValueError("Train/validation extraction provenance differs")
        mean, std = training_normalization(train[0])
        x_train = torch.from_numpy((train[0] - mean) / std).to(args.device)
        x_val = torch.from_numpy((val[0] - mean) / std).to(args.device)
        y_train, y_val = torch.from_numpy(train[1]).to(args.device), torch.from_numpy(val[1]).to(args.device)
        counts = np.bincount(train[1], minlength=6)
        majority = int(counts.argmax())
        summary["training_frames"], summary["validation_frames"] = len(y_train), len(y_val)
        summary["training_class_counts"] = counts.tolist()
        summary["validation_class_counts"] = np.bincount(val[1], minlength=6).tolist()
        summary["majority_baseline"] = dict(classification_metrics(val[1], np.full_like(val[1], majority)),
                                            majority_class=CLASSES[majority])
        choices = ("unweighted", "weighted") if args.weighting == "both" else (args.weighting,)
        for weighting in choices:
            name = f"{pooling}_{weighting}"
            config = {k: getattr(args, k) for k in ("seed", "epochs", "patience", "min_delta", "lr", "weight_decay", "batch_size")}
            config.update(pooling=pooling, weighting=weighting, train_episodes=selection["train"], val_episodes=selection["val"])
            summary["runs"][name] = train_one(x_train, y_train, x_val, y_val, config, args.output / name,
                                              (mean, std), train[-1], val[2], val[3])
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    best = max(summary["runs"], key=lambda k: summary["runs"][k]["validation"]["macro_f1"])
    summary["best_by_validation_macro_f1"] = best
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown(summary, args.output)
    print(json.dumps({"best": best, "output": str(args.output), "majority_baseline": summary["majority_baseline"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_ROOT / "features")
    parser.add_argument("--selection", type=Path, default=DEFAULT_ROOT / "pilot_selection.json")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "probes")
    parser.add_argument("--scope", choices=("pilot", "full"), default="pilot", help="Full uses every original train/val episode; test stays excluded from training")
    parser.add_argument("--pooling", nargs="+", choices=("all", "vision", "language"), default=["all", "vision", "language"])
    parser.add_argument("--weighting", choices=("both", "weighted", "unweighted"), default="both")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
