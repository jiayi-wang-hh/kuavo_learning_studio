"""Evaluate saved linear heads on CPU; export metrics, predictions, and confusion matrices."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from progress_monitor.train_linear_probe import (
    CLASSES, DEFAULT_LABELS, DEFAULT_ROOT, classification_metrics, evaluate, load_split, validate_selection,
)


def load_head(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(checkpoint["classes"]) != CLASSES:
        raise ValueError("Checkpoint class order differs")
    mean, std = checkpoint["mean"], checkpoint["std"]
    if mean.shape != (2048,) or std.shape != (2048,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("Invalid checkpoint normalization")
    model = torch.nn.Linear(2048, 6)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().requires_grad_(False), checkpoint


def export_matrix(directory, name, metrics, plot=False):
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    with (directory / f"{name}_confusion.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/predicted", *CLASSES])
        writer.writerows([[label, *row] for label, row in zip(CLASSES, matrix.tolist())])
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(figsize=(7, 6))
        chart = axes.imshow(matrix, cmap="Blues")
        figure.colorbar(chart, ax=axes)
        axes.set(xticks=range(6), yticks=range(6), xticklabels=CLASSES, yticklabels=CLASSES,
                 xlabel="Predicted phase", ylabel="True phase", title=name)
        for i in range(6):
            for j in range(6):
                axes.text(j, i, str(matrix[i, j]), ha="center", va="center",
                          color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        figure.tight_layout()
        figure.savefig(directory / f"{name}_confusion.png", dpi=160)
        plt.close(figure)


def run(args):
    if args.batch_size < 1 or args.cpu_threads < 1:
        raise ValueError("Use positive batch-size and cpu-threads")
    torch.set_num_threads(args.cpu_threads)
    selection = json.loads(args.selection.read_text())
    validate_selection(selection, args.labels, args.scope)
    if not selection.get(args.split):
        raise ValueError(f"Selection contains no {args.split} episodes")
    if args.split == "test" and args.scope != "full":
        raise ValueError("Test evaluation requires an explicit full selection")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.plot:
        import importlib.util
        if importlib.util.find_spec("matplotlib") is None:
            raise RuntimeError("--plot requires matplotlib; omit --plot to export CSV matrices only")
    summary = {"split": args.split, "selection": selection, "classes": CLASSES, "models": {},
               "note": "Frozen saved heads; saved training normalization only. No optimizer or VLA loaded."}
    names, baseline, identity = set(), None, None
    for path in args.checkpoints:
        model, checkpoint = load_head(path)
        cfg = checkpoint["config"]
        name = f"{cfg['pooling']}_{cfg['weighting']}"
        if name in names:
            raise ValueError(f"Duplicate model {name}")
        names.add(name)
        for split in ("train", "val"):
            if sorted(cfg[f"{split}_episodes"]) != sorted(selection[split]):
                raise ValueError(f"Checkpoint {split} episodes differ from selection")
        x, y, ep, frame, provenance = load_split(args.features, args.labels, selection[args.split], args.split, cfg["pooling"])
        if provenance != checkpoint["provenance"]:
            raise ValueError("Feature/checkpoint provenance differs")
        current_identity = (y, ep, frame)
        if identity is not None:
            for actual, expected in zip(current_identity, identity):
                np.testing.assert_array_equal(actual, expected)
        identity = current_identity
        counts = checkpoint["training_class_counts"].numpy()
        current_baseline = classification_metrics(y, np.full_like(y, int(counts.argmax())))
        if baseline is not None and current_baseline != baseline:
            raise ValueError("Models have different training majority baselines")
        baseline = current_baseline
        normalized = (torch.from_numpy(x) - checkpoint["mean"]) / checkpoint["std"]
        if not torch.isfinite(normalized).all():
            raise ValueError("Nonfinite normalized features")
        metrics, predicted, probabilities = evaluate(model, normalized, torch.from_numpy(y), args.batch_size)
        metrics["per_episode"] = {str(e): classification_metrics(y[ep == e], predicted[ep == e]) for e in np.unique(ep)}
        args.output.mkdir(parents=True, exist_ok=True)
        export_matrix(args.output, name, metrics, args.plot)
        (args.output / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
        np.savez_compressed(args.output / f"{name}_predictions.npz", episode_index=ep, frame_index=frame,
                            phase=y, predicted=predicted, probabilities=probabilities)
        summary["models"][name] = {"checkpoint": str(path.resolve()), "epoch": checkpoint["epoch"], **metrics}
    summary["majority_baseline"] = baseline
    summary["ranking_by_macro_f1"] = sorted(summary["models"], key=lambda name: summary["models"][name]["macro_f1"], reverse=True)
    export_matrix(args.output, "majority_baseline", baseline, args.plot)
    lines = ["# Saved Linear Probe comparison", "", f"Evaluation split: {args.split}. No fitting performed.", "",
             "| Model | Accuracy | Macro-F1 | P2a→P3 | P3→P2a |", "|---|---:|---:|---:|---:|"]
    for name in ["majority_baseline", *summary["ranking_by_macro_f1"]]:
        m = baseline if name == "majority_baseline" else summary["models"][name]
        lines.append(f"| {name} | {m['accuracy']:.2%} | {m['macro_f1']:.2%} | "
                     f"{m['p2a_vs_p3']['P2a_to_P3_rate']:.2%} | {m['p2a_vs_p3']['P3_to_P2a_rate']:.2%} |")
    lines.extend(["", "Each model's metrics JSON includes per-class precision/recall/F1 and episode metrics.",
                  "Confusion CSV/PNG rows are true phases, columns are predictions.",
                  "Validation also selected epochs; close validation scores do not establish a reliable ranking.",
                  "For a test evaluation, freeze the validation-selected choice before inspecting test scores."])
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--features", type=Path, default=DEFAULT_ROOT / "features")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--selection", type=Path, default=DEFAULT_ROOT / "pilot_selection.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "evaluation")
    parser.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--plot", action="store_true", help="Also save confusion-matrix PNGs using matplotlib")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
