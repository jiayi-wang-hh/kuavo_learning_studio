"""CPU-only validation of saved frozen features, labels, splits, and provenance."""
import argparse
import json
from pathlib import Path

import numpy as np

from progress_monitor.train_linear_probe import DEFAULT_LABELS, DEFAULT_ROOT, feature_rows, validate_selection


def verify(features, labels, selection, scope="pilot"):
    validate_selection(selection, labels, scope)
    totals, provenance = {}, None
    for split in ("train", "val", "test"):
        ids = selection.get(split, [])
        expected = {f"episode_{ep:03d}.npz" for ep in ids}
        actual = {p.name for p in (features / split).glob("*.npz")}
        if expected != actual:
            raise ValueError(f"{split} files differ: missing={expected - actual}, extra={actual - expected}")
        count = 0
        for ep in ids:
            with np.load(features / split / f"episode_{ep:03d}.npz") as archive, np.load(labels / "episodes" / f"episode_{ep:03d}.npz") as label:
                if int(archive["episode_index"]) != ep or str(archive["split"]) != split:
                    raise ValueError("Episode/split metadata mismatch")
                for mode in ("all", "vision", "language"):
                    _, mask = feature_rows(archive, label, mode)
                count += int(mask.sum())
                meta = json.loads(str(archive["metadata_json"]))
                current = {k: meta[k] for k in ("checkpoint", "qwen", "camera_order", "preprocessing", "task_instructions")}
                if provenance is not None and current != provenance:
                    raise ValueError("Feature provenance differs across episodes")
                provenance = current
        totals[split] = {"episodes": len(ids), "valid_frames": count}
    return {"status": "passed", "shape": "[T_episode,2048] per pooling", "dtype": "float16",
            "splits": totals, "provenance": provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_ROOT / "features")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--selection", type=Path, default=DEFAULT_ROOT / "pilot_selection.json")
    parser.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    args = parser.parse_args()
    print(json.dumps(verify(args.features, args.labels, json.loads(args.selection.read_text()), args.scope), indent=2))


if __name__ == "__main__":
    main()
