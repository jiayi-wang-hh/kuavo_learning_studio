"""Deterministic episode selection from cycle counts, without feature/evaluation access."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from progress_monitor.extract_lingbot_features import LABELS, load_labels


def select(labels=LABELS, scope="pilot"):
    splits = load_labels(labels)
    with (labels / "manifest.csv").open() as stream:
        rows = {int(r["episode_index"]): {k: int(v) for k, v in r.items()}
                for r in csv.DictReader(stream)}
    if scope == "full":
        result = {"method": "All episodes in their original splits", "scope": "full"}
        for split in ("train", "val", "test"):
            result[split] = sorted(ep for ep in splits if splits[ep] == split)
        return result
    result = {"method": "Evenly spaced episode IDs within cycle-ratio strata; no test episodes", "scope": "pilot", "strata": {}}
    for split, counts in (("train", (3, 8, 1)), ("val", (1, 2, 1))):
        groups = [[], [], []]
        for ep in sorted(splits):
            if splits[ep] != split:
                continue
            r = rows[ep]
            d = r["flip_cycles"] - r["no_flip_cycles"]
            groups[0 if d < 0 else 2 if d > 0 else 1].append(ep)
        chosen = []
        for name, group, n in zip(("no_flip_heavy", "balanced", "flip_heavy"), groups, counts):
            if len(group) < n:
                raise ValueError(f"Insufficient {split}/{name} episodes")
            # A singleton uses the central episode rather than the earliest one.
            indices = [len(group) // 2] if n == 1 else np.rint(np.linspace(0, len(group) - 1, n)).astype(int)
            ids = [group[i] for i in indices]
            result["strata"][f"{split}/{name}"] = ids
            chosen.extend(ids)
        result[split] = sorted(chosen)
    result["episodes"] = {str(ep): dict(rows[ep], split=splits[ep]) for ep in result["train"] + result["val"]}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    args = parser.parse_args()
    selection = select(args.labels, args.scope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(selection, stream, indent=2)
    print(json.dumps(selection, indent=2))
