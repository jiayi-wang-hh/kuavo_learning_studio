"""Offline Grounding DINO package-candidate regression runner.

Example (lingbotvla environment):
    python -m recovery_harness.candidate_regression \
      --manifest recovery_harness/outputs/candidate_pipeline_regression/annotations.json \
      --model-path /path/to/local/grounding-dino-snapshot --device cpu

This never creates a ROS node or loads VLA/SAM2/robot policies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from .detection import (GroundingDinoPromptRunner, box_iou, fraction_covered,
                        filter_package_candidate_groups)


PACKAGE_LABELS = ("blue mailer bag", "pink padded envelope")
JOINT_PROMPT = "blue mailer bag. pink padded envelope. yellow tray."
CLASS_LABELS = (*PACKAGE_LABELS, "yellow tray", "robot arm", "robot gripper")


def _legacy_class(phrase: str) -> str | None:
    label = str(phrase).lower().strip(" .")
    if label in CLASS_LABELS:
        return label
    matches = [candidate for candidate in CLASS_LABELS if label and label in candidate.split()]
    return matches[0] if len(matches) == 1 else None


def _legacy_replay(query: dict[str, Any], arm_candidates: list[dict[str, Any]], *,
                   width: int, height: int, box_threshold: float = .20,
                   min_area: float = .004, max_area: float = .15,
                   min_aspect: float = .35, max_aspect: float = 2.8,
                   nms_iou: float = .50, arm_overlap: float = .35,
                   max_instances: int = 6) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reproduce current phrase mapping, threshold, size, NMS, then arm gate."""
    rejected, package_items = [], []
    static = {"yellow tray"}
    arm_labels = {label for label in CLASS_LABELS if "arm" in label or "gripper" in label}
    for candidate in query.get("postprocess_candidates", []):
        phrase = candidate.get("raw_phrase", "")
        label = _legacy_class(phrase)
        if label is None:
            rejected.append({**candidate, "stage": "phrase_matching", "decision": "rejected",
                             "reason": "not an exact label or unique-token match"})
            continue
        if label in arm_labels or label not in PACKAGE_LABELS:
            rejected.append({**candidate, "stage": "class_filter", "decision": "rejected",
                             "reason": "arm context or non-package class"})
            continue
        score = float(candidate.get("confidence") or 0.0)
        if score < box_threshold:
            rejected.append({**candidate, "stage": "package_confidence", "decision": "rejected",
                             "reason": f"{score:.6f} below {box_threshold:.6f}"})
            continue
        x1, y1, x2, y2 = candidate["box"]
        width_box, height_box = max(0, x2 - x1), max(0, y2 - y1)
        area_ratio = width_box * height_box / max(1, width * height)
        aspect = width_box / max(1, height_box)
        if not min_area <= area_ratio <= max_area:
            rejected.append({**candidate, "stage": "area_filter", "decision": "rejected",
                             "reason": f"area ratio {area_ratio:.6f}"})
            continue
        if not min_aspect <= aspect <= max_aspect:
            rejected.append({**candidate, "stage": "aspect_filter", "decision": "rejected",
                             "reason": f"aspect {aspect:.6f}"})
            continue
        package_items.append({**candidate, "box": list(candidate["box"]), "class": label,
                              "confidence": score, "ambiguous": False})

    nms_kept = []
    for label in PACKAGE_LABELS:
        kept = []
        for candidate in sorted((item for item in package_items if item["class"] == label),
                                key=lambda item: item["confidence"], reverse=True):
            suppressor = next((prior for prior in kept
                               if box_iou(prior["box"], candidate["box"]) >= nms_iou), None)
            if suppressor:
                rejected.append({**candidate, "stage": "nms", "decision": "rejected",
                    "reason": f"same-class IoU {box_iou(suppressor['box'], candidate['box']):.6f}",
                    "suppressed_by_box": suppressor["box"]})
            else:
                kept.append(candidate)
        nms_kept.extend(kept)

    after_arm = []
    for candidate in nms_kept:
        evidence = [{"arm_box": arm["box"], "arm_score": arm["confidence"],
                     "arm_phrase": arm.get("raw_phrase"),
                     "overlap_fraction": fraction_covered(candidate["box"], arm["box"])}
                    for arm in arm_candidates]
        conflict = max(evidence, key=lambda item: item["overlap_fraction"], default=None)
        if conflict and conflict["overlap_fraction"] >= arm_overlap:
            rejected.append({**candidate, "stage": "arm_overlap", "decision": "rejected",
                "reason": f"box-only overlap {conflict['overlap_fraction']:.6f} >= {arm_overlap:.6f}",
                "arm_conflict": conflict})
        else:
            candidate["arm_conflict"] = conflict
            after_arm.append(candidate)
    final = sorted(after_arm, key=lambda item: item["confidence"], reverse=True)[:max_instances]
    for item in after_arm[max_instances:]:
        rejected.append({**item, "stage": "instance_cap", "decision": "rejected",
                         "reason": f"top-{max_instances} limit"})
    return final, rejected


def _metrics(candidates: list[dict[str, Any]], ground_truth: list[dict[str, Any]],
             rejected: list[dict[str, Any]] = ()) -> dict[str, Any]:
    pairs = []
    for pi, pred in enumerate(candidates):
        for gi, gt in enumerate(ground_truth):
            overlap = box_iou(pred["box"], gt["box"])
            if overlap >= .5:
                pairs.append((overlap, pi, gi, pred.get("class", pred.get("label"))))
    pairs.sort(reverse=True)
    matched_p, matched_g = set(), set()
    for overlap, pi, gi, pred_label in pairs:
        if pi in matched_p or gi in matched_g:
            continue
        matched_p.add(pi)
        matched_g.add(gi)
    true_positives = 0
    class_pairs = []
    for pi, pred in enumerate(candidates):
        for gi, gt in enumerate(ground_truth):
            overlap = box_iou(pred["box"], gt["box"])
            label = pred.get("class", pred.get("label"))
            if overlap >= .5 and label == gt["label"] and not pred.get("ambiguous", False):
                class_pairs.append((overlap, pi, gi))
    used_p, used_g = set(), set()
    for overlap, pi, gi in sorted(class_pairs, reverse=True):
        if pi not in used_p and gi not in used_g:
            used_p.add(pi); used_g.add(gi); true_positives += 1
    resolved = [item for item in candidates if item.get("label", item.get("class")) in PACKAGE_LABELS
                and not item.get("ambiguous", False)]
    ambiguous = [item for item in candidates if item.get("ambiguous", False)]
    duplicate_pairs = []
    for left_index, left in enumerate(candidates):
        left_label = left.get("class", left.get("label"))
        for right_index in range(left_index + 1, len(candidates)):
            right = candidates[right_index]
            right_label = right.get("class", right.get("label"))
            overlap = box_iou(left["box"], right["box"])
            if left_label == right_label and overlap >= 0.50:
                duplicate_pairs.append((left_index, right_index, overlap))
    # Also count the per-query boxes merged into one cross-prompt object.
    merged_duplicates = sum(max(0, int(item.get("duplicate_count", 1)) - 1)
                            for item in candidates)
    duplicates = len(duplicate_pairs) + merged_duplicates
    arm_mis_kills = 0
    for item in rejected:
        if item.get("stage") != "arm_overlap":
            continue
        label = item.get("class", item.get("label"))
        if any(label == gt["label"] and box_iou(item["box"], gt["box"]) >= .50
               for gt in ground_truth):
            arm_mis_kills += 1
    arm_ambiguous = sum(bool(item.get("arm_conflict", {}).get("ambiguous"))
                        for item in candidates)
    recovery_gt = {index for index, gt in enumerate(ground_truth)
                   if gt.get("role") == "recovery_target"}
    return {"ground_truth_count": len(ground_truth),
            "recovery_ground_truth_count": len(recovery_gt),
            "candidate_count": len(candidates),
            "resolved_candidate_count": len(resolved),
            "true_positives": true_positives,
            "package_recall": true_positives / len(ground_truth) if ground_truth else None,
            "recovery_package_recall": len(used_g.intersection(recovery_gt)) / len(recovery_gt) if recovery_gt else None,
            "package_candidate_precision": true_positives / len(resolved) if resolved else None,
            "localization_recall_including_ambiguous": len(matched_g) / len(ground_truth) if ground_truth else None,
            "recovery_target_localization_recall": len(matched_g.intersection(recovery_gt)) / len(recovery_gt) if recovery_gt else None,
            "unmatched_resolved_false_positives": len(resolved) - true_positives,
            "ambiguous_candidate_count": len(ambiguous),
            "duplicate_detections": duplicates,
            "duplicate_pairs": duplicate_pairs,
            "arm_overlap_false_kills": arm_mis_kills,
            "arm_conflict_ambiguous_candidates": arm_ambiguous,
            "matched_ground_truth_indices": sorted(used_g)}


def _overlay(image_path: Path, output_path: Path, candidates, ground_truth, raw_queries=(), arms=()):
    from PIL import Image, ImageDraw
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for query in raw_queries:
        for candidate in query.get("raw_candidates", []):
            box = candidate["box"]
            draw.rectangle(tuple(box), outline=(100, 100, 100), width=1)
        for candidate in query.get("postprocess_candidates", []):
            draw.rectangle(tuple(candidate["box"]), outline=(255, 180, 0), width=2)
    for arm in arms:
        draw.rectangle(tuple(arm["box"]), outline=(0, 190, 255), width=3)
    for gt in ground_truth:
        draw.rectangle(tuple(gt["box"]), outline=(255, 0, 255), width=3)
    for candidate in candidates:
        box = candidate["box"]
        color = (255, 160, 0) if candidate.get("ambiguous") else (0, 220, 40)
        draw.rectangle(tuple(box), outline=color, width=3)
        label = candidate.get("class", candidate.get("label", "package_candidate"))
        draw.text((box[0], max(0, box[1] - 14)), f"{label} {candidate.get('confidence', 0):.2f}",
                  fill="white", stroke_width=2, stroke_fill="black")
    image.save(output_path)


def _recovery_target_crop(image_path: Path, output_path: Path, target_box,
                          candidates, rejected, arm_candidates):
    """Readable, target-centered decision overlay; never used for inference."""
    from PIL import Image, ImageDraw
    image = Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = map(float, target_box)
    pad_x, pad_y = max(24, (x2 - x1) * .35), max(24, (y2 - y1) * .25)
    crop_box = (max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
                min(image.width, int(x2 + pad_x)), min(image.height, int(y2 + pad_y)))
    crop = image.crop(crop_box)
    draw = ImageDraw.Draw(crop)
    ox, oy = crop_box[:2]
    def local(box):
        return (float(box[0]) - ox, float(box[1]) - oy,
                float(box[2]) - ox, float(box[3]) - oy)
    gt_local = local(target_box)
    draw.rectangle(gt_local, outline=(255, 0, 220), width=4)
    draw.text((gt_local[0], max(0, gt_local[1] - 15)), "manual recovery target",
              fill="white", stroke_width=2, stroke_fill="black")
    for item in rejected:
        box = item.get("box")
        if not box or box_iou(box, target_box) < .05:
            continue
        decision = item.get("stage_decisions", [{}])[-1]
        color = (255, 50, 50) if decision.get("decision") == "rejected" else (255, 180, 0)
        draw.rectangle(local(box), outline=color, width=2)
    for arm in arm_candidates:
        if box_iou(arm["box"], target_box) >= .05:
            draw.rectangle(local(arm["box"]), outline=(0, 190, 255), width=3)
    for item in candidates:
        box = item["box"]
        if box_iou(box, target_box) < .05:
            continue
        color = (255, 160, 0) if item.get("ambiguous") else (0, 230, 40)
        local_box = local(box)
        draw.rectangle(local_box, outline=color, width=4)
        draw.text((local_box[0], max(0, local_box[1] - 15)),
                  f"{item.get('class', item.get('label'))} {item.get('confidence', 0):.2f}",
                  fill="white", stroke_width=2, stroke_fill="black")
    crop.save(output_path)


def finalize_regression_frames(frames: list[dict[str, Any]], *, device: str,
                              model_path: str, annotation_note: str | None,
                              output_dir: str | Path) -> dict[str, Any]:
    """Recompute summary metrics from saved frame-level candidate evidence."""
    output = Path(output_dir)
    runs_by_mode = {mode: [] for mode in ("legacy_joint", "per_class", "generic")}
    for frame in frames:
        def annotate_iou(item):
            box = item.get("box")
            if not box:
                return
            overlaps = [(box_iou(box, gt["box"]), index, gt)
                        for index, gt in enumerate(frame["ground_truth"])]
            best_iou, best_index, best_gt = max(overlaps, default=(0.0, None, None),
                                                key=lambda row: row[0])
            item["best_ground_truth_iou"] = best_iou
            item["best_ground_truth_match"] = (None if best_gt is None else {
                "ground_truth_index": best_index, "label": best_gt["label"],
                "role": best_gt.get("role"), "iou": best_iou})
        for mode, run in frame["modes"].items():
            diagnostics = run.get("full_diagnostics", {})
            diagnostics.setdefault("roi_filtering", {"applied": False,
                "reason": "DINO proposals are not clipped or rejected by the calibrated platform ROI; ROI relation remains downstream perception evidence."})
            for query in diagnostics.get("queries", []):
                for field in ("raw_candidates", "box_threshold_candidates",
                              "text_threshold_candidates", "postprocess_candidates"):
                    for item in query.get(field, []):
                        annotate_iou(item)
            for item in diagnostics.get("arm_candidates", []):
                annotate_iou(item)
            if mode != "legacy_joint":
                query_candidates = [candidate for query in diagnostics["queries"]
                                    for candidate in query["postprocess_candidates"]]
                width, height = diagnostics["frame_size"]
                accepted, stage_audit = filter_package_candidate_groups(
                    query_candidates, diagnostics["arm_candidates"],
                    package_labels=PACKAGE_LABELS, frame_width=width,
                    frame_height=height, box_threshold=.20,
                    min_area_ratio=.004, max_area_ratio=.15,
                    min_aspect=.35, max_aspect=2.8, nms_iou=.50,
                    max_instances=6, reliable_arm_min_score=.30)
                run["candidates"] = [{**item,
                    "class": item.get("label") or "package_candidate",
                    "confidence": float(item.get("confidence") or 0.0),
                    "box": tuple(map(float, item["box"]))}
                    for item in accepted]
                run["rejected_candidates"] = [item for item in stage_audit
                    if item.get("candidate_validity") == "rejected"]
                _overlay(Path(frame["image"]), Path(run["overlay"]), run["candidates"],
                    frame["ground_truth"], raw_queries=diagnostics["queries"],
                    arms=diagnostics["arm_candidates"])
            run["metrics"] = _metrics(run["candidates"], frame["ground_truth"],
                                      run.get("rejected_candidates", []))
            for item in run["candidates"] + run.get("rejected_candidates", []):
                annotate_iou(item)
            for target in (item for item in frame["ground_truth"]
                           if item.get("role") == "recovery_target"):
                target_overlay = Path(run["overlay"]).with_name(
                    Path(run["overlay"]).stem + "_recovery_target.png")
                _recovery_target_crop(Path(frame["image"]), target_overlay,
                    target["box"], run["candidates"], run.get("rejected_candidates", []),
                    run.get("arm_candidates", []))
                run.setdefault("recovery_target_overlays", []).append(str(target_overlay))
            runs_by_mode[mode].append(run)
    aggregate = {}
    for mode, runs in runs_by_mode.items():
        gt_count = sum(run["metrics"]["ground_truth_count"] for run in runs)
        tp = sum(run["metrics"]["true_positives"] for run in runs)
        resolved = sum(run["metrics"]["resolved_candidate_count"] for run in runs)
        recovery_gt_count = sum(run["metrics"]["recovery_ground_truth_count"] for run in runs)
        aggregate[mode] = {
            "frame_count": len(runs), "ground_truth_count": gt_count,
            "recovery_ground_truth_count": recovery_gt_count,
            "true_positives": tp,
            "package_recall": tp / gt_count if gt_count else None,
            "recovery_package_recall": (
                sum((run["metrics"]["recovery_package_recall"] or 0.0) *
                    run["metrics"]["recovery_ground_truth_count"] for run in runs) / recovery_gt_count
                if recovery_gt_count else None),
            "recovery_target_localization_recall": (
                sum((run["metrics"]["recovery_target_localization_recall"] or 0.0) *
                    run["metrics"]["recovery_ground_truth_count"] for run in runs) / recovery_gt_count
                if recovery_gt_count else None),
            "resolved_candidates": resolved,
            "package_candidate_precision": tp / resolved if resolved else None,
            "localization_recall_including_ambiguous": (
                sum(run["metrics"]["localization_recall_including_ambiguous"] *
                    run["metrics"]["ground_truth_count"] for run in runs
                    if run["metrics"]["localization_recall_including_ambiguous"] is not None) / gt_count
                if gt_count else None),
            "unmatched_resolved_false_positives": sum(run["metrics"]["unmatched_resolved_false_positives"] for run in runs),
            "ambiguous_candidates": sum(run["metrics"]["ambiguous_candidate_count"] for run in runs),
            "duplicate_detections": sum(run["metrics"]["duplicate_detections"] for run in runs),
            "arm_overlap_false_kills": sum(run["metrics"]["arm_overlap_false_kills"] for run in runs),
            "arm_conflict_ambiguous_candidates": sum(run["metrics"]["arm_conflict_ambiguous_candidates"] for run in runs),
            "normal_scene_false_positives": sum(
                run["metrics"]["unmatched_resolved_false_positives"]
                for frame in frames for run in [frame["modes"][mode]]
                if frame["scene"] == "normal_sorting"),
            "mean_detection_seconds": sum(run["total_detection_seconds"] for run in runs) / len(runs) if runs else None,
        }
    baseline = [run["total_detection_seconds"] for run in runs_by_mode["legacy_joint"]]
    per_class = [run["total_detection_seconds"] for run in runs_by_mode["per_class"]]
    aggregate["per_class"]["mean_extra_seconds_vs_joint"] = (
        sum(new - old for old, new in zip(baseline, per_class)) / len(baseline) if baseline else None)
    report = {"device": device, "model_path": model_path,
        "package_labels": list(PACKAGE_LABELS), "box_threshold": .20,
        "text_threshold": .10, "arm_box_threshold": .15,
        "arm_reliable_min_score": .30, "annotation_note": annotation_note,
        "aggregate": aggregate, "frames": frames}
    (output / "candidate_regression_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Package candidate pipeline regression", "",
        "Manually annotated offline frames; no robot/VLA/SAM2 execution. Generic prompt is diagnostic only.", "",
        "| Mode | Recovery recall | Recovery localization incl. ambiguous | All-package recall | Precision | Normal FPs | Ambiguous | Duplicates | Arm false kills | Mean time (s) | Extra vs joint (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in ("legacy_joint", "per_class", "generic"):
        a = aggregate[mode]
        def fmt(value): return "—" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)
        lines.append(f"| {mode} | {fmt(a['recovery_package_recall'])} | {fmt(a['recovery_target_localization_recall'])} | {fmt(a['package_recall'])} | {fmt(a['package_candidate_precision'])} | {a['normal_scene_false_positives']} | {a['ambiguous_candidates']} | {a['duplicate_detections']} | {a['arm_overlap_false_kills']} | {fmt(a['mean_detection_seconds'])} | {fmt(a.get('mean_extra_seconds_vs_joint'))} |")
    lines += ["", "Recovery recall requires a resolved class-correct candidate with IoU ≥ 0.50 against a GT item tagged `recovery_target`; recovery localization also counts ambiguous candidates. All-package recall uses all 17 annotated package instances. Precision is class-correct resolved matches divided by resolved candidates; normal FPs are unmatched resolved candidates in the two normal frames. Duplicate count is cross-prompt boxes merged plus any remaining same-class overlaps.",
        "Per-frame candidate boxes, raw threshold-zero proposals, phrase/score/IoU records, stage rejections, prompts, timing, and overlays are in `candidate_regression_results.json` and `overlays/`.",
        "The annotations are a small manual sample; interpret precision/recall as regression evidence for these frames, not a general benchmark. CPU timings include package prompt(s) and the separate arm query; no ROI candidate filter was applied."]
    (output / "candidate_regression_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_regression(manifest_path: str | Path, *, model_path: str, device: str = "cpu",
                   output_dir: str | Path | None = None) -> dict[str, Any]:
    import cv2
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = Path(output_dir) if output_dir else manifest_path.parent / "results"
    output.mkdir(parents=True, exist_ok=True)
    overlays = output / "overlays"
    overlays.mkdir(exist_ok=True)
    runner = GroundingDinoPromptRunner(model_path, device=device,
        package_labels=PACKAGE_LABELS, joint_prompt=JOINT_PROMPT)
    results, runs_by_mode = [], {"legacy_joint": [], "per_class": [], "generic": []}

    for frame_index, row in enumerate(manifest["frames"]):
        image_path = Path(row["image"])
        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read regression image: {image_path}")
        height, width = bgr.shape[:2]
        ground_truth = row.get("ground_truth", [])
        frame_result = {"scene": row["scene"], "image": str(image_path),
                        "ground_truth": ground_truth, "modes": {}}
        for mode in ("legacy_joint", "per_class", "generic"):
            result = runner.detect(bgr, query_mode=mode)
            diagnostics = result["diagnostics"]
            arm_candidates = result["arm_candidates"]
            if mode == "legacy_joint":
                candidates, rejected = _legacy_replay(
                    diagnostics["queries"][0], arm_candidates,
                    width=width, height=height)
            else:
                candidates = result["detections"]
                rejected = [item for item in diagnostics["filter_audit"]
                            if item.get("candidate_validity") == "rejected"]
            metrics = _metrics(candidates, ground_truth, rejected)
            mode_result = {"mode": mode,
                "prompt_texts": [q["prompt"] for q in diagnostics["queries"]],
                "query_timings_seconds": [q["elapsed_seconds"] for q in diagnostics["queries"]],
                "total_detection_seconds": diagnostics["total_detection_seconds"],
                "arm_query_seconds": diagnostics["arm_query"]["elapsed_seconds"],
                "raw_candidate_count": sum(len(q["raw_candidates"]) for q in diagnostics["queries"]),
                "text_threshold_candidate_count": sum(len(q["postprocess_candidates"]) for q in diagnostics["queries"]),
                "arm_candidates": arm_candidates,
                "candidates": candidates,
                "rejected_candidates": rejected,
                "metrics": metrics,
                "full_diagnostics": diagnostics}
            overlay_path = overlays / f"{frame_index:02d}_{mode}.png"
            _overlay(image_path, overlay_path, candidates, ground_truth,
                     raw_queries=diagnostics["queries"], arms=arm_candidates)
            mode_result["overlay"] = str(overlay_path)
            frame_result["modes"][mode] = mode_result
            runs_by_mode[mode].append(mode_result)
        results.append(frame_result)
        (output / "candidate_regression_frames.partial.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    return finalize_regression_frames(results, device=device, model_path=model_path,
        annotation_note=manifest.get("annotation_note"), output_dir=output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir")
    parser.add_argument("--finalize-existing", action="store_true",
                         help="Summarize candidate_regression_frames.partial.json without rerunning DINO")
    args = parser.parse_args()
    if args.finalize_existing:
        output = Path(args.output_dir or Path(args.manifest).resolve().parent / "results")
        partial_path = output / "candidate_regression_frames.partial.json"
        results_path = output / "candidate_regression_results.json"
        if partial_path.is_file():
            frames = json.loads(partial_path.read_text(encoding="utf-8"))
        elif results_path.is_file():
            frames = json.loads(results_path.read_text(encoding="utf-8"))["frames"]
        else:
            raise FileNotFoundError(f"No partial or completed candidate results in {output}")
        report = finalize_regression_frames(frames, device=args.device,
            model_path=args.model_path,
            annotation_note=json.loads(Path(args.manifest).read_text(encoding="utf-8")).get("annotation_note"),
            output_dir=output)
        print(json.dumps(report["aggregate"], indent=2))
        return 0
    report = run_regression(args.manifest, model_path=args.model_path,
                            device=args.device, output_dir=args.output_dir)
    print(json.dumps(report["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
