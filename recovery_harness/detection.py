"""Explainable package-candidate matching and arm-conflict decisions.

This module has no model/runtime dependencies so its safety-relevant decisions
can be exercised independently from Grounding DINO and SAM 2.
"""
from __future__ import annotations

import re
import time
from typing import Any, Iterable


_GENERIC_PACKAGE_TERMS = {"package", "parcel", "mailer bag", "padded envelope"}


def normalize_dino_phrase(phrase: str | None) -> str:
    """Normalize punctuation and tokenizer artifacts without inventing labels."""
    value = (phrase or "").lower().replace("##", "")
    return " ".join(re.findall(r"[a-z0-9]+", value))


def _contains_phrase(tokens: list[str], phrase: str) -> bool:
    wanted = phrase.split()
    return any(tokens[index:index + len(wanted)] == wanted
               for index in range(max(0, len(tokens) - len(wanted) + 1)))


def classify_dino_phrase(raw_phrase: str | None, package_labels: Iterable[str], *,
                         prompt_label: str | None = None,
                         allow_generic_package: bool = False) -> dict[str, Any]:
    """Return a conservative, auditable phrase-to-class decision.

    Multi-class phrases are retained as ambiguous. Generic package nouns are
    never silently converted into a configured class. A class-specific query
    may ground an unambiguous token such as ``pink`` to its sole prompted class.
    """
    normalized = normalize_dino_phrase(raw_phrase)
    labels = tuple(dict.fromkeys(normalize_dino_phrase(label) for label in package_labels))
    tokens = normalized.split()
    if not normalized:
        return {"raw_phrase": raw_phrase, "normalized_phrase": normalized,
                "normalized_class": None, "class_hypotheses": [],
                "validity": "rejected", "reason": "empty_grounded_phrase"}

    matching = [label for label in labels if _contains_phrase(tokens, label)]
    if prompt_label:
        scoped = normalize_dino_phrase(prompt_label)
        # An individual-label query is evidence only when the decoded phrase
        # consists of, or contains, tokens from that exact query.
        if scoped in labels and _contains_phrase(tokens, scoped):
            matching = [scoped]
        elif scoped in labels and tokens and all(token in scoped.split() for token in tokens):
            matching = [scoped]

    if len(matching) == 1:
        return {"raw_phrase": raw_phrase, "normalized_phrase": normalized,
                "normalized_class": matching[0], "class_hypotheses": matching,
                "validity": "valid", "reason": "unique_configured_package_class"}
    if len(matching) > 1:
        return {"raw_phrase": raw_phrase, "normalized_phrase": normalized,
                "normalized_class": None, "class_hypotheses": matching,
                "validity": "ambiguous", "reason": "phrase_contains_multiple_package_classes"}

    if allow_generic_package and normalized in _GENERIC_PACKAGE_TERMS:
        return {"raw_phrase": raw_phrase, "normalized_phrase": normalized,
                "normalized_class": None, "class_hypotheses": [],
                "validity": "ambiguous", "reason": "generic_package_phrase_has_no_specific_class"}
    return {"raw_phrase": raw_phrase, "normalized_phrase": normalized,
            "normalized_class": None, "class_hypotheses": [],
            "validity": "rejected", "reason": "phrase_matches_no_configured_package_class"}


def box_iou(left: Iterable[float], right: Iterable[float]) -> float:
    a, b = tuple(map(float, left)), tuple(map(float, right))
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def fraction_covered(box: Iterable[float], region: Iterable[float]) -> float:
    a, b = tuple(map(float, box)), tuple(map(float, region))
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    return intersection / area if area else 0.0


def _intersection_over_smaller(left, right) -> float:
    a, b = tuple(map(float, left)), tuple(map(float, right))
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / min(area_a, area_b) if min(area_a, area_b) else 0.0


def merge_prompt_candidates(candidates: Iterable[dict[str, Any]], package_labels: Iterable[str], *,
                            duplicate_iou: float = 0.55,
                            duplicate_containment: float = 0.85) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate overlapping prompt results without comparing cross-prompt scores.

    An unambiguous class-specific response takes precedence over joint/generic
    prompt evidence. Confidence is used only to select among same-prompt
    duplicates. All contributing prompt evidence remains attached.
    """
    labels = tuple(package_labels)
    enriched = []
    for index, original in enumerate(candidates):
        item = dict(original)
        prompt_label = item.get("prompt_label") if item.get("query_role") == "per_class" else None
        item["phrase_match"] = classify_dino_phrase(
            item.get("raw_phrase", item.get("phrase")), labels,
            prompt_label=prompt_label,
            allow_generic_package=item.get("query_role") == "generic")
        item["candidate_index"] = item.get("candidate_index", index)
        enriched.append(item)

    # Build overlap clusters transitively so joint and per-class duplicates
    # become one object while overlapping distinct package classes stay marked.
    parents = list(range(len(enriched)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for i, left in enumerate(enriched):
        for j in range(i + 1, len(enriched)):
            right = enriched[j]
            same_query = (
                left.get("query_role") == right.get("query_role")
                and left.get("prompt_label") == right.get("prompt_label")
                and left.get("prompt") == right.get("prompt")
            )
            if same_query:
                # Same-query proposals are handled by classwise NMS below;
                # cross-prompt clustering is only for repeated evidence.
                continue
            if (box_iou(left["box"], right["box"]) >= duplicate_iou
                    or _intersection_over_smaller(left["box"], right["box"]) >= duplicate_containment):
                ri, rj = root(i), root(j)
                if ri != rj:
                    parents[rj] = ri

    clusters: dict[int, list[dict[str, Any]]] = {}
    for index, item in enumerate(enriched):
        clusters.setdefault(root(index), []).append(item)

    merged, rejected = [], []
    role_order = {"per_class": 0, "joint": 1, "generic": 2}
    for group in clusters.values():
        valid = [item for item in group if item["phrase_match"]["validity"] == "valid"]
        classes = sorted({item["phrase_match"]["normalized_class"] for item in valid})
        ambiguous_evidence = [item for item in group if item["phrase_match"]["validity"] == "ambiguous"]
        # Prefer semantic evidence from a per-class query. Confidence only
        # breaks ties within the same prompt, never across prompt families.
        source_pool = valid if valid else (ambiguous_evidence if ambiguous_evidence else group)
        source_pool = sorted(source_pool, key=lambda item: (
            role_order.get(item.get("query_role", "joint"), 1),
            item.get("prompt_label") or "", item.get("prompt") or "",
            -float(item.get("confidence") or 0.0)))
        representative = dict(source_pool[0])
        evidence = [{"raw_phrase": item.get("raw_phrase", item.get("phrase")),
                     "confidence": item.get("confidence"), "box": item.get("box"),
                     "prompt": item.get("prompt"), "query_role": item.get("query_role"),
                     "prompt_label": item.get("prompt_label"),
                     "phrase_match": item["phrase_match"]} for item in group]
        representative["prompt_evidence"] = evidence
        representative["raw_phrase"] = representative.get("raw_phrase", representative.get("phrase"))
        specific_classes = {item["phrase_match"]["normalized_class"] for item in valid
                            if item.get("query_role") == "per_class"}
        if len(classes) == 1 and (not ambiguous_evidence or len(specific_classes) == 1):
            representative["label"] = classes[0]
            representative["normalized_class"] = classes[0]
            representative["candidate_validity"] = "valid"
            representative["candidate_reason"] = (
                "class_specific_prompt_resolved_merged_phrase" if ambiguous_evidence
                else "deduplicated_unambiguous_package_evidence")
        elif len(classes) > 1 or ambiguous_evidence:
            representative["label"] = "package_candidate"
            representative["normalized_class"] = None
            representative["candidate_validity"] = "ambiguous"
            representative["candidate_reason"] = (
                "overlapping_prompt_evidence_has_conflicting_package_classes" if len(classes) > 1
                else "merged_or_generic_package_phrase_requires_class_specific_evidence")
            representative["class_hypotheses"] = sorted(set(classes).union(
                label for item in ambiguous_evidence
                for label in item["phrase_match"].get("class_hypotheses", [])))
        else:
            representative["label"] = None
            representative["normalized_class"] = None
            representative["candidate_validity"] = "rejected"
            representative["candidate_reason"] = "no_prompt_evidence_maps_to_a_configured_package_class"
        representative["duplicate_count"] = len(group)
        representative["duplicate_candidate_indices"] = [item["candidate_index"] for item in group]
        (merged if representative["candidate_validity"] != "rejected" else rejected).append(representative)
    return merged, rejected


def arm_box_geometry(box: Iterable[float], frame_width: int, frame_height: int, *,
                     edge_margin_px: float = 4.0,
                     elongated_aspect_max: float = 0.80,
                     elongated_aspect_min: float = 1.25) -> dict[str, Any]:
    """Describe arm-like image geometry; a box alone is not an arm mask."""
    x1, y1, x2, y2 = tuple(map(float, box))
    width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
    aspect = width / max(1.0, height)
    edge_distances = {"left": x1, "top": y1, "right": frame_width - x2, "bottom": frame_height - y2}
    nearest_edge = min(edge_distances, key=edge_distances.get)
    touches_edge = edge_distances[nearest_edge] <= edge_margin_px
    elongated = aspect <= elongated_aspect_max or aspect >= elongated_aspect_min
    return {"aspect_ratio": aspect, "nearest_frame_edge": nearest_edge,
            "edge_distance_px": edge_distances[nearest_edge],
            "touches_frame_edge": touches_edge, "elongated": elongated,
            "reliable_arm_geometry": bool(touches_edge and elongated)}


def resolve_arm_conflict(package_candidate: dict[str, Any], arm_candidates: Iterable[dict[str, Any]], *,
                         frame_width: int, frame_height: int,
                         overlap_reject_threshold: float = 0.35,
                         partial_overlap_threshold: float = 0.02,
                         reliable_arm_min_score: float = 0.30) -> dict[str, Any]:
    """Reject only geometrically supported arms; otherwise retain as ambiguous.

    Score is used to establish minimum standalone arm detection quality, not
    compared with package scores. Conflicts without arm geometry are preserved
    but can never be treated as executable recovery evidence.
    """
    box = package_candidate["box"]
    conflicts = []
    for arm in arm_candidates:
        overlap = fraction_covered(box, arm["box"])
        geometry = arm_box_geometry(arm["box"], frame_width, frame_height)
        reliable = bool(geometry["reliable_arm_geometry"]
                        and float(arm.get("confidence") or 0.0) >= reliable_arm_min_score)
        conflicts.append({"package_score": package_candidate.get("confidence"),
                          "package_box": list(box),
                          "package_raw_phrase": package_candidate.get("raw_phrase"),
                          "arm_score": arm.get("confidence"), "arm_box": list(arm["box"]),
                          "arm_raw_phrase": arm.get("raw_phrase", arm.get("phrase")),
                          "arm_overlap_fraction": overlap,
                          "arm_geometry": geometry,
                          "arm_reliable": reliable})
    conflict = max(conflicts, key=lambda item: item["arm_overlap_fraction"], default=None)
    if conflict is None or conflict["arm_overlap_fraction"] < partial_overlap_threshold:
        return {"decision": "keep", "ambiguous": False,
                "partial": False, "occluded": False, "conflict": conflict,
                "reason": "no_material_arm_overlap"}
    if conflict["arm_overlap_fraction"] >= overlap_reject_threshold:
        if conflict["arm_reliable"]:
            return {"decision": "reject", "ambiguous": False,
                    "partial": True, "occluded": True, "conflict": conflict,
                    "reason": "high_overlap_with_reliable_edge_anchored_arm_geometry"}
        return {"decision": "ambiguous", "ambiguous": True,
                "partial": True, "occluded": True, "conflict": conflict,
                "reason": "high_overlap_but_arm_box_lacks_reliable_geometry"}
    return {"decision": "keep", "ambiguous": False,
            "partial": True, "occluded": True, "conflict": conflict,
            "reason": "partial_arm_overlap_retained_as_occluded"}


def filter_package_candidate_groups(candidates: Iterable[dict[str, Any]],
                                    arm_candidates: Iterable[dict[str, Any]], *,
                                    package_labels: Iterable[str],
                                    frame_width: int, frame_height: int,
                                    box_threshold: float = 0.20,
                                    min_area_ratio: float = 0.004,
                                    max_area_ratio: float = 0.15,
                                    min_aspect: float = 0.35,
                                    max_aspect: float = 2.8,
                                    nms_iou: float = 0.50,
                                    arm_overlap_max: float = 0.35,
                                    arm_partial_overlap: float = 0.02,
                                    max_instances: int = 6,
                                    reliable_arm_min_score: float = 0.30) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply package gates and return accepted detections plus stage audit.

    Ambiguous candidates meeting the unchanged confidence/geometry gates are
    returned for diagnostic tracking with ``ambiguous=True``. EventTrigger
    must exclude these from recovery evidence.
    """
    package_labels = tuple(package_labels)
    mergeable = []
    rejected_phrase = []
    for index, candidate in enumerate(candidates):
        item = dict(candidate)
        prompt_label = item.get("prompt_label") if item.get("query_role") == "per_class" else None
        phrase_match = classify_dino_phrase(item.get("raw_phrase", item.get("phrase")),
            package_labels, prompt_label=prompt_label,
            allow_generic_package=item.get("query_role") == "generic")
        item["phrase_match"] = phrase_match
        item["candidate_index"] = item.get("candidate_index", index)
        if phrase_match["validity"] == "rejected":
            item.update({"candidate_validity": "rejected", "stage": "phrase_matching",
                         "stage_decision": "rejected", "stage_reason": phrase_match["reason"]})
            rejected_phrase.append(item)
        elif float(item.get("confidence") or 0.0) < box_threshold:
            item.update({"candidate_validity": "rejected", "stage_decisions": [{
                "stage": "confidence_threshold", "decision": "rejected",
                "reason": f"score {float(item.get('confidence') or 0.0):.6f} below {box_threshold:.6f}"}]})
            rejected_phrase.append(item)
        else:
            mergeable.append(item)
    merged, rejected_by_phrase = merge_prompt_candidates(mergeable, package_labels)
    rejected_phrase.extend(rejected_by_phrase)
    arms = list(arm_candidates)
    audit = [{**item, "stage": "phrase_matching", "stage_decision": "rejected",
              "stage_reason": item.get("candidate_reason")} for item in rejected_phrase]
    pre_nms = []
    for item in merged:
        item = dict(item)
        item["stage_decisions"] = []
        score = float(item.get("confidence") or 0.0)
        item["stage_decisions"].append({"stage": "confidence_threshold", "decision": "passed",
                                         "reason": f"score {score:.6f} >= {box_threshold:.6f}"})
        x1, y1, x2, y2 = map(float, item["box"])
        width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
        area_ratio = width * height / max(1.0, frame_width * frame_height)
        aspect = width / max(1.0, height)
        item["area_ratio"] = area_ratio
        item["aspect_ratio"] = aspect
        if not min_area_ratio <= area_ratio <= max_area_ratio:
            item["candidate_validity"] = "rejected"
            item["stage_decisions"].append({"stage": "area_filter", "decision": "rejected",
                                             "reason": f"area ratio {area_ratio:.6f} outside [{min_area_ratio:.6f},{max_area_ratio:.6f}]"})
            audit.append(item)
            continue
        if not min_aspect <= aspect <= max_aspect:
            item["candidate_validity"] = "rejected"
            item["stage_decisions"].append({"stage": "aspect_filter", "decision": "rejected",
                                             "reason": f"aspect {aspect:.6f} outside [{min_aspect:.6f},{max_aspect:.6f}]"})
            audit.append(item)
            continue
        item["stage_decisions"].append({"stage": "area_aspect_filter", "decision": "passed",
                                         "reason": "box has plausible package size and aspect"})
        pre_nms.append(item)

    # Classwise NMS preserves separate, semantically distinct package classes.
    by_class: dict[str, list[dict[str, Any]]] = {}
    for item in pre_nms:
        group = item.get("normalized_class") or "ambiguous:" + ",".join(item.get("class_hypotheses", []))
        by_class.setdefault(group, []).append(item)
    post_nms = []
    for class_name, items in by_class.items():
        kept: list[dict[str, Any]] = []
        for item in sorted(items, key=lambda value: float(value.get("confidence") or 0.0), reverse=True):
            suppressor = next((prior for prior in kept
                               if box_iou(prior["box"], item["box"]) >= nms_iou), None)
            if suppressor:
                item["candidate_validity"] = "rejected"
                item["stage_decisions"].append({"stage": "nms", "decision": "rejected",
                    "reason": f"same-class IoU {box_iou(suppressor['box'], item['box']):.6f} >= {nms_iou:.6f}",
                    "suppressed_by_box": list(suppressor["box"])})
                audit.append(item)
            else:
                item["stage_decisions"].append({"stage": "nms", "decision": "passed",
                    "reason": "no higher-scoring same-class overlap above threshold"})
                kept.append(item)
        post_nms.extend(kept)

    after_arm = []
    for item in post_nms:
        conflict = resolve_arm_conflict(item, arms, frame_width=frame_width,
            frame_height=frame_height, overlap_reject_threshold=arm_overlap_max,
            partial_overlap_threshold=arm_partial_overlap,
            reliable_arm_min_score=reliable_arm_min_score)
        item["arm_conflict"] = conflict
        item["partial"] = bool(conflict["partial"])
        item["occluded"] = bool(conflict["occluded"])
        if conflict["decision"] == "reject":
            item["candidate_validity"] = "rejected"
            item["stage_decisions"].append({"stage": "arm_overlap", "decision": "rejected",
                                             "reason": conflict["reason"], "evidence": conflict["conflict"]})
            audit.append(item)
            continue
        if conflict["decision"] == "ambiguous":
            item["candidate_validity"] = "ambiguous"
            item["ambiguous"] = True
            item["stage_decisions"].append({"stage": "arm_overlap", "decision": "ambiguous",
                                             "reason": conflict["reason"], "evidence": conflict["conflict"]})
        else:
            item["ambiguous"] = item.get("candidate_validity") == "ambiguous"
            item["stage_decisions"].append({"stage": "arm_overlap", "decision": "passed",
                                             "reason": conflict["reason"], "evidence": conflict["conflict"]})
        after_arm.append(item)

    # Respect the legacy cap without comparing scores from distinct prompts.
    label_order = {normalize_dino_phrase(label): index for index, label in enumerate(package_labels)}
    after_arm.sort(key=lambda item: (label_order.get(item.get("normalized_class"), len(label_order)),
                                     item.get("prompt_label") or "",
                                     -float(item.get("confidence") or 0.0)))
    accepted = after_arm[:max_instances]
    for item in accepted:
        item["stage_decisions"].append({"stage": "instance_cap", "decision": "passed",
                                         "reason": f"within max_instances={max_instances}"})
    for item in after_arm[max_instances:]:
        item["candidate_validity"] = "rejected"
        item["stage_decisions"].append({"stage": "instance_cap", "decision": "rejected",
                                         "reason": f"exceeds max_instances={max_instances}"})
        audit.append(item)
    audit.extend(accepted)
    return accepted, audit


class GroundingDinoPromptRunner:
    """Load local Grounding DINO once and expose auditable prompt queries."""

    def __init__(self, model_path: str, *, device: str,
                 package_labels: Iterable[str], joint_prompt: str,
                 generic_prompt: str = "package. parcel. mailer bag. padded envelope.",
                 box_threshold: float = 0.20, text_threshold: float = 0.10,
                 arm_box_threshold: float = 0.15, arm_text_threshold: float = 0.05,
                 min_area_ratio: float = 0.004, max_area_ratio: float = 0.15,
                 nms_iou: float = 0.50):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = device
        self.package_labels = tuple(package_labels)
        self.joint_prompt = joint_prompt
        self.generic_prompt = generic_prompt
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.arm_box_threshold = float(arm_box_threshold)
        self.arm_text_threshold = float(arm_text_threshold)
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        self.nms_iou = float(nms_iou)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path, local_files_only=True).to(device).eval()

    @staticmethod
    def _labels(result) -> list[str]:
        values = result.get("text_labels")
        if values is None:
            values = result.get("labels", [])
        return [str(value).lower().strip(" .") for value in values]

    def _run_query(self, image, prompt: str, *, query_role: str,
                   prompt_label: str | None, box_threshold: float,
                   text_threshold: float, frame_size: tuple[int, int]) -> dict[str, Any]:
        width, height = frame_size
        started = time.perf_counter()
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = [image.size[::-1]]

        def post(box_cut: float, text_cut: float):
            return self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=box_cut,
                text_threshold=text_cut, target_sizes=target_sizes)[0]

        raw = post(0.0, 0.0)
        box_only = post(box_threshold, 0.0)
        text_only = post(0.0, text_threshold)
        production = post(box_threshold, text_threshold)

        def unpack(result):
            labels = self._labels(result)
            rows = []
            for index, (box, score, phrase) in enumerate(zip(
                    result.get("boxes", []), result.get("scores", []), labels)):
                rows.append({"candidate_index": index,
                             "box": [float(value) for value in box.tolist()],
                             "confidence": float(score), "raw_phrase": phrase,
                             "prompt": prompt, "query_role": query_role,
                             "prompt_label": prompt_label})
            return rows

        raw_rows, box_rows, text_rows, production_rows = map(unpack, (raw, box_only, text_only, production))

        def same_box(row, others):
            return max((box_iou(row["box"], candidate["box"]) for candidate in others), default=0.0)

        for candidate in raw_rows:
            box_pass = candidate["confidence"] >= box_threshold
            text_match = same_box(candidate, text_rows)
            joint_match = same_box(candidate, production_rows)
            box_and_text_match = same_box(candidate, box_rows)
            candidate["postprocess_audit"] = {
                "box_threshold_passed": bool(box_pass),
                "box_threshold_only_match_iou": box_and_text_match,
                "text_threshold_only_match_iou": text_match,
                "both_thresholds_match_iou": joint_match,
                "box_threshold_rejected": not box_pass,
                "text_threshold_rejected": bool(box_pass and text_match < 0.99),
                "postprocessor_rejection_reason": (
                    "box_threshold" if not box_pass else
                    "text_threshold" if text_match < 0.99 else None),
            }
            candidate["phrase_match"] = classify_dino_phrase(
                candidate["raw_phrase"], self.package_labels,
                prompt_label=prompt_label if query_role == "per_class" else None,
                allow_generic_package=query_role == "generic")

        return {"prompt": prompt, "query_role": query_role,
                "prompt_label": prompt_label,
                "elapsed_seconds": time.perf_counter() - started,
                "raw_candidates": raw_rows,
                "box_threshold_candidates": box_rows,
                "text_threshold_candidates": text_rows,
                "postprocess_candidates": production_rows}

    def detect(self, frame, *, query_mode: str = "per_class") -> dict[str, Any]:
        """Run prompts and return filtered candidates plus complete diagnostics.

        Modes are ``legacy_joint``, ``per_class``, ``joint_plus_per_class`` and
        ``generic``. Generic mode is intended only for offline diagnosis.
        """
        import cv2
        from PIL import Image

        height, width = frame.shape[:2]
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if query_mode == "legacy_joint":
            queries = [(self.joint_prompt, "joint", None)]
        elif query_mode == "per_class":
            queries = [(f"{label}.", "per_class", label) for label in self.package_labels]
        elif query_mode == "joint_plus_per_class":
            queries = [(self.joint_prompt, "joint", None)] + [
                (f"{label}.", "per_class", label) for label in self.package_labels]
        elif query_mode == "generic":
            queries = [(self.generic_prompt, "generic", None)]
        else:
            raise ValueError(f"Unsupported Grounding DINO query mode: {query_mode}")

        started = time.perf_counter()
        query_results = [self._run_query(image, prompt, query_role=role,
            prompt_label=label, box_threshold=0.02, text_threshold=self.text_threshold,
            frame_size=(width, height)) for prompt, role, label in queries]
        package_evidence = [candidate for result in query_results
                            for candidate in result["postprocess_candidates"]]

        arm_result = self._run_query(image, "robot arm. robot gripper.",
            query_role="arm", prompt_label=None, box_threshold=self.arm_box_threshold,
            text_threshold=self.arm_text_threshold, frame_size=(width, height))
        arm_candidates = []
        arm_rejected = []
        for candidate in arm_result["postprocess_candidates"]:
            x1, y1, x2, y2 = candidate["box"]
            area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1, width * height)
            if self.min_area_ratio <= area_ratio <= self.max_area_ratio:
                arm_candidates.append({**candidate, "class": "robot arm",
                                       "arm_context_only": True,
                                       "area_ratio": area_ratio})
            else:
                arm_rejected.append({**candidate, "candidate_validity": "rejected",
                                     "stage_decisions": [{"stage": "area_filter",
                                         "decision": "rejected",
                                         "reason": f"arm area ratio {area_ratio:.6f} outside [{self.min_area_ratio:.6f},{self.max_area_ratio:.6f}]"}]})

        arm_kept = []
        for arm in sorted(arm_candidates, key=lambda item: float(item.get("confidence") or 0.0), reverse=True):
            suppressor = next((prior for prior in arm_kept
                               if box_iou(prior["box"], arm["box"]) >= self.nms_iou), None)
            if suppressor:
                arm_rejected.append({**arm, "candidate_validity": "rejected",
                    "stage_decisions": [{"stage": "arm_nms", "decision": "rejected",
                        "reason": f"same-arm IoU {box_iou(suppressor['box'], arm['box']):.6f} >= {self.nms_iou:.6f}",
                        "suppressed_by_box": list(suppressor["box"])}]})
            else:
                arm_kept.append(arm)
        arm_candidates = arm_kept

        accepted, filter_audit = filter_package_candidate_groups(
            package_evidence, arm_candidates, package_labels=self.package_labels,
            frame_width=width, frame_height=height,
            box_threshold=self.box_threshold,
            min_area_ratio=self.min_area_ratio,
            max_area_ratio=self.max_area_ratio,
            nms_iou=self.nms_iou)
        filter_audit.extend(arm_rejected)
        diagnostics = {
            "query_mode": query_mode,
            "frame_size": [width, height],
            "configured_thresholds": {"package_box": self.box_threshold,
                "package_text": self.text_threshold,
                "arm_box": self.arm_box_threshold,
                "arm_text": self.arm_text_threshold,
                "nms_iou": self.nms_iou},
            "queries": query_results,
            "arm_query": arm_result,
            "arm_candidates": arm_candidates,
            "filter_audit": filter_audit,
            "final_package_candidates": accepted,
            "roi_filtering": {"applied": False,
                "reason": "DINO package proposals are not clipped or rejected by the calibrated platform ROI; ROI relation remains downstream perception evidence."},
            "total_detection_seconds": time.perf_counter() - started,
        }
        # Only candidates that pass the unchanged score/geometry gates are
        # passed to SAM2. Ambiguous candidates remain diagnostic, occluded,
        # and non-actionable in EventTrigger.
        detections = [{**item, "class": item.get("label") or "package_candidate",
                       "confidence": float(item.get("confidence") or 0.0),
                       "box": tuple(map(float, item["box"])),
                       "source": "grounding_dino_classwise",
                       "partial": bool(item.get("partial")),
                       "occluded": bool(item.get("occluded")),
                       "ambiguous": bool(item.get("ambiguous")),
                       "diagnostics": item}
                      for item in accepted]
        for arm in arm_candidates:
            arm["raw_phrase"] = arm.get("raw_phrase", "")
        return {"detections": detections, "arm_candidates": arm_candidates,
                "diagnostics": diagnostics}
