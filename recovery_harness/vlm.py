from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from .models import MultiViewObservation, SkillRecommendation
from .registry import SkillRegistry

log = logging.getLogger("kuavo.recovery.vlm")


SYSTEM_PROMPT = """You are a visual scene reasoner for a Kuavo package-sorting robot.
Inspect only the supplied images and evidence. Distinguish current observation from history.
Use the original RGB images as primary scene evidence. Optional detector/tracker geometry can be empty or unreliable; that alone is not a reason to stop if the scene is visually interpretable.
When head-camera platform polygon and body-side edge coordinates are supplied, compare the package location with those image-pixel regions. Do not apply that fixed geometry to wrist-camera images or treat a partial bounding box as the full package footprint.
Do not infer a grasp from tactile contact alone. Choose stop_and_verify only when the available images and evidence leave the scene or target genuinely ambiguous.
Recommend exactly one available skill or stop_and_verify. Never produce robot actions or joint commands.
Return one JSON object with keys: scene_state, task_progress, recommended_skill,
target_object, reason, requires_verification. Use unknown when the evidence is insufficient."""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("VLM did not return a JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("VLM output must be a JSON object")
    return value


class QwenVLMAdapter:
    """Loads exactly one configured backend with its model-specific class/processor."""
    def __init__(self, backend: str, model_path: str, *, device_map: str = "auto", dtype: str = "bfloat16"):
        self.backend, self.model_path = backend, str(Path(model_path).expanduser())
        if not Path(self.model_path).is_dir():
            raise FileNotFoundError(f"Configured VLM model directory not found: {self.model_path}")
        import torch
        from transformers import AutoProcessor
        if backend == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration
            model_cls = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        elif backend == "qwen3_vl":
            try:
                from transformers import Qwen3VLForConditionalGeneration
            except ImportError as exc:
                raise RuntimeError("Installed transformers lacks Qwen3VLForConditionalGeneration; install the model's supported transformers build") from exc
            model_cls = Qwen3VLForConditionalGeneration
            # Qwen3 has a distinct processor/chat template and class.
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        else:
            raise ValueError(f"Unsupported VLM backend: {backend}")
        dtype_value = getattr(torch, dtype, None)
        if dtype_value is None:
            raise ValueError(f"Unknown torch dtype: {dtype}")
        self.model = model_cls.from_pretrained(
            self.model_path, torch_dtype=dtype_value, device_map=device_map,
            trust_remote_code=True,
        ).eval()

    def generate_json(self, images: list[Any], prompt: str, *, max_new_tokens: int = 256) -> tuple[dict[str, Any], float]:
        import torch
        from PIL import Image
        normalized_images = [Image.fromarray(image) if not isinstance(image, Image.Image) else image for image in images]
        content = [{"type": "image", "image": image} for image in normalized_images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=normalized_images, padding=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1]:]
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return _extract_json(decoded), time.perf_counter() - start


class SceneReasoner:
    def __init__(self, registry: SkillRegistry, adapter: QwenVLMAdapter):
        self.registry, self.adapter = registry, adapter

    def reason(self, observation: MultiViewObservation) -> tuple[SkillRecommendation, float]:
        frames = [observation.images[name] for name in ("head", "left_wrist", "right_wrist") if observation.images.get(name) is not None]
        if not frames:
            return SkillRecommendation("unknown", "unknown", "stop_and_verify", None, "No current camera image is available", True), 0.0
        image_values = [frame.image for frame in frames]
        view_notes = [f"Image {i+1} is {frame.name} view; stamp={frame.timestamp}; timestamp_source={frame.timestamp_source}." for i, frame in enumerate(frames)]
        context = {
            "available_skills": self.registry.prompt_descriptions(),
            "available_views": view_notes,
            "perception_evidence": observation.perception,
            "tactile_contact": observation.tactile,
            "robot_state": observation.robot_state,
            "recent_history": observation.history,
            "output_contract": {"allowed_recommended_skill": sorted(self.registry.allowed_recommendations())},
        }
        prompt = "Available robot skills (generated from the live Skill Registry):\n" + context["available_skills"] + "\n\nObservation context:\n" + json.dumps(context, default=str, ensure_ascii=False) + "\nCompare the visible package arrangement with every registered skill description and recommend the single best matching skill. Use the original RGB images directly; do not require detector boxes or numeric geometry when the scene itself is clear. Return JSON only."
        start = time.perf_counter()
        try:
            raw, latency = self.adapter.generate_json(image_values, prompt)
        except Exception as exc:
            latency = time.perf_counter() - start
            log.exception("VLM reasoning failed; routing to stop_and_verify")
            return SkillRecommendation("unknown", "unknown", "stop_and_verify", None,
                                       f"VLM inference failed: {type(exc).__name__}: {exc}", True), latency
        skill = raw.get("recommended_skill")
        required_valid = (
            skill in self.registry.allowed_recommendations()
            and isinstance(raw.get("scene_state", "unknown"), str)
            and isinstance(raw.get("task_progress", "unknown"), str)
            and isinstance(raw.get("reason", ""), str)
            and isinstance(raw.get("requires_verification", True), bool)
            and (raw.get("target_object") is None or isinstance(raw.get("target_object"), str))
        )
        if not required_valid:
            raw = {"scene_state": "unknown", "task_progress": "unknown", "recommended_skill": "stop_and_verify", "target_object": None, "reason": f"Invalid VLM output or unregistered skill recommendation: {skill!r}", "requires_verification": True}
        rec = SkillRecommendation(
            scene_state=str(raw.get("scene_state", "unknown")), task_progress=str(raw.get("task_progress", "unknown")),
            recommended_skill=str(raw["recommended_skill"]), target_object=raw.get("target_object"),
            reason=str(raw.get("reason", "")), requires_verification=bool(raw.get("requires_verification", True)), raw=raw,
        )
        # Keep the exact registry-derived tool descriptions alongside the model answer so the
        # routing test can audit that both available skills were present in this request.
        rec.raw["registry_context"] = {
            "skill_ids": [skill.skill_id for skill in self.registry],
            "descriptions_injected": self.registry.prompt_descriptions(),
        }
        return rec, latency

