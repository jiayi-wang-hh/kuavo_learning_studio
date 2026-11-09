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
    def __init__(self, backend: str, model_path: str, *, device_map: str = "auto", dtype: str = "bfloat16",
                 quantization: str | None = None):
        self.backend, self.model_path = backend, str(Path(model_path).expanduser())
        if not Path(self.model_path).is_dir():
            raise FileNotFoundError(f"Configured VLM model directory not found: {self.model_path}")
        import torch
        from transformers import AutoProcessor
        if backend == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration
            model_cls = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True, use_fast=True,
            )
        elif backend == "qwen3_vl":
            try:
                from transformers import Qwen3VLForConditionalGeneration
            except ImportError as exc:
                raise RuntimeError("Installed transformers lacks Qwen3VLForConditionalGeneration; install the model's supported transformers build") from exc
            model_cls = Qwen3VLForConditionalGeneration
            # Qwen3 has a distinct processor/chat template and class.
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        elif backend == "qwen3_5":
            from transformers import Qwen3_5ForConditionalGeneration
            model_cls = Qwen3_5ForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        else:
            raise ValueError(f"Unsupported VLM backend: {backend}")
        dtype_value = getattr(torch, dtype, None)
        if dtype_value is None:
            raise ValueError(f"Unknown torch dtype: {dtype}")
        load_options = {}
        if backend == "qwen3_5":
            load_options.update(dtype=dtype_value, local_files_only=True, attn_implementation="sdpa")
        else:
            load_options.update(torch_dtype=dtype_value, trust_remote_code=True)
        if quantization is not None:
            if backend != "qwen3_5" or quantization != "nf4":
                raise ValueError("Optional NF4 loading is supported only for qwen3_5")
            from transformers import BitsAndBytesConfig
            load_options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype_value,
            )
        self.model = model_cls.from_pretrained(self.model_path, device_map=device_map, **load_options).eval()
        self._torch = torch
        device_map_report = getattr(self.model, "hf_device_map", None)
        if device_map_report is None:
            device_map_report = {"first_parameter": str(next(self.model.parameters()).device)}
        self._cuda_sync_devices = set()
        for placement in device_map_report.values():
            text = str(placement)
            if text.startswith("cuda"):
                self._cuda_sync_devices.add(text)
            elif isinstance(placement, int):
                self._cuda_sync_devices.add(f"cuda:{placement}")
        if not self._cuda_sync_devices and torch.cuda.is_available():
            self._cuda_sync_devices.add(str(next(self.model.parameters()).device))
        self.last_timing: dict[str, Any] = {}
        log.info("vlm_loaded backend=%s processor=%s image_processor=%s dtype=%s device_map=%s attention=%s cuda_available=%s",
            backend, type(self.processor).__name__, type(getattr(self.processor, "image_processor", None)).__name__,
            dtype, device_map_report, getattr(getattr(self.model, "config", None), "_attn_implementation", None),
            torch.cuda.is_available())

    def _synchronize_model_devices(self) -> None:
        if not self._torch.cuda.is_available():
            return
        for device in self._cuda_sync_devices:
            self._torch.cuda.synchronize(device)

    def generate_json(self, images: list[Any], prompt: str, *, max_new_tokens: int = 256) -> tuple[dict[str, Any], float]:
        from PIL import Image
        torch = self._torch
        total_start = time.perf_counter()
        preprocess_start = total_start
        normalized_images = [Image.fromarray(image) if not isinstance(image, Image.Image) else image for image in images]
        content = [{"type": "image", "image": image} for image in normalized_images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=normalized_images, padding=True, return_tensors="pt")
        preprocess_seconds = time.perf_counter() - preprocess_start
        device = next(self.model.parameters()).device
        transfer_start = time.perf_counter()
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        transfer_seconds = time.perf_counter() - transfer_start
        self._synchronize_model_devices()
        generation_start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        self._synchronize_model_devices()
        generation_seconds = time.perf_counter() - generation_start
        decode_start = time.perf_counter()
        generated = output[:, inputs["input_ids"].shape[1]:]
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        parsed = _extract_json(decoded)
        decode_seconds = time.perf_counter() - decode_start
        self.last_timing = {
            "preprocess_seconds": preprocess_seconds,
            "input_transfer_seconds": transfer_seconds,
            "generation_seconds": generation_seconds,
            "decode_and_json_seconds": decode_seconds,
            "adapter_total_seconds": time.perf_counter() - total_start,
            "input_image_count": len(normalized_images),
            "input_text_tokens": int(inputs["input_ids"].shape[1]),
            "generated_tokens": int(generated.shape[-1]),
            "image_grid_thw": inputs.get("image_grid_thw").detach().cpu().tolist()
                if inputs.get("image_grid_thw") is not None else None,
        }
        return parsed, generation_seconds + decode_seconds


class Qwen35SingleFrameAdapter(QwenVLMAdapter):
    """Single RGB frame, caller-supplied prompt, raw text; no registry or routing."""

    def __init__(self, model_path: str, *, precision: str = "nf4", device_map: str = "cuda:0"):
        if precision not in {"nf4", "bf16"}:
            raise ValueError("precision must be nf4 or bf16")
        super().__init__("qwen3_5", model_path, device_map=device_map, dtype="bfloat16",
                         quantization="nf4" if precision == "nf4" else None)

    def generate_raw(self, image, prompt: str, *, max_new_tokens: int = 24,
                     before_generate=None) -> tuple[str, dict]:
        start = time.perf_counter()
        content = [{"type": "text", "text": prompt}, {"type": "image", "image": image}]
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        self._synchronize_model_devices()
        if before_generate is not None:
            before_generate()
        generation_start = time.perf_counter()
        with self._torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        self._synchronize_model_devices()
        generation_seconds = time.perf_counter() - generation_start
        generated = output[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return raw, {"generation_seconds": generation_seconds,
                     "adapter_total_seconds": time.perf_counter() - start,
                     "input_image_count": 1, "input_tokens": int(inputs["input_ids"].shape[1]),
                     "generated_tokens": int(generated.shape[1])}


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
            inference_timing = getattr(self.adapter, "last_timing", {})
        except Exception as exc:
            latency = time.perf_counter() - start
            log.exception("VLM reasoning failed; routing to stop_and_verify")
            return SkillRecommendation("unknown", "unknown", "stop_and_verify", None,
                                       f"VLM inference failed: {type(exc).__name__}: {exc}", True), latency
        skill = raw.get("recommended_skill") if isinstance(raw, dict) else None
        required_valid = (
            isinstance(skill, str)
            and skill in self.registry.allowed_recommendations()
            and isinstance(raw.get("scene_state"), str)
            and isinstance(raw.get("task_progress"), str)
            and isinstance(raw.get("reason"), str)
            and isinstance(raw.get("requires_verification"), bool)
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
        if inference_timing:
            rec.raw["inference_timing"] = inference_timing
        return rec, latency

