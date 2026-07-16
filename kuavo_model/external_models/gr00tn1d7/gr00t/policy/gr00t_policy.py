# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gr00t Policy implementation for inference.

This module provides the core policy classes for running Gr00t models:
- Gr00tPolicy: Base policy class for model inference
- Gr00tSimPolicyWrapper: Wrapper for compatibility with existing Gr00t simulation environments
"""

import types
import logging
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoProcessor

from gr00t.data.embodiment_tags import FINETUNE_ONLY_TAGS, POSTTRAIN_TAGS, EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.types import MessageType, ModalityConfig, VLAStepData

from .policy import BasePolicy, PolicyWrapper


logger = logging.getLogger(__name__)


def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    """Recursively convert all floating point tensors in a nested structure to the given dtype.

    Args:
        x: Input data structure (tensor, dict, list, or other)
        dtype: Target torch dtype for floating point tensors

    Returns:
        Data structure with floating point tensors converted to target dtype

    Warning:
        Non-floating point tensors will be left as is.
    """
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    # Handle dict-like objects (tianshou.BatchFeature is not dict but has items() method)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}  # type: ignore
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    else:
        return x


def _rec_float_to_fp32(x: Any) -> Any:
    """Recursively convert floating tensors in a nested structure to float32."""
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.float()
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_float_to_fp32(v) for k, v in x.items()}  # type: ignore
    elif isinstance(x, list):
        return [_rec_float_to_fp32(v) for v in x]
    else:
        return x


def _patch_lora_linear_forward(module: torch.nn.Module) -> int:
    patched = 0
    for child in module.modules():
        base_layer = getattr(child, "base_layer", None)
        lora_a = getattr(child, "lora_A", None)
        lora_b = getattr(child, "lora_B", None)
        if base_layer is None or lora_a is None or lora_b is None:
            continue
        if not hasattr(base_layer, "weight") or not hasattr(base_layer, "bias"):
            continue

        def safe_forward(self, x, *args, _base_layer=base_layer, _lora_a=lora_a, _lora_b=lora_b, **kwargs):
            out = F.linear(x, _base_layer.weight, _base_layer.bias)
            active_adapters = getattr(self, "active_adapters", [])
            for adapter in active_adapters:
                if adapter not in _lora_a or adapter not in _lora_b:
                    continue
                lora_out = _lora_b[adapter](_lora_a[adapter](x.to(_lora_a[adapter].weight.dtype)))
                scaling = self.scaling[adapter] if isinstance(self.scaling, dict) else self.scaling
                out = out + lora_out.to(out.dtype) * scaling
            return out

        child.forward = types.MethodType(safe_forward, child)
        patched += 1
    return patched


def _require_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
        return
    finite = torch.isfinite(tensor)
    if bool(finite.all().item()):
        return
    bad_count = int((~finite).sum().item())
    finite_values = tensor[finite]
    stats = ""
    if finite_values.numel() > 0:
        finite_values = finite_values.float()
        stats = (
            f", finite_min={float(finite_values.min().item()):.6g}, "
            f"finite_max={float(finite_values.max().item()):.6g}, "
            f"finite_abs_max={float(finite_values.abs().max().item()):.6g}"
        )
    raise FloatingPointError(
        f"{name} contains NaN/Inf: shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, bad_count={bad_count}{stats}"
    )


def _restore_visual_from_checkpoint(visual: torch.nn.Module, model_dir: Path) -> int:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        logger.warning("Skipped visual checkpoint restore: %s not found", index_path)
        return 0

    weight_map = json.loads(index_path.read_text())["weight_map"]
    shard_cache = {}
    restored = 0

    with torch.no_grad():
        for local_name, target in visual.state_dict().items():
            checkpoint_name = f"backbone.model.model.visual.{local_name}"
            shard_name = weight_map.get(checkpoint_name)
            if shard_name is None:
                continue
            if shard_name not in shard_cache:
                shard_cache[shard_name] = load_file(str(model_dir / shard_name), device="cpu")
            source = shard_cache[shard_name][checkpoint_name].float()
            _require_finite_tensor(f"visual_checkpoint.{checkpoint_name}", source)
            target.copy_(source.to(device=target.device, dtype=target.dtype))
            _require_finite_tensor(f"visual_restored.{local_name}", target)
            restored += 1

    logger.warning("Restored %d visual tensors from checkpoint in CPU fp32 path", restored)
    return restored


def _load_checkpoint_tensor(model_dir: Path, checkpoint_name: str) -> torch.Tensor | None:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return None
    weight_map = json.loads(index_path.read_text())["weight_map"]
    shard_name = weight_map.get(checkpoint_name)
    if shard_name is None:
        return None
    tensors = load_file(str(model_dir / shard_name), device="cpu")
    return tensors[checkpoint_name].float().contiguous()


def _load_visual_lora_scaling(model_dir: Path) -> float:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return 1.0
    config = json.loads(config_path.read_text())
    alpha = config.get("visual_lora_alpha", config.get("lora_alpha", 1))
    rank = config.get("visual_lora_rank", config.get("lora_rank", 1))
    return float(alpha) / float(rank)


def _patch_checkpoint_linear_cpu_forward(
    module: torch.nn.Module,
    model_dir: Path,
    checkpoint_prefix: str,
    log_name: str,
) -> None:
    base_layer = getattr(module, "base_layer", module)
    if not hasattr(base_layer, "weight"):
        logger.warning(
            "Skipped %s CPU patch: module has no weight (%s)",
            log_name,
            type(module).__name__,
        )
        return

    weight_cpu = _load_checkpoint_tensor(model_dir, f"{checkpoint_prefix}.base_layer.weight")
    if weight_cpu is None:
        weight_cpu = base_layer.weight.detach().float().cpu().contiguous()
    bias_cpu = _load_checkpoint_tensor(model_dir, f"{checkpoint_prefix}.base_layer.bias")
    if bias_cpu is None:
        bias = getattr(base_layer, "bias", None)
        bias_cpu = None if bias is None else bias.detach().float().cpu().contiguous()
    lora_a = getattr(module, "lora_A", None)
    lora_b = getattr(module, "lora_B", None)
    lora_a_cpu = _load_checkpoint_tensor(model_dir, f"{checkpoint_prefix}.lora_A.default.weight")
    lora_b_cpu = _load_checkpoint_tensor(model_dir, f"{checkpoint_prefix}.lora_B.default.weight")
    scaling_attr = getattr(module, "scaling", None)
    if isinstance(scaling_attr, dict):
        lora_scaling = scaling_attr.get("default", _load_visual_lora_scaling(model_dir))
    elif scaling_attr is not None:
        lora_scaling = scaling_attr
    else:
        lora_scaling = _load_visual_lora_scaling(model_dir)

    _require_finite_tensor(f"{log_name}.cpu_patch.weight", weight_cpu)
    if bias_cpu is not None:
        _require_finite_tensor(f"{log_name}.cpu_patch.bias", bias_cpu)

    def cpu_forward(self, x, *args, **kwargs):
        _require_finite_tensor(f"{log_name}.cpu_patch.input", x)
        x_cpu = x.detach().float().cpu()
        out = F.linear(x_cpu, weight_cpu, bias_cpu)
        active_adapters = getattr(self, "active_adapters", [])
        if lora_a_cpu is not None and lora_b_cpu is not None:
            out = out + F.linear(F.linear(x_cpu, lora_a_cpu), lora_b_cpu) * lora_scaling
        elif lora_a is not None and lora_b is not None:
            for adapter in active_adapters:
                if adapter not in lora_a or adapter not in lora_b:
                    continue
                adapter_out = lora_b[adapter].cpu().float()(lora_a[adapter].cpu().float()(x_cpu))
                scaling = self.scaling[adapter] if isinstance(self.scaling, dict) else self.scaling
                out = out + adapter_out * scaling
        _require_finite_tensor(f"{log_name}.cpu_patch.output", out)
        return out.to(device=x.device, dtype=x.dtype)

    module.forward = types.MethodType(cpu_forward, module)
    logger.warning(
        "Patched %s to CPU fp32 forward (%s -> %s)",
        log_name,
        type(module).__name__,
        type(base_layer).__name__,
    )


def _patch_visual_attention_cpu_forwards(visual: torch.nn.Module, model_dir: Path) -> None:
    blocks = getattr(visual, "blocks", None)
    if not blocks:
        logger.warning("Skipped visual attention CPU patch: no visual blocks")
        return
    patched = 0
    for idx, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        qkv = getattr(attn, "qkv", None)
        if qkv is not None:
            _patch_checkpoint_linear_cpu_forward(
                qkv,
                model_dir,
                f"backbone.model.model.visual.blocks.{idx}.attn.qkv",
                f"visual.blocks[{idx}].attn.qkv",
            )
            patched += 1
        proj = getattr(attn, "proj", None)
        if proj is not None:
            _patch_checkpoint_linear_cpu_forward(
                proj,
                model_dir,
                f"backbone.model.model.visual.blocks.{idx}.attn.proj",
                f"visual.blocks[{idx}].attn.proj",
            )
            patched += 1
    logger.warning("Patched %d visual attention linear modules to CPU fp32 forward", patched)


# Backward-compatible alias for older local imports during iterative debugging.
_patch_first_visual_attention_cpu_forwards = _patch_visual_attention_cpu_forwards


def _patch_qwen_image_feature_output_dtype(qwen_model: torch.nn.Module) -> None:
    core_model = getattr(qwen_model, "model", qwen_model)
    get_image_features = getattr(core_model, "get_image_features", None)
    if get_image_features is None:
        logger.warning("Skipped Qwen image feature dtype patch: get_image_features not found")
        return

    def cast_tree(value, *, dtype: torch.dtype, device: torch.device):
        if torch.is_tensor(value) and torch.is_floating_point(value):
            return value.to(device=device, dtype=dtype)
        if isinstance(value, tuple):
            return tuple(cast_tree(item, dtype=dtype, device=device) for item in value)
        if isinstance(value, list):
            return [cast_tree(item, dtype=dtype, device=device) for item in value]
        return value

    def patched_get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        output = get_image_features(pixel_values, image_grid_thw, **kwargs)
        embedding_weight = self.get_input_embeddings().weight
        target_dtype = embedding_weight.dtype
        target_device = embedding_weight.device
        if isinstance(output, tuple):
            return cast_tree(output, dtype=target_dtype, device=target_device)
        output.pooler_output = cast_tree(output.pooler_output, dtype=target_dtype, device=target_device)
        if getattr(output, "deepstack_features", None) is not None:
            output.deepstack_features = cast_tree(
                output.deepstack_features,
                dtype=target_dtype,
                device=target_device,
            )
        return output

    core_model.get_image_features = types.MethodType(patched_get_image_features, core_model)
    logger.warning("Patched Qwen image feature outputs to language embedding dtype")


class Gr00tPolicy(BasePolicy):
    """Core policy class for Gr00t model inference.

    This policy handles the end-to-end inference pipeline:
    1. Validates input observations
    2. Processes observations with pretrained VLA processor
    3. Runs model inference
    4. Decodes and returns actions

    The policy expects observations with specific modalities (video, state, language)
    and returns actions in the format defined by the model's modality configuration.
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag | str,
        model_path: str,
        *,
        device: int | str,
        strict: bool = True,
        use_bf16: bool = True,
        use_fp16: bool = False,
        action_head_fp32: bool = False,
        disable_flash_attention: bool = False,
        visual_fp32: bool = False,
    ):
        """Initialize the Gr00t Policy.

        Args:
            embodiment_tag: The embodiment tag defining the robot/environment type.
                Accepts an EmbodimentTag enum or a string (resolved case-insensitively).
            model_path: Path to the pretrained model checkpoint directory
            device: Device to run the model on (e.g., 'cuda:0', 0, 'cpu')
            strict: Whether to enforce strict input validation (default: True)
            use_bf16: Whether to move model weights to bfloat16 for inference.
            use_fp16: Whether to move model weights to float16 for inference.
            action_head_fp32: Whether to keep the action head and diffusion sampling in fp32.
            disable_flash_attention: Whether to disable flash attention in the backbone.
            visual_fp32: Whether to keep only the vision encoder in fp32.
        """
        # Import this to register all models.
        import gr00t.model  # noqa: F401

        super().__init__(strict=strict)
        if isinstance(embodiment_tag, str):
            embodiment_tag = EmbodimentTag.resolve(embodiment_tag)
        model_dir = Path(model_path)

        # Load the pretrained model and move to target device with bfloat16 precision.
        model_config = AutoConfig.from_pretrained(model_dir)
        if disable_flash_attention and hasattr(model_config, "use_flash_attention"):
            model_config.use_flash_attention = False
            model_config._attn_implementation = "eager"
        if (use_fp16 or not use_bf16) and hasattr(model_config, "load_bf16"):
            model_config.load_bf16 = False
        loading_kwargs = {"attn_implementation": "eager"} if disable_flash_attention else {}
        model = AutoModel.from_pretrained(model_dir, config=model_config, **loading_kwargs)
        model.eval()  # Set model to evaluation mode
        if use_fp16:
            inference_dtype = torch.float16
        elif use_bf16:
            inference_dtype = torch.bfloat16
        else:
            inference_dtype = torch.float32
        self.inference_dtype = inference_dtype
        model.to(device=device, dtype=inference_dtype)
        if visual_fp32:
            backbone = getattr(model, "backbone", None)
            qwen_model = getattr(backbone, "model", None)
            visual = getattr(qwen_model, "visual", None)
            if visual is None:
                raise AttributeError("visual_fp32 requested but Qwen visual encoder was not found")
            visual.to(device=device, dtype=torch.float32)
            _restore_visual_from_checkpoint(visual, model_dir)
            _patch_lora_linear_forward(visual)
            _patch_visual_attention_cpu_forwards(visual, model_dir)
            _patch_qwen_image_feature_output_dtype(qwen_model)
            backbone.force_visual_fp32 = True
        if action_head_fp32 and hasattr(model, "action_head"):
            model.action_head.to(device=device, dtype=torch.float32)
            model.force_action_head_fp32 = True
        self.model = model

        # Load the processor for input/output transformation.
        # Training saves processor files under a "processor/" subdirectory, but
        # AutoProcessor expects them at the model root.  Fall back to the
        # subdirectory when the root lacks a processor_config.json.
        processor_dir = (
            model_dir / "processor"
            if (model_dir / "processor").is_dir()
            and not (model_dir / "processor_config.json").exists()
            else model_dir
        )
        self.processor: BaseProcessor = AutoProcessor.from_pretrained(processor_dir)
        self.processor.eval()

        # Store embodiment-specific configurations
        self.embodiment_tag = embodiment_tag
        all_modality_configs = self.processor.get_modality_configs()
        if self.embodiment_tag.value not in all_modality_configs:
            # Map raw checkpoint tag values to user-friendly enum names where possible.
            supported_lines = []
            for tag_value in sorted(all_modality_configs.keys()):
                enum_name = EmbodimentTag.reverse_lookup(tag_value)
                if enum_name != tag_value:
                    supported_lines.append(f"  {enum_name:30s} (--embodiment-tag {enum_name})")
                else:
                    supported_lines.append(f"  {tag_value:30s} (internal, no public enum)")
            supported_str = "\n".join(supported_lines)

            hint = ""
            if self.embodiment_tag in POSTTRAIN_TAGS:
                hint = (
                    f"\n\nHint: '{self.embodiment_tag.name}' is a posttrain tag that requires "
                    f"a finetuned checkpoint, not the base model. "
                    f"See the example READMEs for how to finetune and download checkpoints."
                )
            elif self.embodiment_tag in FINETUNE_ONLY_TAGS:
                hint = (
                    f"\n\nHint: '{self.embodiment_tag.name}' is for finetuning custom robots. "
                    f"Use it with launch_finetune.py, not with the base model directly."
                )

            raise ValueError(
                f"Embodiment tag '{self.embodiment_tag.name}' "
                f"(value='{self.embodiment_tag.value}') is not supported "
                f"by this checkpoint.\n\n"
                f"Supported tags in this checkpoint:\n{supported_str}"
                f"{hint}"
            )
        self.modality_configs = {
            k: v
            for k, v in all_modality_configs[self.embodiment_tag.value].items()
            if k != "rl_info"
        }
        self.collate_fn = self.processor.collator

        # Extract and validate language configuration
        # Some embodiments (e.g. OXE_DROID) define multiple language keys for
        # training-time augmentation (paraphrases). At inference we only use the first key.
        language_keys = self.modality_configs["language"].modality_keys
        language_delta_indices = self.modality_configs["language"].delta_indices
        assert len(language_keys) >= 1, "At least one language key is required"
        assert len(language_delta_indices) == 1, "Only one language delta index is supported"
        self.language_key = language_keys[0]

    def _unbatch_observation(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        """Unbatch a batched observation into a list of single observations.

        Args:
            value: Batched observation with shape (B, ...) for each modality

        Returns:
            List of B observations, each with the batch dimension removed
        """
        unbatched_obs = []
        # Infer batch size from the first video key
        batch_size = value["video"][list(value["video"].keys())[0]].shape[0]

        # Split each modality along the batch dimension
        for i in range(batch_size):
            unbatched_value = {
                "video": {k: v[i] for k, v in value["video"].items()},
                "state": {k: v[i] for k, v in value["state"].items()},
                "language": {k: v[i] for k, v in value["language"].items()},
            }
            unbatched_obs.append(unbatched_value)
        return unbatched_obs

    def _to_vla_step_data(self, observation: dict[str, Any]) -> VLAStepData:
        """Convert a single observation into a VLAStepData object for processing.

        Args:
            observation: Single observation dict with video, state, and language

        Returns:
            VLAStepData object ready for processor input
        """
        return VLAStepData(
            images=observation["video"],
            states=observation["state"],
            actions={},  # No ground truth actions during inference
            text=observation["language"][self.language_key][0],
            embodiment=self.embodiment_tag,
        )

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate that the observation has the correct structure and types.

        This method ensures that all required modalities are present and that their
        data types, shapes, and dimensions match the model's expectations.

        Expected observation structure:
            - video: dict[str, np.ndarray[np.uint8, (B, T, H, W, C)]]
                - B: batch size
                - T: temporal horizon (number of frames)
                - H, W: image height and width
                - C: number of channels (must be 3 for RGB)
            - state: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: temporal horizon (number of state observations)
                - D: state dimension
            - language: dict[str, list[list[str]]]
                - Shape: (B, T) where each element is a string
                - T: temporal horizon (typically 1 for language)

        Args:
            observation: Dictionary containing video, state, and language modalities

        Raises:
            AssertionError: If any validation check fails
        """
        # Check that observation contains all required top-level modality keys
        for modality in ["video", "state", "language"]:
            assert modality in observation, f"Observation must contain a '{modality}' key"
            assert isinstance(observation[modality], dict), (
                f"Observation '{modality}' must be a dictionary. Got {type(observation[modality])}: {observation[modality]}"
            )

        # Track batch size across modalities to ensure consistency
        bs = -1

        # ===== VIDEO VALIDATION =====
        # Validate each video stream defined in the modality config
        for video_key in self.modality_configs["video"].modality_keys:
            assert video_key in observation["video"], (
                f"Video key '{video_key}' must be in observation"
            )

            # Set or verify batch size consistency across all video keys
            if bs == -1:
                bs = len(observation["video"][video_key])
            else:
                assert len(observation["video"][video_key]) == bs, (
                    f"Video key '{video_key}' must have batch size {bs}. Got {len(observation['video'][video_key])}"
                )

            batched_video = observation["video"][video_key]

            # Verify data type is numpy array
            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"
            )

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(self.modality_configs["video"].delta_indices), (
                f"Video key '{video_key}'s horizon must be {len(self.modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"
            )

            # Verify channel dimension is 3 (RGB images)
            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ===== STATE VALIDATION =====
        # Validate each state stream defined in the modality config
        for state_key in self.modality_configs["state"].modality_keys:
            # Check that the expected state key exists in the observation
            # (must happen before indexing — see video validation above)
            assert state_key in observation["state"], (
                f"State key '{state_key}' must be in observation"
            )

            # Set or verify batch size consistency across all state keys
            if bs == -1:
                bs = len(observation["state"][state_key])
            else:
                assert len(observation["state"][state_key]) == bs, (
                    f"State key '{state_key}' must have batch size {bs}. Got {len(observation['state'][state_key])}"
                )

            batched_state = observation["state"][state_key]

            # Verify data type is numpy array
            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )

            # Verify dtype is float32 (standard for continuous state values)
            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(self.modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be {len(self.modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"
            )

        # ===== LANGUAGE VALIDATION =====
        # Validate each language stream defined in the modality config
        for language_key in self.modality_configs["language"].modality_keys:
            # Check that the expected language key exists in the observation
            # (must happen before indexing — see video validation above)
            assert language_key in observation["language"], (
                f"Language key '{language_key}' must be in observation"
            )

            # Set or verify batch size consistency (language uses len instead of .shape)
            if bs == -1:
                bs = len(observation["language"][language_key])
            else:
                assert len(observation["language"][language_key]) == bs, (
                    f"Language key '{language_key}' must have batch size {bs}. Got {len(observation['language'][language_key])}"
                )

            batched_language: list[list[str]] = observation["language"][language_key]

            # Verify outer structure is a list (batch dimension)
            assert isinstance(batched_language, list), (
                f"Language key '{language_key}' must be a list. Got {type(batched_language)}"
            )

            # Validate each batch item
            for batch_item in batched_language:
                # Verify temporal dimension matches expected horizon
                assert len(batch_item) == len(self.modality_configs["language"].delta_indices), (
                    f"Language key '{language_key}'s horizon must be {len(self.modality_configs['language'].delta_indices)}. Got {len(batched_language)}"
                )

                # Verify inner structure is also a list (temporal dimension)
                assert isinstance(batch_item, list), (
                    f"Language batch item must be a list. Got {type(batch_item)}"
                )

                # Current implementation expects exactly one language instruction per timestep
                assert len(batch_item) == 1, (
                    f"Language batch item must have exactly one item. Got {len(batch_item)}"
                )

                # Verify the instruction itself is a string
                assert isinstance(batch_item[0], str), (
                    f"Language batch item must be a string. Got {type(batch_item[0])}"
                )

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Internal method to compute actions from observations.

        Pipeline:
        1. Unbatch observations into individual samples
        2. Convert each to VLAStepData and process
        3. Collate into model input batch
        4. Run model inference
        5. Decode and unnormalize actions

        Args:
            observation: Batched observation dictionary
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (actions_dict, info_dict)
        """
        # Step 1: Split batched observation into individual observations
        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []

        # Step 2: Process each observation through the VLA processor
        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)  # dict[str, np.ndarray[np.float32, (T, D)]]
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        # Step 3: Collate processed inputs into a single batch for model
        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=self.inference_dtype)

        # Step 4: Run model inference to predict actions
        with torch.inference_mode():
            model_pred = self.model.get_action(**collated_inputs)
        normalized_action = model_pred["action_pred"].float()

        # Step 5: Decode actions from normalized space back to physical units
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)  # (B, T, D)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        # Cast all actions to float32 for consistency
        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }
        return casted_action, {}

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate that the action has the correct structure and types.

        This method ensures that all required action keys are present and that their
        data types, shapes, and dimensions match the model's action space.

        Expected action structure:
            - action: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension (e.g., joint positions, velocities, gripper state)

        Args:
            action: Dictionary containing action arrays for each action key

        Raises:
            AssertionError: If any validation check fails
        """
        # Validate each action key defined in the modality config
        for action_key in self.modality_configs["action"].modality_keys:
            # Check that the expected action key exists
            assert action_key in action, f"Action key '{action_key}' must be in action"

            action_arr = action[action_key]

            # Verify data type is numpy array
            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )

            # Verify dtype is float32 (standard for continuous actions)
            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"
            )

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(self.modality_configs["action"].delta_indices), (
                f"Action key '{action_key}'s horizon must be {len(self.modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"
            )

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.modality_configs

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy to its initial state.

        Args:
            options: Dictionary containing the options for the reset

        Returns:
            Dictionary containing the info after resetting the policy
        """
        return {}


class Gr00tSimPolicyWrapper(PolicyWrapper):
    """Wrapper for Gr00tPolicy to enable compatibility with existing Gr00t simulation environments.

    This wrapper is specifically designed for retro-fitting the Gr00t policy with the current
    Gr00t simulation environment interface. It handles the transformation between the flat
    observation format used by Gr00t sim environments (with keys like 'video.camera_name',
    'state.joint_positions') and the nested format expected by Gr00tPolicy.

    **Important**: If you are using other environments, custom robots, or building new environments,
    you should use `Gr00tPolicy` directly and format your observations according to its interface.
    This wrapper is only needed for compatibility with the existing Gr00t sim infrastructure.

    Key transformations performed by this wrapper:
    - Observation keys: 'video.cam' -> observation['video']['cam']
    - Observation keys: 'state.joints' -> observation['state']['joints']
    - Language keys: 'task' or 'annotation.human.coarse_action' -> observation['language']['task']
    - Action keys: action['joints'] -> 'action.joints'
    """

    def __init__(self, policy: Gr00tPolicy, *, strict: bool = True):
        """Initialize the wrapper around a Gr00tPolicy instance.

        Args:
            policy: The Gr00tPolicy instance to wrap
            strict: Whether to enforce strict validation (default: True)
        """
        super().__init__(policy, strict=strict)
        self.policy: Gr00tPolicy = policy
        assert len(self.policy.modality_configs["language"].delta_indices) == 1, (
            "Only one language delta index is supported"
        )

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate observation from Gr00t sim environment format.

        This validation is specific to the flat observation format used by Gr00t sim environments.
        Unlike Gr00tPolicy.check_observation which expects nested dicts, this expects flat keys.

        Expected observation structure (Gr00t sim format):
            - Flat keys like 'video.camera_name': np.ndarray[np.uint8, (B, T, H, W, C)]
            - Flat keys like 'state.state_name': np.ndarray[np.float32, (B, T, D)]
            - Language keys: tuple[str] or list[str] with shape (B,)
                - Key can be 'task' or 'annotation.human.coarse_action' (for DC envs)

        Args:
            observation: Flat observation dictionary from Gr00t sim environment

        Raises:
            AssertionError: If any validation check fails
        """
        modality_configs = self.get_modality_config()

        # ===== VIDEO VALIDATION =====
        # Check video modalities with flat key format: 'video.camera_name'
        for video_key in modality_configs["video"].modality_keys:
            # Construct flat key expected in Gr00t sim environment
            parsed_key = f"video.{video_key}"
            assert parsed_key in observation, f"Video key '{parsed_key}' must be in observation"

            batched_video = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"
            )

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(modality_configs["video"].delta_indices), (
                f"Video key '{video_key}'s horizon must be {len(modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"
            )

            # Verify channel dimension is 3 (RGB images)
            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ===== STATE VALIDATION =====
        # Check state modalities with flat key format: 'state.state_name'
        for state_key in modality_configs["state"].modality_keys:
            # Construct flat key expected in Gr00t sim environment
            parsed_key = f"state.{state_key}"
            assert parsed_key in observation, f"State key '{parsed_key}' must be in observation"

            batched_state = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )

            # Verify dtype is float32 (standard for continuous state values)
            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be {len(modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"
            )

        # ===== LANGUAGE VALIDATION =====
        # Check language modalities (special handling for DC environment compatibility)
        for language_key in modality_configs["language"].modality_keys:
            # PATCH: Legacy compatibility for DC environments
            # DC envs use 'annotation.human.coarse_action' instead of 'task'
            if language_key == "task" and "annotation.human.coarse_action" in observation:
                language_key = "annotation.human.coarse_action"
            # /PATCH

            # Check that the expected language key exists
            assert language_key in observation, (
                f"Language key '{language_key}' must be in observation"
            )

            # In Gr00t sim format, language is a tuple of strings (B,)
            batched_language: tuple[str] | list[str] = observation[language_key]  # (B,)

            # Verify outer structure is a tuple (batch dimension)
            assert isinstance(batched_language, (tuple, list)), (
                f"Language key '{language_key}' must be a tuple or list. Got {type(batched_language)}"
            )

            # Verify each batch item is a string
            assert isinstance(batched_language[0], str), (
                f"Language batch item must be a string. Got {type(batched_language[0])}"
            )

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Transform Gr00t sim observation format and compute actions.

        This method transforms the flat observation format from Gr00t sim environments
        into the nested format expected by Gr00tPolicy, computes actions, and transforms
        them back to the flat format expected by Gr00t sim environments.

        Input format (Gr00t sim):
            - Flat keys: 'video.camera_name', 'state.state_name'
            - Language: tuple[str] (B,)

        Output format (Gr00t sim):
            - Flat keys: 'action.action_name'

        Args:
            observation: Flat observation dictionary from Gr00t sim environment
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (flat_actions_dict, info_dict)
        """
        # Transform flat observation format to nested format expected by Gr00tPolicy
        new_obs = {}
        for modality in ["video", "state", "language"]:
            new_obs[modality] = {}
            for key in self.policy.modality_configs[modality].modality_keys:
                if modality == "language":
                    # PATCH: Legacy compatibility for DC environments
                    if key == "task" and "annotation.human.coarse_action" in observation:
                        parsed_key = "annotation.human.coarse_action"
                    # /PATCH
                    else:
                        parsed_key = key
                else:
                    # Construct flat key (e.g., 'video.camera' or 'state.joints')
                    parsed_key = f"{modality}.{key}"

                arr = observation[parsed_key]

                # Transform to nested format
                if modality == "language":
                    # Convert from tuple[str] or list[str] (B,) to list[list[str]] (B, 1)
                    # Each element becomes a list with one string for temporal dimension
                    new_obs[modality][key] = [[str(item)] for item in arr]
                else:
                    # Video and state arrays are already in correct format (B, T, ...)
                    new_obs[modality][key] = arr

        # Compute actions using the underlying Gr00tPolicy
        action, info = self.policy.get_action(new_obs, options)

        # Transform actions back to flat format for Gr00t sim environment
        # action['joints'] -> 'action.joints'
        return {f"action.{key}": action[key] for key in action}, info

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate action in Gr00t sim environment format.

        This validation is specific to the flat action format used by Gr00t sim environments.
        Unlike Gr00tPolicy.check_action which expects nested dicts, this expects flat keys.

        Expected action structure (Gr00t sim format):
            - Flat keys like 'action.action_name': np.ndarray[np.float32, (B, T, D)]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension

        Args:
            action: Flat action dictionary for Gr00t sim environment

        Raises:
            AssertionError: If any validation check fails
        """
        modality_configs = self.get_modality_config()

        # Validate each action key defined in the modality config
        for action_key in modality_configs["action"].modality_keys:
            # Construct flat key expected in Gr00t sim environment (e.g., 'action.joints')
            parsed_key = f"action.{action_key}"
            assert parsed_key in action, f"Action key '{parsed_key}' must be in action"

            action_arr = action[parsed_key]

            # Verify data type is numpy array
            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )

            # Verify dtype is float32 (standard for continuous actions)
            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"
            )

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(modality_configs["action"].delta_indices), (
                f"Action key '{action_key}'s horizon must be {len(modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"
            )

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        """Get the modality configuration from the underlying policy.

        Returns:
            Dictionary mapping modality names to their configurations
        """
        return self.policy.get_modality_config()
