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

import logging

import torch
from transformers.feature_extraction_utils import BatchFeature


logger = logging.getLogger(__name__)


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


def _require_finite_tree(name: str, value) -> None:
    if torch.is_tensor(value):
        _require_finite_tensor(name, value)
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _require_finite_tree(f"{name}[{idx}]", item)
    elif isinstance(value, dict):
        for key, item in value.items():
            _require_finite_tree(f"{name}.{key}", item)


def _make_finite_hook(name: str):
    return lambda _module, _inputs, output: _require_finite_tree(name, output)


def _make_finite_pre_hook(name: str):
    return lambda _module, inputs: _require_finite_tree(name, inputs)


try:
    from transformers import Qwen3VLForConditionalGeneration

    _QWEN3VL_AVAILABLE = True
except ImportError:
    _QWEN3VL_AVAILABLE = False


class Qwen3Backbone(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = True,
        use_flash_attention: bool = False,
        projector_dim: int = -1,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        transformers_loading_kwargs: dict = {},
    ):
        """
        Qwen3Backbone is to generate n_queries to represent the future action hidden states.
        Args:
            model_name: nvidia/Cosmos-Reason2-2B
            tune_llm: whether to tune the LLM model (default: False)
            tune_visual: whether to tune the visual model (default: False)
        """
        if not _QWEN3VL_AVAILABLE:
            raise ImportError(
                "Qwen3VLForConditionalGeneration is not available. "
                "Please upgrade transformers to a version that supports Qwen3-VL: "
                "pip install transformers>=4.57.0"
            )

        super().__init__()

        # Add attention kwargs
        extra_kwargs = {}
        if use_flash_attention:
            try:
                import flash_attn  # noqa: F401

                extra_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                logger.warning(
                    "flash_attn is not installed. Falling back to sdpa attention. "
                    "Install flash-attn for better performance: pip install flash-attn"
                )
                extra_kwargs["attn_implementation"] = "sdpa"
        if load_bf16:
            extra_kwargs["torch_dtype"] = torch.bfloat16

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            **extra_kwargs,
            **transformers_loading_kwargs,
        ).eval()

        # needed since we don't use these layers. Also saves compute
        while len(self.model.language_model.layers) > select_layer:
            self.model.language_model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)
        if load_bf16 and trainable_params_fp32:
            # cast trainable parameters to fp32
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
                    logger.debug(f"Casting trainable parameter {n} to fp32")

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool, tune_top_llm_layers: int):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.model.language_model.requires_grad_(False)
        if not tune_visual:
            self.model.visual.requires_grad_(False)

        if tune_top_llm_layers > 0:
            for layer in self.model.language_model.layers[-tune_top_llm_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        logger.debug(f"Tune backbone llm: {self.tune_llm}")
        logger.debug(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, log a warning.
        for name, p in self.named_parameters():
            if p.requires_grad:
                logger.debug(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.model.language_model and not self.tune_llm:
                self.model.language_model.eval()
            if self.model.visual and not self.tune_visual:
                self.model.visual.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        # 0. Set frozen module to eval
        keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        vl_input = {k: vl_input[k] for k in keys_to_use}
        if getattr(self, "force_visual_fp32", False):
            vl_input["pixel_values"] = vl_input["pixel_values"].float()
        for key, value in vl_input.items():
            _require_finite_tensor(f"qwen3_backbone.input.{key}", value)
        visual = self.model.visual
        hooks = [
            visual.patch_embed.register_forward_hook(_make_finite_hook("qwen3_backbone.visual.patch_embed.output")),
            visual.merger.register_forward_hook(_make_finite_hook("qwen3_backbone.visual.merger.output")),
            visual.register_forward_hook(_make_finite_hook("qwen3_backbone.visual.output")),
        ]
        if len(visual.blocks) > 0:
            block0 = visual.blocks[0]
            hooks.extend(
                [
                    block0.norm1.register_forward_hook(
                        _make_finite_hook("qwen3_backbone.visual.blocks[0].norm1.output")
                    ),
                    block0.attn.qkv.register_forward_pre_hook(
                        _make_finite_pre_hook("qwen3_backbone.visual.blocks[0].attn.qkv.input")
                    ),
                    block0.attn.qkv.register_forward_hook(
                        _make_finite_hook("qwen3_backbone.visual.blocks[0].attn.qkv.output")
                    ),
                    block0.attn.proj.register_forward_hook(
                        _make_finite_hook("qwen3_backbone.visual.blocks[0].attn.proj.output")
                    ),
                    block0.attn.register_forward_hook(
                        _make_finite_hook("qwen3_backbone.visual.blocks[0].attn.output")
                    ),
                    block0.norm2.register_forward_hook(
                        _make_finite_hook("qwen3_backbone.visual.blocks[0].norm2.output")
                    ),
                    block0.mlp.register_forward_hook(
                        _make_finite_hook("qwen3_backbone.visual.blocks[0].mlp.output")
                    ),
                ]
            )
            qkv_base_layer = getattr(block0.attn.qkv, "base_layer", None)
            if qkv_base_layer is not None:
                hooks.append(
                    qkv_base_layer.register_forward_hook(
                        _make_finite_hook("qwen3_backbone.visual.blocks[0].attn.qkv.base_layer.output")
                    )
                )
        hooks.extend(
            block.register_forward_hook(_make_finite_hook(f"qwen3_backbone.visual.blocks[{idx}].output"))
            for idx, block in enumerate(visual.blocks)
        )
        try:
            outputs = self.model(**vl_input, output_hidden_states=True)
        finally:
            for hook in hooks:
                hook.remove()
        for idx, hidden_state in enumerate(outputs.hidden_states):
            _require_finite_tensor(f"qwen3_backbone.output.hidden_states[{idx}]", hidden_state)
        outputs = outputs.hidden_states[-1]
        image_mask = vl_input["input_ids"] == self.model.config.image_token_id
        attention_mask = vl_input["attention_mask"] == 1
        return BatchFeature(
            data={
                "backbone_features": outputs,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )  # [B, T2, hidden_size]
