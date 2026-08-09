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

import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from gr00t.configs.base_config import Config
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.experiment.dist_utils import get_rank
from gr00t.model.base.model_pipeline import ModelPipeline
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from gr00t.model.registry import register_model


# Convert tensors to lists for JSON serialization
def convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists in nested dictionaries/lists."""
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


class Gr00tN1d7Pipeline(ModelPipeline):
    model_class = Gr00tN1d7
    processor_class = Gr00tN1d7Processor

    def __init__(self, config: Config, save_cfg_dir: Path):
        super().__init__(config)
        self.save_cfg_dir = save_cfg_dir

        # Build transformers loading kwargs from training config
        transformers_loading_kwargs = {
            "trust_remote_code": self.config.training.transformers_trust_remote_code,
            "local_files_only": self.config.training.transformers_local_files_only,
        }
        if self.model_config.model_revision is not None:
            transformers_loading_kwargs["revision"] = self.model_config.model_revision
        if self.config.training.transformers_cache_dir is not None:
            transformers_loading_kwargs["cache_dir"] = self.config.training.transformers_cache_dir
        if self.config.training.transformers_access_token is not None:
            transformers_loading_kwargs["token"] = self.config.training.transformers_access_token

        self.transformers_loading_kwargs = transformers_loading_kwargs

    @property
    def model_config(self):
        return self.config.model

    def setup(self):
        self.model = self._create_model()
        # AutoModel.from_pretrained keeps normalization metadata from the base
        # checkpoint in model.config. Keep it aligned with the processor used by
        # this fine-tuning run so saved checkpoint config.json is authoritative.
        self.model.config.use_percentiles = self.model_config.use_percentiles
        self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
        self.data_collator = self._create_collator()

    def _create_model(self):
        """Setup model with proper vocabulary expansion."""
        skip_weight_loading = getattr(self.config.training, "skip_weight_loading", False)
        if self.config.training.start_from_checkpoint is not None and not skip_weight_loading:
            checkpoint_config = Gr00tN1d7Config.from_pretrained(
                self.config.training.start_from_checkpoint,
                **self.transformers_loading_kwargs,
            )
            checkpoint_has_visual_lora = getattr(
                checkpoint_config, "use_visual_lora", False
            )
            checkpoint_has_diffusion_lora = getattr(
                checkpoint_config, "use_diffusion_lora", False
            )
            enable_visual_lora = (
                self.config.model.use_visual_lora or checkpoint_has_visual_lora
            )
            enable_diffusion_lora = (
                self.config.model.use_diffusion_lora or checkpoint_has_diffusion_lora
            )
            visual_lora_rank = (
                checkpoint_config.visual_lora_rank
                if checkpoint_has_visual_lora
                else self.config.model.visual_lora_rank
            )
            visual_lora_alpha = (
                checkpoint_config.visual_lora_alpha
                if checkpoint_has_visual_lora
                else self.config.model.visual_lora_alpha
            )
            visual_lora_dropout = (
                checkpoint_config.visual_lora_dropout
                if checkpoint_has_visual_lora
                else self.config.model.visual_lora_dropout
            )
            visual_lora_targets = (
                checkpoint_config.visual_lora_target_modules
                if checkpoint_has_visual_lora
                else self.config.model.visual_lora_target_modules
            )
            diffusion_lora_rank = (
                checkpoint_config.diffusion_lora_rank
                if checkpoint_has_diffusion_lora
                else self.config.model.diffusion_lora_rank
            )
            diffusion_lora_alpha = (
                checkpoint_config.diffusion_lora_alpha
                if checkpoint_has_diffusion_lora
                else self.config.model.diffusion_lora_alpha
            )
            diffusion_lora_dropout = (
                checkpoint_config.diffusion_lora_dropout
                if checkpoint_has_diffusion_lora
                else self.config.model.diffusion_lora_dropout
            )
            diffusion_lora_targets = (
                checkpoint_config.diffusion_lora_target_modules
                if checkpoint_has_diffusion_lora
                else self.config.model.diffusion_lora_target_modules
            )

            # Recreate adapters before loading only when they already exist in
            # the checkpoint. New adapters are injected after the base weights
            # load, avoiding PEFT's base_layer key renaming mismatch.
            model, loading_info = AutoModel.from_pretrained(
                self.config.training.start_from_checkpoint,
                tune_llm=self.config.model.tune_llm,
                tune_visual=self.config.model.tune_visual,
                use_visual_lora=checkpoint_has_visual_lora,
                visual_lora_rank=visual_lora_rank,
                visual_lora_alpha=visual_lora_alpha,
                visual_lora_dropout=visual_lora_dropout,
                visual_lora_target_modules=visual_lora_targets,
                tune_projector=self.config.model.tune_projector,
                tune_diffusion_model=self.config.model.tune_diffusion_model,
                use_diffusion_lora=checkpoint_has_diffusion_lora,
                diffusion_lora_rank=diffusion_lora_rank,
                diffusion_lora_alpha=diffusion_lora_alpha,
                diffusion_lora_dropout=diffusion_lora_dropout,
                diffusion_lora_target_modules=diffusion_lora_targets,
                tune_vlln=self.config.model.tune_vlln,
                state_dropout_prob=self.config.model.state_dropout_prob,
                backbone_trainable_params_fp32=self.config.model.backbone_trainable_params_fp32,
                load_bf16=self.config.model.load_bf16,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                output_loading_info=True,
                **self.transformers_loading_kwargs,
            )

            missing_keys = loading_info.get("missing_keys", [])
            mask_token_missing = any("mask_token" in key for key in missing_keys)
            if mask_token_missing and model.action_head.mask_token is not None:
                with torch.no_grad():
                    model.action_head.mask_token.data.copy_(
                        0.02 * torch.randn_like(model.action_head.mask_token)
                    )
                logging.info("mask_token not in checkpoint - initialized")

            unexpected_keys = loading_info.get("unexpected_keys", [])
            mismatched_keys = loading_info.get("mismatched_keys", [])
            other_missing = [key for key in missing_keys if "mask_token" not in key]
            errors = []
            if other_missing:
                errors.append(f"Missing keys ({len(other_missing)}): {other_missing}")
            if unexpected_keys:
                errors.append(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}")
            if mismatched_keys:
                errors.append(f"Mismatched keys ({len(mismatched_keys)}): {mismatched_keys}")
            if errors:
                raise RuntimeError(
                    "Checkpoint weight mismatch for "
                    f"{self.config.training.start_from_checkpoint}:\n" + "\n".join(errors)
                )

            if enable_visual_lora and not checkpoint_has_visual_lora:
                model.backbone.enable_visual_lora(
                    rank=visual_lora_rank,
                    alpha=visual_lora_alpha,
                    dropout=visual_lora_dropout,
                    target_modules=visual_lora_targets,
                )
                if self.config.model.backbone_trainable_params_fp32:
                    for name, parameter in model.backbone.named_parameters():
                        if parameter.requires_grad and "lora_" in name:
                            parameter.data = parameter.data.to(torch.float32)
                logging.info("Initialized visual LoRA after loading base checkpoint")

            if enable_diffusion_lora and not checkpoint_has_diffusion_lora:
                model.action_head.enable_diffusion_lora()
                logging.info("Initialized diffusion LoRA after loading base checkpoint")

            model.config.use_visual_lora = enable_visual_lora
            model.config.use_diffusion_lora = enable_diffusion_lora
            self.config.model.use_visual_lora = enable_visual_lora
            self.config.model.use_diffusion_lora = enable_diffusion_lora
            self.config.model.visual_lora_rank = visual_lora_rank
            self.config.model.visual_lora_alpha = visual_lora_alpha
            self.config.model.visual_lora_dropout = visual_lora_dropout
            self.config.model.visual_lora_target_modules = visual_lora_targets
            self.config.model.diffusion_lora_rank = diffusion_lora_rank
            self.config.model.diffusion_lora_alpha = diffusion_lora_alpha
            self.config.model.diffusion_lora_dropout = diffusion_lora_dropout
            self.config.model.diffusion_lora_target_modules = diffusion_lora_targets

        else:
            model = self.model_class(
                self.config.model,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        logging.debug(f"Model Config: {model.config}")
        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_model_config.json", "w") as f:
                f.write(model.config.to_filtered_json())
        # Print parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        logging.debug(f"Model architecture: {model}")

        return model

    def _get_statistics(
        self,
    ) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None:
        return None

    def _get_embodiment_id_mapping(self) -> dict[str, int]:
        return None

    def _create_dataset(self, save_cfg_dir: Path):
        """Create appropriate dataset based on task and mode."""
        if self.config.training.start_from_checkpoint is not None:
            processor = AutoProcessor.from_pretrained(
                self.config.training.start_from_checkpoint,
                # Overrides
                modality_configs=self.config.data.modality_configs,
                use_percentiles=self.model_config.use_percentiles,
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                use_alternate_vl_dit=self.model_config.use_alternate_vl_dit,
                use_relative_action=self.model_config.use_relative_action,
                # State augmentation overrides
                exclude_state=self.model_config.exclude_state,
                state_dropout_prob=self.model_config.state_dropout_prob,
                use_mean_std=self.model_config.use_mean_std,
                **self.transformers_loading_kwargs,
            )
        else:
            processor = self.processor_class(
                modality_configs=self.config.data.modality_configs,
                use_percentiles=self.model_config.use_percentiles,
                statistics=self._get_statistics(),  # By default is None, so this will be computed and set later.
                embodiment_id_mapping=self._get_embodiment_id_mapping(),  # By default is None, so this will be set later.
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                max_state_dim=self.model_config.max_state_dim,
                max_action_dim=self.model_config.max_action_dim,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                use_relative_action=self.model_config.use_relative_action,
                # State augmentation
                exclude_state=self.model_config.exclude_state,
                state_dropout_prob=self.model_config.state_dropout_prob,
                use_mean_std=self.model_config.use_mean_std,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        if processor.use_percentiles != self.model_config.use_percentiles:
            raise RuntimeError(
                "Normalization configuration mismatch: "
                f"processor.use_percentiles={processor.use_percentiles}, "
                f"model_config.use_percentiles={self.model_config.use_percentiles}"
            )

        logging.debug(
            f"Processor configs for training: {json.dumps({k: str(v) for k, v in vars(processor).items()}, indent=2)}"
        )
        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_processor_config.json", "w") as f:
                json.dump({k: str(v) for k, v in vars(processor).items()}, f, indent=2)

        self.processor = processor
        dataset_factory = DatasetFactory(config=self.config)
        train_dataset, eval_dataset = dataset_factory.build(processor=self.processor)

        # Save dataset statistics for inference
        stats = train_dataset.get_dataset_statistics()
        stats_dict = convert_tensors_to_lists(stats)
        # Save statistics
        with open(save_cfg_dir / "dataset_statistics.json", "w") as f:
            json.dump(stats_dict, f, indent=2)
        logging.info("Saved dataset statistics for inference")

        return train_dataset, eval_dataset

    def _create_collator(self):
        data_collator = self.processor.collator
        return data_collator


register_model(Gr00tN1d7Config, Gr00tN1d7Pipeline)
