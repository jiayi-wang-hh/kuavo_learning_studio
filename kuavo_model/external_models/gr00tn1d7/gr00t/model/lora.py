# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Sequence

from torch import nn


logger = logging.getLogger(__name__)


def inject_lora_adapter(
    module: nn.Module,
    target_modules: Sequence[str],
    rank: int,
    alpha: int,
    dropout: float,
    adapter_name: str,
) -> nn.Module:
    """Inject a PEFT LoRA adapter in-place without wrapping the module forward."""
    from peft import LoraConfig, inject_adapter_in_model

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        lora_dropout=dropout,
        bias="none",
    )
    module = inject_adapter_in_model(lora_config, module, adapter_name=adapter_name)
    logger.info(
        "Injected LoRA adapter '%s' into %s with targets=%s, r=%s, alpha=%s, dropout=%s",
        adapter_name,
        module.__class__.__name__,
        list(target_modules),
        rank,
        alpha,
        dropout,
    )
    return module


def freeze_non_lora_parameters(module: nn.Module) -> None:
    """Freeze every parameter except PEFT LoRA adapter weights."""
    for name, param in module.named_parameters():
        param.requires_grad = "lora_" in name

