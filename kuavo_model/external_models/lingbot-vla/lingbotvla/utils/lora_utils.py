# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
from safetensors import safe_open


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False, device=base_layer.weight.device)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False, device=base_layer.weight.device)
        self.register_buffer("lora_scaling", torch.tensor(alpha / rank, dtype=torch.float32), persistent=True)

        for param in self.base_layer.parameters():
            param.requires_grad = False

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.weight.data = self.lora_A.weight.data.to(torch.float32)
        self.lora_B.weight.data = self.lora_B.weight.data.to(torch.float32)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        base_output = self.base_layer(x)
        lora_input = self.dropout(x).to(self.lora_A.weight.dtype)
        lora_output = self.lora_B(self.lora_A(lora_input)) * self.lora_scaling
        return base_output + lora_output.to(base_output.dtype)


def freeze_parameters(model: nn.Module):
    model.requires_grad_(False)
    model.eval()
    model.train()


def _split_csv(value: str | Iterable[str] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    if "." in module_name:
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = root.get_submodule(parent_name)
    else:
        parent = root
        child_name = module_name
    setattr(parent, child_name, new_module)


def add_lora_to_model(
    model: nn.Module,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    lora_target_scope="model.qwenvl_with_expert.qwen_expert.model.layers",
    init_lora_weights=None,
    pretrained_lora_path=None,
    state_dict_converter=None,
    lora_target_modules_support=None,
) -> Tuple[int, List[str]]:
    del init_lora_weights
    target_modules = set(_split_csv(lora_target_modules))
    target_scopes = _split_csv(lora_target_scope)
    if lora_target_modules_support is not None:
        unsupported = target_modules - set(lora_target_modules_support)
        if unsupported:
            raise ValueError(f"Unsupported LoRA target modules: {sorted(unsupported)}")

    replacements = []
    for module_name, module in list(model.named_modules()):
        leaf_name = module_name.rsplit(".", 1)[-1]
        if leaf_name not in target_modules:
            continue
        if target_scopes and not any(scope in module_name for scope in target_scopes):
            continue
        if isinstance(module, LoRALinear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        replacements.append((module_name, module))

    for module_name, module in replacements:
        _replace_module(
            model,
            module_name,
            LoRALinear(
                module,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            ),
        )

    model.lora_alpha = lora_alpha
    model.lora_rank = lora_rank
    model.lora_dropout = lora_dropout

    # Lora pretrained lora weights
    if pretrained_lora_path is not None:
        state_dict = load_state_dict(pretrained_lora_path)
        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        all_keys = [i for i, _ in model.named_parameters()]
        num_updated_keys = len(all_keys) - len(missing_keys)
        num_unexpected_keys = len(unexpected_keys)
        print(
            f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected."
        )
    return len(replacements), [name for name, _ in replacements]


def mark_only_lora_and_modules_trainable(
    model: nn.Module,
    trainable_module_patterns: str | Iterable[str] | None = None,
) -> List[str]:
    freeze_parameters(model)
    trainable_patterns = _split_csv(trainable_module_patterns)
    trainable_names = []

    for name, param in model.named_parameters():
        should_train = ".lora_A." in name or ".lora_B." in name
        should_train = should_train or any(pattern in name for pattern in trainable_patterns)
        param.requires_grad = should_train
        if should_train:
            param.data = param.data.to(torch.float32)
            trainable_names.append(name)
    return trainable_names


def count_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        numel = param.numel()
        total += numel
        if param.requires_grad:
            trainable += numel
    return trainable, total


def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)


def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None):
    state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict
