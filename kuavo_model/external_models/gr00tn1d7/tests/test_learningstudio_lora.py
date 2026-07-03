import importlib.util
from pathlib import Path
import sys

import torch

SCRIPT = Path(__file__).parents[1] / "launch_finetune_lora.py"
SPEC = importlib.util.spec_from_file_location("learningstudio_lora", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TinyGr00t(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.ModuleDict({"q_proj": torch.nn.Linear(4, 4)})
        self.action_head = torch.nn.ModuleDict(
            {"to_q": torch.nn.Linear(4, 4), "other": torch.nn.Linear(4, 4)}
        )


def test_target_discovery():
    assert MODULE.discover_lora_targets(TinyGr00t()) == [
        "backbone.q_proj",
        "action_head.to_q",
    ]


def test_action_head_only_target_discovery():
    assert MODULE.discover_lora_targets(TinyGr00t(), action_head_only=True) == [
        "action_head.to_q"
    ]


def test_injection_only_trains_lora_weights():
    model = MODULE.inject_peft_lora(TinyGr00t(), rank=2, alpha=4, dropout=0.0)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert all("lora_" in name for name in trainable)
