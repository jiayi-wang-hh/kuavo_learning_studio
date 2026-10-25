from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Skill:
    skill_id: str
    skill_type: str
    description: str
    checkpoint: str | None
    checkpoint_root: str | None
    checkpoint_selector: str | None
    exported_hf_checkpoint: str | None
    base_model: str
    model_repo_root: str
    tokenizer: str
    training_config: str
    robot_config: str
    norm_stats: str
    task_instruction: str
    expected_preconditions: tuple[str, ...]
    expected_postconditions: tuple[str, ...]
    supported_scenarios: tuple[str, ...]
    execution_timeout: float
    execution_status: str
    execution_horizon: int

    @property
    def policy_checkpoint(self) -> str:
        if self.exported_hf_checkpoint:
            return self.exported_hf_checkpoint
        if self.checkpoint:
            return self.checkpoint
        if self.checkpoint_root and self.checkpoint_selector:
            return str(Path(self.checkpoint_root) / self.checkpoint_selector)
        return ""


class SkillRegistry:
    def __init__(self, skills: list[Skill]):
        if not skills:
            raise ValueError("Skill registry must contain at least one skill")
        ids = [skill.skill_id for skill in skills]
        if len(ids) != len(set(ids)):
            raise ValueError("Skill IDs must be unique")
        self._skills = {skill.skill_id: skill for skill in skills}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SkillRegistry":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        skills = []
        for item in data.get("skills", []):
            skills.append(Skill(
                skill_id=item["skill_id"], skill_type=item["skill_type"],
                description=item["description"], checkpoint=item.get("checkpoint"),
                checkpoint_root=item.get("checkpoint_root"),
                checkpoint_selector=item.get("checkpoint_selector"),
                exported_hf_checkpoint=item.get("exported_hf_checkpoint"),
                base_model=item.get("base_model", ""), tokenizer=item.get("tokenizer", ""),
                model_repo_root=item.get("model_repo_root", ""),
                training_config=item.get("training_config", ""), robot_config=item.get("robot_config", ""),
                norm_stats=item.get("norm_stats", ""), task_instruction=item["task_instruction"],
                expected_preconditions=tuple(item.get("expected_preconditions", [])),
                expected_postconditions=tuple(item.get("expected_postconditions", [])),
                supported_scenarios=tuple(item.get("supported_scenarios", [])),
                execution_timeout=float(item.get("execution_timeout", 60)),
                execution_status=item.get("execution_status", "idle"),
                execution_horizon=int(item.get("execution_horizon", 25)),
            ))
        return cls(skills)

    def get(self, skill_id: str) -> Skill | None:
        return self._skills.get(skill_id)

    def update_execution_status(self, skill_id: str, status: str) -> None:
        if skill_id not in self._skills:
            raise KeyError(skill_id)
        self._skills[skill_id] = replace(self._skills[skill_id], execution_status=status)

    def __iter__(self):
        return iter(self._skills.values())

    def prompt_descriptions(self) -> str:
        # This text is always rendered from the registry; prompts do not duplicate skill names.
        return "\n".join(
            f"{index}. {skill.skill_id}\n"
            f"   Function: {skill.description.strip()}\n"
            f"   Task instruction: {skill.task_instruction}\n"
            f"   Appropriate when: {'; '.join(skill.supported_scenarios)}\n"
            f"   Preconditions: {'; '.join(skill.expected_preconditions)}\n"
            f"   Execution status: {skill.execution_status}"
            for index, skill in enumerate(self, start=1)
        )

    def allowed_recommendations(self) -> set[str]:
        return set(self._skills) | {"stop_and_verify"}

    @staticmethod
    def resolve_checkpoint(skill: Skill) -> Path:
        if skill.exported_hf_checkpoint:
            path = Path(skill.exported_hf_checkpoint).expanduser()
            if not (path / "config.json").is_file():
                raise FileNotFoundError(f"Configured HF export is missing config.json: {path}")
            return path
        if skill.checkpoint:
            path = Path(skill.checkpoint).expanduser()
            if (path / "config.json").is_file() and any(path.glob("*.safetensors")):
                return path
            # Accept a run directory whose selected step contains an HF export.
            step = path / skill.checkpoint_selector if skill.checkpoint_selector else path
            for candidate in (step / "hf_ckpt", step / "pretrained_model", step):
                if (candidate / "config.json").is_file() and any(candidate.glob("*.safetensors")):
                    return candidate
            raise FileNotFoundError(f"No loadable HF policy checkpoint found under {path}")
        root = Path(skill.checkpoint_root or "").expanduser()
        if not skill.checkpoint_selector:
            raise ValueError(f"Skill {skill.skill_id} needs checkpoint_selector for checkpoint_root")
        selected = root / skill.checkpoint_selector
        if skill.exported_hf_checkpoint:
            return cls.resolve_checkpoint(skill)
        # DCP snapshots are intentionally not passed to the HF-only inference loader.
        if (selected / "model" / ".metadata").is_file():
            raise RuntimeError(
                f"{selected} is a distributed DCP checkpoint, not an inference-ready HF checkpoint. "
                "Export this selected checkpoint to safetensors and set exported_hf_checkpoint."
            )
        raise FileNotFoundError(f"Selected checkpoint does not exist: {selected}")

