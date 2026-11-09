from __future__ import annotations

from argparse import Namespace
import gc
import threading
from typing import Any

from .registry import Skill, SkillRegistry


class LingBotPolicyAdapter:
    """Lazy, single-policy adapter around the repository's existing LingBot server adapter."""
    def __init__(self, registry: SkillRegistry, *, device: str = "cuda"):
        self.registry, self.device = registry, device
        self.active_skill: Skill | None = None
        self.adapter: Any = None
        self._cancelled = threading.Event()

    def initialize(self, skill_id: str) -> None:
        if self.active_skill and self.active_skill.skill_id == skill_id:
            return
        self.shutdown()
        skill = self.registry.get(skill_id)
        if skill is None:
            raise KeyError(f"No skill registered as {skill_id!r}")
        checkpoint = self.registry.resolve_checkpoint(skill)
        try:
            from kuavo_server.adapters.lingbot_vla import LingBotVlaAdapter
        except ImportError as exc:
            raise RuntimeError(
                "Could not import the existing LingBot adapter. Add the project root to PYTHONPATH "
                "and install the project's runtime dependencies."
            ) from exc
        args = Namespace(
            checkpoint=str(checkpoint), model_repo_root=skill.model_repo_root,
            which_arm="both", execution_horizon=skill.execution_horizon,
            qwen25_path=skill.tokenizer, training_config_path=skill.training_config,
            robot_config_path=skill.robot_config, robot_norm_path=skill.norm_stats,
            num_denoising_step=10, use_compile=False, use_fp32=False,
        )
        self.adapter = LingBotVlaAdapter.from_args(args)
        self.active_skill = skill
        self._cancelled.clear()

    def infer_action(self, observation: dict) -> Any:
        if self.adapter is None or self.active_skill is None:
            raise RuntimeError("No VLA skill is initialized")
        obs = dict(observation)
        obs["prompt"] = self.active_skill.task_instruction
        return self.adapter.select_action_chunk(obs)

    def request_stop(self) -> None:
        self._cancelled.set()
        if self.adapter is not None:
            # Prevent any unconsumed old-policy actions from being returned after a switch.
            self.adapter._pending_actions.clear()

    def get_execution_status(self) -> dict:
        return {"active_skill": self.active_skill.skill_id if self.active_skill else None,
                "status": "loaded" if self.adapter is not None else "unloaded"}

    @property
    def stop_requested(self) -> bool:
        return self._cancelled.is_set()

    def shutdown(self) -> None:
        if self.adapter is not None:
            self.request_stop()
            del self.adapter
            self.adapter = None
            self.active_skill = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


class ActionExecutor:
    """Motion is fail-closed; a site-specific Kuavo executor must be supplied explicitly."""
    def __init__(self, enabled: bool = False, backend: Any = None):
        self.enabled, self.backend = enabled, backend

    def execute_action(self, action) -> None:
        if not self.enabled:
            raise PermissionError("Robot motion is disabled; use live-dry-run or configure explicit motion authorization")
        if self.backend is None:
            raise RuntimeError("No verified Kuavo action executor backend is configured")
        self.backend.execute_action(action)

    def request_stop(self) -> None:
        if self.backend is not None:
            self.backend.request_stop()

    def wait_until_stopped(self, timeout_seconds: float) -> bool:
        if self.backend is None:
            return False
        waiter = getattr(self.backend, "wait_until_stopped", None)
        return bool(waiter(timeout_seconds)) if callable(waiter) else False

    def get_execution_status(self) -> dict:
        if self.backend is None:
            return {"status": "unavailable", "motion_enabled": self.enabled}
        getter = getattr(self.backend, "get_execution_status", None)
        return getter() if callable(getter) else {"status": "unknown"}

    def shutdown(self) -> None:
        if self.backend is not None:
            closer = getattr(self.backend, "shutdown", None)
            if callable(closer):
                closer()


class MockExecutionBackend:
    """Explicit synthetic backend for orchestration tests; has no ROS/motor interface."""
    is_simulated = True

    def __init__(self):
        self.actions = []
        self.stopped = True

    def execute_action(self, action) -> None:
        self.actions.append(action)

    def request_stop(self) -> None:
        self.stopped = True

    def wait_until_stopped(self, timeout_seconds: float) -> bool:
        return self.stopped

    def get_execution_status(self) -> dict:
        return {"status": "simulated", "action_count": len(self.actions), "robot_motion": False}

