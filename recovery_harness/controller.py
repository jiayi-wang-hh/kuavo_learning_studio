from __future__ import annotations

from dataclasses import asdict
from enum import Enum
import logging
import time
import numpy as np

from .models import SkillRecommendation
from .policies import ActionExecutor, LingBotPolicyAdapter
from .registry import SkillRegistry

log = logging.getLogger("kuavo.recovery")


class ControllerState(str, Enum):
    NORMAL = "NORMAL"
    FAILURE_CANDIDATE = "FAILURE_CANDIDATE"
    STOPPING = "STOPPING"
    RECOVERY_PRECHECK = "RECOVERY_PRECHECK"
    RECOVERING = "RECOVERING"
    VERIFY_RECOVERY = "VERIFY_RECOVERY"
    RESUME_PRECHECK = "RESUME_PRECHECK"
    STOPPED = "STOPPED"
    TASK_COMPLETED = "TASK_COMPLETED"


class RecoveryController:
    """Safety gate separating VLM recommendation, controller decision, and execution."""
    def __init__(self, registry: SkillRegistry, policies: LingBotPolicyAdapter,
                 executor: ActionExecutor, *, max_observation_age: float = 0.5):
        self.registry, self.policies, self.executor = registry, policies, executor
        self.max_observation_age = max_observation_age
        self.state = ControllerState.NORMAL
        self.active_skill = "normal_sort"
        self.last_decision: dict = {}
        self._recovery_started_at: float | None = None
        self._policy_started_at: float | None = None

    def decide(self, recommendation: SkillRecommendation, *, observation_age: float | None,
               robot_state: dict, dry_run: bool = True) -> dict:
        selected = "stop_and_verify"
        reason = "controller default: no safe transition established"
        if recommendation.recommended_skill not in self.registry.allowed_recommendations():
            reason = f"stop because VLM recommended an unregistered skill: {recommendation.recommended_skill!r}"
            self._stop_all()
        elif observation_age is None or observation_age > self.max_observation_age:
            reason = "stop because the observation used for reasoning is stale or missing"
            self._stop_all()
        elif robot_state.get("valid") is not True or robot_state.get("safety_ok") is not True:
            reason = "stop because robot state or safety status is unknown/invalid"
            self._stop_all()
        elif recommendation.recommended_skill == "normal_sort" and self.state == ControllerState.NORMAL:
            selected, reason = "normal_sort", "continue registered full-task policy"
        elif recommendation.recommended_skill == "recover_package" and self.state in (ControllerState.NORMAL, ControllerState.FAILURE_CANDIDATE):
            self.state = ControllerState.FAILURE_CANDIDATE
            if not recommendation.target_object:
                reason = "recovery recommendation has no identified target"
                self._stop_all()
            elif robot_state.get("safety_ok") is not True or robot_state.get("motion_state") != "running":
                reason = "recovery target recognized; normal policy must be safely stopped and state validated"
                self.state = ControllerState.STOPPING
                self.policies.request_stop()
            else:
                reason = "recovery candidate recognized; stop normal policy before switching"
                self.state = ControllerState.STOPPING
                self.policies.request_stop()
        elif recommendation.recommended_skill == "stop_and_verify":
            reason = recommendation.reason or "VLM requested human verification"
            self._stop_all()
        else:
            reason = f"stop because recommendation {recommendation.recommended_skill!r} is not allowed in {self.state.value}"
            self._stop_all()
        decision = {
            "controller_state": self.state.value, "active_skill": self.active_skill,
            "vlm_recommendation": asdict(recommendation), "controller_selected_skill": selected,
            "decision_reason": reason, "execution_status": "dry_run_no_motion" if dry_run else "not_started",
            "timestamp": time.time(),
        }
        self.last_decision = decision
        log.info("routing_decision %s", decision)
        return decision

    def _stop_all(self) -> None:
        self.policies.request_stop()
        self.executor.request_stop()
        if self.active_skill in {skill.skill_id for skill in self.registry}:
            self.registry.update_execution_status(self.active_skill, "stopped")
        self.state = ControllerState.STOPPED

    def emergency_stop(self, reason: str) -> dict:
        """Latch the controller in STOPPED and request policy/executor cancellation."""
        self._stop_all()
        self.last_decision = {
            "controller_state": self.state.value,
            "active_skill": self.active_skill,
            "controller_selected_skill": "stop_and_verify",
            "decision_reason": reason,
            "execution_status": "stop_requested",
            "timestamp": time.time(),
        }
        log.warning("operator_stop_signal %s", self.last_decision)
        return self.last_decision

    def approve_recovery_switch(self, *, robot_state: dict, observations_fresh: bool,
                                target_localized: bool, collision_clear: bool,
                                recovery_checkpoint_ready: bool, policy_cancel_acknowledged: bool = False,
                                action_queue_drained: bool = False, mock_execution: bool = False) -> tuple[bool, str]:
        checks = {
            "controller is stopping": self.state == ControllerState.STOPPING,
            "robot state is valid": robot_state.get("valid") is True,
            "joint state is valid": robot_state.get("joint_state_valid") is True,
            "end-effector state is valid": robot_state.get("end_effector_state_valid") is True,
            "robot reports stopped": robot_state.get("motion_state") == "stopped",
            "safety state valid": robot_state.get("safety_ok") is True,
            "active policy cancellation acknowledged": policy_cancel_acknowledged,
            "pending action queue drained": action_queue_drained,
            "observations are fresh": observations_fresh,
            "target is localized": target_localized,
            "collision check passed": collision_clear,
            "recovery checkpoint is loadable": recovery_checkpoint_ready,
        }
        if not mock_execution:
            try:
                self.registry.resolve_checkpoint(self.registry.get("recover_package"))
            except Exception:
                checks["selected recovery checkpoint resolves to inference format"] = False
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            self.state = ControllerState.STOPPED
            return False, "recovery precheck failed: " + ", ".join(failed)
        self.state = ControllerState.RECOVERY_PRECHECK
        return True, "recovery prechecks passed; explicit executor authorization is still required"

    def begin_recovery(self, *, explicitly_authorized: bool = False) -> None:
        if self.state != ControllerState.RECOVERY_PRECHECK:
            raise RuntimeError(f"Cannot start recovery from {self.state.value}")
        if not explicitly_authorized or not self.executor.enabled:
            raise PermissionError("Recovery motion requires explicit authorization and an enabled executor")
        self.policies.initialize("recover_package")
        self.registry.update_execution_status("normal_sort", "stopped")
        self.registry.update_execution_status("recover_package", "running")
        self.active_skill = "recover_package"
        self._recovery_started_at = time.monotonic()
        self._policy_started_at = self._recovery_started_at
        self.state = ControllerState.RECOVERING

    def verify_recovery(self, *, package_on_platform: bool | None, target_visible: bool | None,
                        tactile_contact: int | None, robot_safe: bool,
                        release_confirmed: bool = False) -> tuple[bool, str]:
        if self.state != ControllerState.VERIFY_RECOVERY:
            raise RuntimeError(f"Cannot verify recovery from {self.state.value}")
        # VLA chunk completion and aggregate force do not prove successful placement.
        if (package_on_platform is True and target_visible is True and tactile_contact in (0, None)
                and release_confirmed is True and robot_safe):
            self.registry.update_execution_status("recover_package", "verified")
            self.state = ControllerState.RESUME_PRECHECK
            return True, "package visually verified on platform; assess normal-policy initial state"
        self.state = ControllerState.STOPPED
        self.registry.update_execution_status("recover_package", "failed_verification")
        return False, f"recovery not verified (package_on_platform={package_on_platform}, target_visible={target_visible}, contact={tactile_contact}, release_confirmed={release_confirmed}, robot_safe={robot_safe})"

    def finish_recovery_execution(self) -> None:
        if self.state != ControllerState.RECOVERING:
            raise RuntimeError(f"Cannot finish recovery from {self.state.value}")
        self.policies.request_stop()
        self.registry.update_execution_status("recover_package", "awaiting_verification")
        self.state = ControllerState.VERIFY_RECOVERY

    def approve_resume(self, *, normal_initial_state_supported: bool, robot_state: dict,
                       observations_fresh: bool) -> tuple[bool, str]:
        if self.state != ControllerState.RESUME_PRECHECK:
            return False, f"cannot resume from {self.state.value}"
        if not (normal_initial_state_supported and robot_state.get("valid") is True
                and robot_state.get("joint_state_valid") is True
                and robot_state.get("end_effector_state_valid") is True
                and robot_state.get("safety_ok") is True and observations_fresh
                and robot_state.get("motion_state") == "stopped"):
            self.state = ControllerState.STOPPED
            return False, "normal policy initial-state support or resume preconditions not verified"
        self.policies.initialize("normal_sort")
        self.registry.update_execution_status("recover_package", "verified")
        self.registry.update_execution_status("normal_sort", "running")
        self.active_skill = "normal_sort"
        self._policy_started_at = time.monotonic()
        self.state = ControllerState.NORMAL
        return True, "normal policy reloaded and resume checks passed"

    def start_normal(self, *, explicitly_authorized: bool = False, robot_state: dict | None = None,
                     observations_fresh: bool = False) -> None:
        if not explicitly_authorized or not self.executor.enabled:
            raise PermissionError("Normal-policy motion requires explicit authorization and an enabled executor")
        if self.state not in (ControllerState.NORMAL, ControllerState.TASK_COMPLETED):
            raise RuntimeError(f"Cannot start normal policy from {self.state.value}")
        if (not observations_fresh or not robot_state or robot_state.get("valid") is not True
                or robot_state.get("joint_state_valid") is not True
                or robot_state.get("end_effector_state_valid") is not True
                or robot_state.get("safety_ok") is not True or robot_state.get("motion_state") != "stopped"):
            raise PermissionError("Normal-policy start requires fresh observations and verified stopped/safe robot state")
        self.policies.initialize("normal_sort")
        self.registry.update_execution_status("normal_sort", "running")
        self.active_skill = "normal_sort"
        self._policy_started_at = time.monotonic()
        self.state = ControllerState.NORMAL

    def step_active_policy(self, policy_observation: dict, *, robot_state: dict,
                           observations_fresh: bool) -> None:
        if self.state not in (ControllerState.NORMAL, ControllerState.RECOVERING):
            raise RuntimeError(f"Action submission is forbidden in {self.state.value}")
        if not observations_fresh or robot_state.get("valid") is not True or robot_state.get("safety_ok") is not True:
            self.policies.request_stop()
            self.executor.request_stop()
            self.state = ControllerState.STOPPED
            raise PermissionError("Action submission requires fresh observations and a verified safe robot state")
        skill = self.registry.get(self.active_skill)
        if skill and self._policy_started_at is not None and time.monotonic() - self._policy_started_at > skill.execution_timeout:
            self.policies.request_stop()
            self.executor.request_stop()
            self.state = ControllerState.STOPPED
            raise TimeoutError(f"Skill {skill.skill_id} exceeded {skill.execution_timeout}s")
        action_chunk = self.policies.infer_action(policy_observation)
        for action in action_chunk:
            if self.state not in (ControllerState.NORMAL, ControllerState.RECOVERING) or self.policies.stop_requested:
                break
            if not np.isfinite(np.asarray(action)).all():
                self.state = ControllerState.STOPPED
                raise ValueError("Policy produced a non-finite action; stopping action submission")
            self.executor.execute_action(action)

    def mark_task_completed(self, *, all_six_in_basket: bool, robot_safe: bool) -> None:
        if all_six_in_basket is not True or robot_safe is not True:
            raise RuntimeError("Task completion requires visual confirmation of all six packages in the basket and a safe robot state")
        self.policies.request_stop()
        self.state = ControllerState.TASK_COMPLETED
        self.last_decision = {"controller_state": self.state.value,
                              "controller_selected_skill": "stop_and_verify",
                              "reason": "six-package sorting completion externally verified"}

