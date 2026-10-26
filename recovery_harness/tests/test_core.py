from pathlib import Path
import unittest
from types import SimpleNamespace

from recovery_harness.models import SkillRecommendation
from recovery_harness.perception import box_platform_overlap, crosses_body_edge
from recovery_harness.registry import SkillRegistry
from recovery_harness.controller import ControllerState, RecoveryController
from recovery_harness.tactile import BinaryContactProcessor
from recovery_harness.events import EventTrigger
from recovery_harness.scheduler import VLMInvocationScheduler
from recovery_harness.policies import ActionExecutor, MockExecutionBackend
from recovery_harness.models import CameraFrame
import numpy as np
import time


ROOT = Path(__file__).resolve().parents[2]


class GeometryTests(unittest.TestCase):
    def test_box_overlap_and_body_side(self):
        platform = [(0, 0), (10, 0), (10, 10), (0, 10)]
        self.assertAlmostEqual(box_platform_overlap((5, 5, 15, 15), platform, complete_object=True), 0.25)
        self.assertIsNone(box_platform_overlap((5, 5, 15, 15), platform, complete_object=False))
        self.assertTrue(crosses_body_edge((4, 11, 6, 13), [(0, 10), (10, 10)], complete_object=True, platform_polygon=platform))
        self.assertIsNone(crosses_body_edge((4, 11, 6, 13), [(0, 10), (10, 10)], complete_object=False, platform_polygon=platform))


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = SkillRegistry.from_yaml(ROOT / "recovery_harness/config/skill_registry.yaml")

    def test_skill_prompt_is_registry_derived(self):
        prompt = self.registry.prompt_descriptions()
        self.assertIn("normal_sort", prompt)
        self.assertIn("recover_package", prompt)
        self.assertEqual(self.registry.allowed_recommendations(), {"normal_sort", "recover_package", "stop_and_verify"})

    def test_checkpoint_resolution(self):
        normal = self.registry.get("normal_sort")
        if Path(normal.checkpoint).is_dir():
            self.assertTrue(self.registry.resolve_checkpoint(normal).joinpath("config.json").is_file())
        recovery = self.registry.get("recover_package")
        with self.assertRaises(RuntimeError):
            self.registry.resolve_checkpoint(recovery)


class TactileTests(unittest.TestCase):
    def test_two_fingers_and_sustain_duration(self):
        processor = BinaryContactProcessor([5] * 5, [0], sustain_seconds=0.1)
        finger = lambda force: SimpleNamespace(status=0, normal_force1=force, normal_force2=0, normal_force3=0)
        msg = SimpleNamespace(left_hand=[finger(6), finger(6), finger(0), finger(0), finger(0)],
                              right_hand=[finger(0) for _ in range(5)])
        self.assertEqual(processor.process(msg, now=1.0)["left_multifinger_contact"], 0)
        self.assertEqual(processor.process(msg, now=1.11)["left_multifinger_contact"], 1)
        unknown = BinaryContactProcessor([5] * 5, None)
        self.assertIsNone(unknown.process(msg, now=2.0)["left_multifinger_contact"])


class ControllerTests(unittest.TestCase):
    def test_bad_recommendation_stops_and_never_executes(self):
        class Policies:
            def request_stop(self): pass
        class Executor:
            enabled = False
            def request_stop(self): pass
        registry = SkillRegistry.from_yaml(ROOT / "recovery_harness/config/skill_registry.yaml")
        controller = RecoveryController(registry, Policies(), Executor())
        rec = SkillRecommendation("unknown", "unknown", "stop_and_verify", None, "unclear", True)
        decision = controller.decide(rec, observation_age=0.1, robot_state={"valid": True}, dry_run=True)
        self.assertEqual(decision["controller_state"], ControllerState.STOPPED.value)
        self.assertEqual(decision["execution_status"], "dry_run_no_motion")

    def setUp(self):
        self.registry = SkillRegistry.from_yaml(ROOT / "recovery_harness/config/skill_registry.yaml")

    def test_invalid_skill_fails_closed(self):
        controller = RecoveryController(self.registry, SimpleNamespace(request_stop=lambda: None), ActionExecutor(False))
        rec = SkillRecommendation("x", "x", "not_registered", None, "bad", True)
        self.assertEqual(controller.decide(rec, observation_age=0.1, robot_state={"valid": True, "safety_ok": True})["controller_state"], "STOPPED")

    def test_normal_and_recovery_recommendations_enter_expected_states(self):
        class Policies:
            def __init__(self): self.stopped = False
            def request_stop(self): self.stopped = True
        policies = Policies()
        controller = RecoveryController(self.registry, policies, ActionExecutor(False))
        normal = SkillRecommendation("normal", "sorting", "normal_sort", None, "ok", True)
        self.assertEqual(controller.decide(normal, observation_age=.1, robot_state={"valid": True, "safety_ok": True})["controller_selected_skill"], "normal_sort")
        recovery = SkillRecommendation("edge", "package displaced", "recover_package", "pink package", "edge", True)
        decision = controller.decide(recovery, observation_age=.1, robot_state={"valid": True, "safety_ok": True, "motion_state": "running"})
        self.assertEqual(decision["controller_state"], "STOPPING")
        self.assertTrue(policies.stopped)

    def test_recovery_precheck_failure_stops_without_execution(self):
        controller = RecoveryController(self.registry, SimpleNamespace(request_stop=lambda: None), ActionExecutor(False))
        controller.state = ControllerState.STOPPING
        ok, _ = controller.approve_recovery_switch(robot_state={"motion_state": "unknown"}, observations_fresh=False,
            target_localized=False, collision_clear=False, recovery_checkpoint_ready=False)
        self.assertFalse(ok)
        self.assertEqual(controller.state, ControllerState.STOPPED)

    def test_unknown_tactile_is_not_release_evidence(self):
        controller = RecoveryController(self.registry, SimpleNamespace(request_stop=lambda: None), ActionExecutor(False))
        controller.state = ControllerState.VERIFY_RECOVERY
        ok, _ = controller.verify_recovery(package_on_platform=True, target_visible=True, tactile_contact=None, robot_safe=True)
        self.assertFalse(ok)

    def test_trustworthy_release_allows_resume_precheck(self):
        controller = RecoveryController(self.registry, SimpleNamespace(request_stop=lambda: None), ActionExecutor(False))
        controller.state = ControllerState.VERIFY_RECOVERY
        ok, _ = controller.verify_recovery(package_on_platform=True, target_visible=True, tactile_contact=0,
                                           release_confirmed=True, robot_safe=True)
        self.assertTrue(ok)
        self.assertEqual(controller.state, ControllerState.RESUME_PRECHECK)

    def test_mock_recovery_execution_returns_to_verification(self):
        class Policies:
            stop_requested = False
            def initialize(self, skill_id): self.stop_requested = False
            def request_stop(self): self.stop_requested = True
            def infer_action(self, observation): return [[0.0]]
        policies = Policies(); backend = MockExecutionBackend()
        controller = RecoveryController(self.registry, policies, ActionExecutor(True, backend))
        rec = SkillRecommendation("edge", "displaced", "recover_package", "package", "mock", True)
        controller.decide(rec, observation_age=.1, robot_state={"valid": True, "safety_ok": True, "motion_state": "running"})
        ok, _ = controller.approve_recovery_switch(robot_state={"valid": True, "joint_state_valid": True,
            "end_effector_state_valid": True, "safety_ok": True, "motion_state": "stopped"},
            observations_fresh=True, target_localized=True, collision_clear=True,
            recovery_checkpoint_ready=True, policy_cancel_acknowledged=True,
            action_queue_drained=True, mock_execution=True)
        self.assertTrue(ok)
        controller.begin_recovery(explicitly_authorized=True)
        controller.step_active_policy({}, robot_state={"valid": True, "safety_ok": True}, observations_fresh=True)
        controller.finish_recovery_execution()
        self.assertEqual(controller.state, ControllerState.VERIFY_RECOVERY)
        self.assertEqual(backend.get_execution_status()["robot_motion"], False)

    def test_invalid_resume_state_stops(self):
        controller = RecoveryController(self.registry, SimpleNamespace(request_stop=lambda: None), ActionExecutor(False))
        controller.state = ControllerState.RESUME_PRECHECK
        ok, _ = controller.approve_resume(normal_initial_state_supported=False,
            robot_state={"safety_ok": True, "motion_state": "stopped"}, observations_fresh=True)
        self.assertFalse(ok)
        self.assertEqual(controller.state, ControllerState.STOPPED)

    def test_emergency_stop_requests_both_policy_and_executor_stop(self):
        class StopTarget:
            def __init__(self): self.called = False
            def request_stop(self): self.called = True
        policies, executor = StopTarget(), StopTarget()
        controller = RecoveryController(self.registry, policies, executor)
        decision = controller.emergency_stop("test stop signal")
        self.assertEqual(controller.state, ControllerState.STOPPED)
        self.assertTrue(policies.called)
        self.assertTrue(executor.called)
        self.assertEqual(decision["execution_status"], "stop_requested")


class EventAndSchedulerTests(unittest.TestCase):
    def test_cv_abnormality_persistent_triggers(self):
        trigger = EventTrigger(persistence_hits=2, cooldown_seconds=0)
        p = {"geometry_validity": "bbox_approximation", "candidate_tracks": [{"track_id": "p1", "track_valid": True,
            "partial": False, "track_source": "detector", "confidence": .9, "body_side_violation": True}]}
        self.assertEqual(trigger.update(p, {}, {}, now=1), [])
        self.assertEqual(trigger.update(p, {}, {}, now=2)[0].event_type, "POSSIBLE_PACKAGE_OUTSIDE_PLATFORM")

    def test_contact_transition_triggers_but_unknown_never_means_lost(self):
        trigger = EventTrigger(persistence_hits=1, cooldown_seconds=0)
        trigger.update({}, {"left_multifinger_contact": 1}, {}, now=1)
        events = trigger.update({}, {"left_multifinger_contact": 0}, {}, now=2)
        self.assertIn("CONTACT_LOST", [e.event_type for e in events])
        events = trigger.update({}, {"left_multifinger_contact": None}, {}, now=3)
        self.assertNotIn("CONTACT_LOST", [e.event_type for e in events])

    def test_simultaneous_events_result_in_one_inference(self):
        class Reasoner:
            def __init__(self): self.calls = 0
            def reason(self, obs): self.calls += 1; return ("ok", .01)
        reasoner = Reasoner(); scheduler = VLMInvocationScheduler(reasoner, periodic_interval=100, max_observation_age=10)
        class Event:
            source = "cv"; event_type = "x"; evidence = {}
        frame = CameraFrame("head", np.zeros((2, 2, 3)), None, time.monotonic())
        from recovery_harness.models import MultiViewObservation
        obs = MultiViewObservation({"head": frame})
        scheduler.tick(obs, [Event(), Event()])
        deadline = time.monotonic() + 1
        result = None
        while time.monotonic() < deadline and result is None:
            result = scheduler.poll(); time.sleep(.005)
        scheduler.close()
        self.assertIsNotNone(result)
        self.assertEqual(reasoner.calls, 1)

    def test_no_inference_before_periodic_interval_and_stale_result_rejected(self):
        class Reasoner:
            def reason(self, obs): return ("ok", .01)
        scheduler = VLMInvocationScheduler(Reasoner(), periodic_interval=10, max_observation_age=.01)
        from recovery_harness.models import MultiViewObservation
        frame = CameraFrame("head", np.zeros((2, 2, 3)), None, time.monotonic())
        obs = MultiViewObservation({"head": frame})
        base = scheduler._created_at
        scheduler.tick(obs, [], now=base)
        self.assertIsNone(scheduler._future)
        scheduler.tick(obs, [], now=base + 11)
        time.sleep(.01)
        result = scheduler.poll(now=base + 12)
        scheduler.close()
        self.assertFalse(result["valid"])

    def test_pause_discards_an_inflight_result(self):
        import threading
        entered, release = threading.Event(), threading.Event()
        class Reasoner:
            def reason(self, obs):
                entered.set(); release.wait(1); return ("ok", .01)
        scheduler = VLMInvocationScheduler(Reasoner(), periodic_interval=100, max_observation_age=10)
        from recovery_harness.models import MultiViewObservation
        frame = CameraFrame("head", np.zeros((2, 2, 3)), None, time.monotonic())
        obs = MultiViewObservation({"head": frame})
        class Event:
            source = "tactile"; event_type = "CONTACT_LOST"; evidence = {}
        scheduler.tick(obs, [Event()])
        self.assertTrue(entered.wait(1))
        scheduler.set_paused(True)
        release.set()
        deadline = time.monotonic() + 1
        result = None
        while time.monotonic() < deadline and result is None:
            result = scheduler.poll(); time.sleep(.005)
        scheduler.close()
        self.assertFalse(result["valid"])
        self.assertEqual(result["discard_reason"], "pause_signal")

    def test_mock_backend_never_uses_robot_interface(self):
        backend = MockExecutionBackend(); executor = ActionExecutor(True, backend)
        executor.execute_action([0.0, 1.0])
        self.assertEqual(backend.get_execution_status(), {"status": "simulated", "action_count": 1, "robot_motion": False})

    def test_missing_real_backend_fails_closed(self):
        executor = ActionExecutor(enabled=True, backend=None)
        with self.assertRaises(RuntimeError): executor.execute_action([0.0])


if __name__ == "__main__":
    unittest.main()

