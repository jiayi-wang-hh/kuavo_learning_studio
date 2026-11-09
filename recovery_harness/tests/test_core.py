from pathlib import Path
import unittest
from types import SimpleNamespace

from recovery_harness.models import SkillRecommendation
from recovery_harness.models import MultiViewObservation
from recovery_harness.perception import box_platform_overlap, crosses_body_edge, visible_mask_roi_evidence
from recovery_harness.registry import SkillRegistry
from recovery_harness.controller import ControllerState, RecoveryController
from recovery_harness.tactile import BinaryContactProcessor
from recovery_harness.events import EventTrigger
from recovery_harness.scheduler import VLMInvocationScheduler
from recovery_harness.policies import ActionExecutor, MockExecutionBackend
from recovery_harness.models import CameraFrame
from recovery_harness.ros_bridge import decode_compressed_rgb
from recovery_harness.vlm import SceneReasoner
from recovery_harness.diagnostics import classify_live_outcome, save_live_snapshot
from recovery_harness.detection import (classify_dino_phrase, filter_package_candidate_groups,
    merge_prompt_candidates, resolve_arm_conflict)
from tempfile import TemporaryDirectory
import json
import numpy as np
import time
from io import BytesIO
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


class PackageCandidateDecisionTests(unittest.TestCase):
    labels = ("blue mailer bag", "pink padded envelope")

    def test_phrase_matching_distinguishes_exact_ambiguous_and_unknown(self):
        exact = classify_dino_phrase("pink", self.labels, prompt_label="pink padded envelope")
        merged = classify_dino_phrase("blue mailer bag pink padded envelope", self.labels)
        unknown = classify_dino_phrase("yellow tray", self.labels)
        self.assertEqual((exact["validity"], exact["normalized_class"]),
                         ("valid", "pink padded envelope"))
        self.assertEqual(merged["validity"], "ambiguous")
        self.assertEqual(set(merged["class_hypotheses"]), set(self.labels))
        self.assertEqual(unknown["validity"], "rejected")

    def test_cross_prompt_duplicate_prefers_specific_evidence_without_score_comparison(self):
        candidates = [
            {"box": [10, 10, 90, 90], "confidence": .92,
             "raw_phrase": "blue mailer bag pink padded envelope",
             "prompt": "blue mailer bag. pink padded envelope.", "query_role": "joint"},
            {"box": [11, 10, 89, 90], "confidence": .44,
             "raw_phrase": "pink padded envelope", "prompt": "pink padded envelope.",
             "query_role": "per_class", "prompt_label": "pink padded envelope"},
        ]
        merged, rejected = merge_prompt_candidates(candidates, self.labels)
        self.assertEqual(rejected, [])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["normalized_class"], "pink padded envelope")
        self.assertEqual(merged[0]["confidence"], .44)
        self.assertEqual(len(merged[0]["prompt_evidence"]), 2)

    def test_overlapping_conflicting_classes_are_ambiguous_not_double_counted(self):
        candidates = [
            {"box": [10, 10, 90, 90], "confidence": .7, "raw_phrase": "blue mailer bag",
             "prompt": "blue mailer bag.", "query_role": "per_class", "prompt_label": "blue mailer bag"},
            {"box": [11, 10, 89, 90], "confidence": .8, "raw_phrase": "pink padded envelope",
             "prompt": "pink padded envelope.", "query_role": "per_class", "prompt_label": "pink padded envelope"},
        ]
        merged, _ = merge_prompt_candidates(candidates, self.labels)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["candidate_validity"], "ambiguous")
        self.assertEqual(set(merged[0]["class_hypotheses"]), set(self.labels))

    def test_same_prompt_proposals_are_left_for_nms_not_cross_prompt_merging(self):
        candidates = [
            {"box": [10, 10, 90, 90], "confidence": .7,
             "raw_phrase": "pink padded envelope", "prompt": "pink padded envelope.",
             "query_role": "per_class", "prompt_label": "pink padded envelope"},
            {"box": [11, 10, 89, 90], "confidence": .6,
             "raw_phrase": "pink padded envelope", "prompt": "pink padded envelope.",
             "query_role": "per_class", "prompt_label": "pink padded envelope"},
        ]
        merged, _ = merge_prompt_candidates(candidates, self.labels)
        self.assertEqual(len(merged), 2)

    def test_low_score_class_proposal_cannot_resolve_a_merged_high_score_phrase(self):
        candidates = [
            {"box": [300, 300, 480, 479], "confidence": .488,
             "raw_phrase": "blue mailer bag pink padded envelope",
             "prompt": "blue mailer bag. pink padded envelope.", "query_role": "joint"},
            {"box": [301, 300, 479, 478], "confidence": .174,
             "raw_phrase": "pink padded envelope", "prompt": "pink padded envelope.",
             "query_role": "per_class", "prompt_label": "pink padded envelope"},
        ]
        accepted, audit = filter_package_candidate_groups(candidates, [],
            package_labels=self.labels, frame_width=848, frame_height=480)
        self.assertEqual(len(accepted), 1)
        self.assertTrue(accepted[0]["ambiguous"])
        low = next(item for item in audit if item.get("confidence") == .174)
        self.assertEqual(low["stage_decisions"][0]["stage"], "confidence_threshold")

    def test_interior_square_arm_false_positive_marks_ambiguous_and_preserves_evidence(self):
        package = {"box": [305, 305, 476, 478], "confidence": .446,
                   "raw_phrase": "pink padded envelope"}
        arm = {"box": [306, 305, 477, 478], "confidence": .151,
               "raw_phrase": "robot arm robot gripper"}
        decision = resolve_arm_conflict(package, [arm], frame_width=848, frame_height=480)
        self.assertEqual(decision["decision"], "ambiguous")
        self.assertTrue(decision["ambiguous"])
        self.assertAlmostEqual(decision["conflict"]["arm_overlap_fraction"], .99, places=2)
        self.assertEqual(decision["conflict"]["package_score"], .446)
        self.assertEqual(decision["conflict"]["arm_score"], .151)

    def test_edge_anchored_elongated_arm_still_rejects_high_overlap_package(self):
        package = {"box": [12, 100, 150, 200], "confidence": .8,
                   "raw_phrase": "blue mailer bag"}
        arm = {"box": [0, 80, 160, 205], "confidence": .47,
               "raw_phrase": "robot arm robot gripper"}
        decision = resolve_arm_conflict(package, [arm], frame_width=848, frame_height=480)
        self.assertEqual(decision["decision"], "reject")
        self.assertTrue(decision["conflict"]["arm_reliable"])

    def test_confidence_gate_is_unchanged_and_arm_conflict_has_full_stage_record(self):
        candidate = {"box": [300, 300, 480, 479], "confidence": .174,
            "raw_phrase": "pink padded envelope", "query_role": "per_class",
            "prompt_label": "pink padded envelope", "prompt": "pink padded envelope."}
        accepted, audit = filter_package_candidate_groups([candidate], [],
            package_labels=self.labels, frame_width=848, frame_height=480,
            box_threshold=.20)
        self.assertEqual(accepted, [])
        confidence_record = next(item for item in audit if item.get("stage_decisions"))
        self.assertEqual(confidence_record["stage_decisions"][0]["stage"], "confidence_threshold")
        self.assertEqual(confidence_record["stage_decisions"][0]["decision"], "rejected")

    def test_ambiguous_candidate_does_not_create_recovery_event(self):
        trigger = EventTrigger(persistence_hits=1, cooldown_seconds=0)
        track = {"track_id": "p1", "track_valid": True, "track_source": "grounding_dino_sam2",
            "label": "pink padded envelope", "confidence": .9, "ambiguous": True,
            "partial": True, "occluded": True, "mask_available": True,
            "mask_confidence": .99, "mask_geometry_valid": True, "detector_age_seconds": .1,
            "platform_visible_pixels": 20, "mask_body_side_extension": True}
        events = trigger.update({"candidate_tracks": [track]}, {}, {}, now=1)
        self.assertNotIn("POSSIBLE_PACKAGE_OUTSIDE_PLATFORM", [event.event_type for event in events])


class GeometryTests(unittest.TestCase):
    def test_box_overlap_and_body_side(self):
        platform = [(0, 0), (10, 0), (10, 10), (0, 10)]
        self.assertAlmostEqual(box_platform_overlap((5, 5, 15, 15), platform, complete_object=True), 0.25)
        self.assertIsNone(box_platform_overlap((5, 5, 15, 15), platform, complete_object=False))
        self.assertTrue(crosses_body_edge((4, 11, 6, 13), [(0, 10), (10, 10)], complete_object=True, platform_polygon=platform))
        self.assertIsNone(crosses_body_edge((4, 11, 6, 13), [(0, 10), (10, 10)], complete_object=False, platform_polygon=platform))

    def test_visible_mask_crosses_platform_edge_using_visible_pixels(self):
        mask = np.zeros((16, 16), dtype=bool)
        mask[6:14, 4:8] = True
        polygon = [(2, 2), (10, 2), (10, 10), (2, 10)]
        evidence = visible_mask_roi_evidence(mask, polygon, [(2, 10), (10, 10)])
        self.assertTrue(evidence["visible_mask_extends_body_side"])
        self.assertGreater(evidence["platform_visible_pixels"], 0)
        self.assertGreater(evidence["body_side_visible_pixels"], 0)

    def test_conveyor_mask_and_partial_mask_do_not_confirm_extension(self):
        polygon = [(2, 2), (10, 2), (10, 10), (2, 10)]
        edge = [(2, 10), (10, 10)]
        conveyor = np.zeros((16, 16), dtype=bool)
        conveyor[11:15, 4:8] = True
        self.assertFalse(visible_mask_roi_evidence(conveyor, polygon, edge)["visible_mask_extends_body_side"])
        crossing = np.zeros((16, 16), dtype=bool)
        crossing[6:14, 4:8] = True
        self.assertIsNone(visible_mask_roi_evidence(crossing, polygon, edge, partial=True)["visible_mask_extends_body_side"])

    def test_live_diagnostic_snapshot_keeps_frame_roi_boxes_mask_and_events_together(self):
        from recovery_harness.models import MultiViewObservation
        image = Image.new("RGB", (16, 16), (20, 30, 40))
        frame = CameraFrame("head", image, 1.0, 1.0, "test")
        mask = np.zeros((16, 16), dtype=bool); mask[5:12, 4:9] = True
        observation = MultiViewObservation({"head": frame}, perception={"candidate_tracks": [{
            "track_id": "p1", "label": "pink padded envelope", "partial": False, "occluded": False,
            "mask_available": True, "mask_geometry_valid": True, "platform_visible_pixels": 10,
            "body_side_visible_pixels": 10, "mask_body_side_extension": True,
            "mask_confidence": .9,
        }]}, diagnostics={"platform_polygon_px": [[2, 2], [10, 2], [10, 10], [2, 10]],
            "body_side_edge_px": [[2, 10], [10, 10]], "tracks": [{"track_id": "p1",
                "label": "pink padded envelope", "confidence": .8, "detector_bbox_xyxy": [3, 4, 10, 12],
                "mask_bbox_xyxy": [4, 5, 9, 12], "mask": mask}]})
        event = {"event_type": "POSSIBLE_PACKAGE_OUTSIDE_PLATFORM", "source": "cv", "evidence": {}}
        recommendation = SkillRecommendation("edge", "displaced", "recover_package", "pink package", "recover", True)
        with TemporaryDirectory() as temp_dir:
            saved = save_live_snapshot(temp_dir, observation, [event], freshness={"valid": False},
                recommendation=recommendation, outcome="vlm_recommended_recover_package_rejected_stale")
            path = Path(saved)
            self.assertTrue((path / "head_rgb.png").is_file())
            self.assertTrue((path / "head_platform_roi.png").is_file())
            self.assertTrue((path / "platform_roi_crop.png").is_file())
            self.assertTrue((path / "head_detection_overlay.png").is_file())
            self.assertTrue((path / "sam2_mask_p1.png").is_file())
            record = json.loads((path / "snapshot.json").read_text())
            self.assertEqual(record["raw_vlm_recommendation"]["recommended_skill"], "recover_package")
            self.assertTrue(record["cv_tracks"][0]["mask_intersects_body_side_region"])

    def test_live_outcome_distinguishes_stale_recovery_and_cv_vlm_disagreement(self):
        from recovery_harness.models import SkillRecommendation
        recovery = SkillRecommendation("edge", "displaced", "recover_package", "package", "edge", True)
        normal = SkillRecommendation("normal", "sorting", "normal_sort", None, "normal", False)
        perception = {"candidate_tracks": [{"label": "blue mailer bag", "mask_available": True,
            "mask_body_side_extension": True, "mask_geometry_valid": True,
            "platform_visible_pixels": 20, "body_side_visible_pixels": 20}]}
        events = [{"event_type": "POSSIBLE_PACKAGE_OUTSIDE_PLATFORM"}]
        self.assertEqual(classify_live_outcome(perception, events, recovery, accepted=False,
            controller_decision=None), "vlm_recommended_recover_package_rejected_stale")
        self.assertEqual(classify_live_outcome(perception, events, normal, accepted=True,
            controller_decision=None), "cv_potential_failure_vlm_recommended_normal_sort")


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

    def test_reasoner_rejects_missing_or_unregistered_skill_fields(self):
        frame = CameraFrame("head", np.zeros((2, 2, 3)), None, time.monotonic())
        observation = MultiViewObservation({"head": frame})
        for raw in (
            {"recommended_skill": ["recover_package"], "scene_state": "x", "task_progress": "x",
             "target_object": None, "reason": "x", "requires_verification": True},
            {"recommended_skill": "invented", "scene_state": "x", "task_progress": "x",
             "target_object": None, "reason": "x", "requires_verification": True},
            {"recommended_skill": "recover_package", "target_object": None, "reason": "x",
             "requires_verification": True},
        ):
            reasoner = SceneReasoner(self.registry, SimpleNamespace(generate_json=lambda *args, value=raw, **kwargs: (value, .01)))
            recommendation, _ = reasoner.reason(observation)
            self.assertEqual(recommendation.recommended_skill, "stop_and_verify")


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
    def test_compressed_ros_image_decodes_without_cv_bridge(self):
        stream = BytesIO()
        Image.new("RGB", (4, 3), (20, 80, 160)).save(stream, format="JPEG")
        decoded = decode_compressed_rgb(stream.getvalue())
        self.assertEqual(decoded.mode, "RGB")
        self.assertEqual(decoded.size, (4, 3))

    def test_cv_abnormality_persistent_triggers(self):
        trigger = EventTrigger(persistence_hits=2, cooldown_seconds=0)
        p = {"geometry_validity": "bbox_approximation", "candidate_tracks": [{"track_id": "p1", "track_valid": True,
            "partial": False, "track_source": "detector", "confidence": .9, "body_side_violation": True}]}
        self.assertEqual(trigger.update(p, {}, {}, now=1), [])
        self.assertEqual(trigger.update(p, {}, {}, now=2), [])

    def test_cv_outside_event_requires_complete_visible_package_mask(self):
        trigger = EventTrigger(persistence_hits=1, cooldown_seconds=0)
        candidate = {"track_id": "p1", "track_valid": True, "partial": False, "occluded": False,
            "track_source": "grounding_dino_sam2", "label": "blue mailer bag", "confidence": .9,
            "mask_available": True, "mask_geometry_valid": True, "mask_confidence": .9,
            "detector_age_seconds": .1, "platform_visible_pixels": 100, "mask_body_side_extension": True}
        events = trigger.update({"candidate_tracks": [candidate]}, {}, {}, now=1)
        self.assertEqual([event.event_type for event in events], ["POSSIBLE_PACKAGE_OUTSIDE_PLATFORM"])
        candidate["occluded"] = True
        events = trigger.update({"candidate_tracks": [candidate]}, {}, {}, now=2)
        self.assertNotIn("POSSIBLE_PACKAGE_OUTSIDE_PLATFORM", [event.event_type for event in events])
        candidate["occluded"] = False
        candidate["platform_visible_pixels"] = 0
        events = trigger.update({"candidate_tracks": [candidate]}, {}, {}, now=3)
        self.assertEqual(events, [])

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
        base = scheduler._created_at
        frame = CameraFrame("head", np.zeros((2, 2, 3)), None, base + 10.995)
        obs = MultiViewObservation({"head": frame})
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

    def test_result_age_records_completion_separately_from_late_consumption(self):
        class Reasoner:
            def reason(self, obs): return ("ok", .01)
        scheduler = VLMInvocationScheduler(Reasoner(), periodic_interval=0, max_observation_age=1, max_result_age=3)
        from recovery_harness.models import MultiViewObservation
        received = time.monotonic()
        frame = CameraFrame("head", np.zeros((2, 2, 3)), None, received)
        obs = MultiViewObservation({"head": frame})
        scheduler.tick(obs)
        deadline = time.monotonic() + 1
        while scheduler._future is not None and not scheduler._future.done() and time.monotonic() < deadline:
            time.sleep(.002)
        result = scheduler.poll(now=received + 4)
        scheduler.close()
        self.assertFalse(result["valid"])
        self.assertLess(result["result_age_at_completion_seconds"], 3)
        self.assertGreater(result["result_age_at_consumption_seconds"], 3)
        self.assertEqual(result["discard_reason"], "stale_before_consumption")

    def test_mock_backend_never_uses_robot_interface(self):
        backend = MockExecutionBackend(); executor = ActionExecutor(True, backend)
        executor.execute_action([0.0, 1.0])
        self.assertEqual(backend.get_execution_status(), {"status": "simulated", "action_count": 1, "robot_motion": False})

    def test_missing_real_backend_fails_closed(self):
        executor = ActionExecutor(enabled=True, backend=None)
        with self.assertRaises(RuntimeError): executor.execute_action([0.0])


if __name__ == "__main__":
    unittest.main()

