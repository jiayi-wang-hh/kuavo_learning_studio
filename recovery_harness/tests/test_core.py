from pathlib import Path
import unittest
from types import SimpleNamespace

from recovery_harness.models import SkillRecommendation
from recovery_harness.perception import box_platform_overlap, crosses_body_edge
from recovery_harness.registry import SkillRegistry
from recovery_harness.controller import ControllerState, RecoveryController
from recovery_harness.tactile import BinaryContactProcessor


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


if __name__ == "__main__":
    unittest.main()

