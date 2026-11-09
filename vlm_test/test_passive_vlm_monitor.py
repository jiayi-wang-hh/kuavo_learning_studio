"""User-run offline unit checks; no ROS master, model, SDK, or robot required."""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from vlm_test.passive_vlm_monitor import (
    ExecutionEvidence, Journal, LatestRGB, RGBPacket, classify, freshness, strict_skill, image_header_stamp,
    reference_calibration, select_prompt,
)


def packet(stamp=100.0, received=10.0, epoch=0, data=b""):
    return RGBPacket(1, stamp, int((stamp or 0) * 1e9), epoch, received, 1000.0,
                     100.0, "head_color", 1, "jpeg", "/camera", data)


def clock(epoch=0, sim=False, advanced=10.0):
    return dict(epoch=epoch, use_sim_time=sim, last_advance_mono=advanced)


def message(stamp):
    return SimpleNamespace(header=SimpleNamespace(
        stamp=SimpleNamespace(to_sec=lambda: stamp, to_nsec=lambda: int(stamp * 1e9)),
        seq=1, frame_id="head_color"), format="jpeg", data=b"image")


class FreshnessTests(unittest.TestCase):
    def test_positive_wire_fields_do_not_require_to_sec(self):
        stamp = SimpleNamespace(secs=11863, nsecs=424928000)
        seconds, nanos, secs, nsecs, error = image_header_stamp(stamp)
        self.assertAlmostEqual(seconds, 11863.424928)
        self.assertEqual(nanos, 11863424928000)
        self.assertEqual((secs, nsecs, error), (11863, 424928000, None))
        self.assertEqual(image_header_stamp(SimpleNamespace(secs=0, nsecs=0))[0], None)

    def test_device_uptime_is_not_unix_capture_time(self):
        self.assertEqual(freshness(packet(stamp=11863.424928), 1793181873, 10.1, clock(), .5),
                         "header_clock_domain_mismatch")

    def test_fresh_receipt_cannot_hide_old_header(self):
        self.assertEqual(freshness(packet(stamp=90), 100.1, 10.1, clock(), .5), "stale_frame")

    def test_header_and_receipt_both_required(self):
        self.assertIsNone(freshness(packet(), 100.2, 10.2, clock(), .5))
        self.assertEqual(freshness(packet(received=8), 100.2, 10.2, clock(), .5), "stale_frame")
        self.assertEqual(freshness(packet(stamp=None), 100.2, 10.2, clock(), .5), "missing_header_stamp")
        self.assertEqual(freshness(packet(stamp=101), 100.2, 10.2, clock(), .5), "future_header_stamp")

    def test_late_result_and_paused_sim_clock(self):
        self.assertEqual(freshness(packet(), 102, 12, clock(), 1.5), "stale_frame")
        self.assertEqual(freshness(packet(), 100, 12, clock(sim=True), .5), "ros_clock_stalled")
        self.assertEqual(freshness(packet(), 0, 10, clock(), .5), "ros_time_unavailable")

    def test_latest_only_and_header_regression(self):
        slot = LatestRGB()
        slot.receive(message(100), 100.1, 10, 1000)
        slot.receive(message(100), 100.2, 10.1, 1000.1)
        slot.receive(message(99), 100.3, 10.2, 1000.2)
        current, _ = slot.snapshot(100.4, 10.3)
        self.assertEqual(current.generation, 1)
        self.assertEqual(slot.rejected, 2)
        slot.receive(message(100.4), 100.5, 10.4, 1000.4)
        self.assertEqual(slot.snapshot(100.5, 10.5)[0].stamp_s, 100.4)

    def test_bag_rewind_invalidates_in_flight_epoch(self):
        slot = LatestRGB(use_sim_time=True)
        slot.receive(message(100), 100.1, 10, 1000)
        old, _ = slot.snapshot(100.2, 10.1)
        new, state = slot.snapshot(10, 10.2)
        self.assertIsNone(new)
        self.assertEqual(freshness(old, 10, 10.2, state, 1.5), "ros_clock_epoch_changed")
        slot.receive(message(10.1), 10.2, 10.3, 1000.3)
        self.assertEqual(slot.snapshot(10.3, 10.4)[0].epoch, 1)


class OutputTests(unittest.TestCase):
    def test_revised_task_semantics_apply_with_or_without_vla(self):
        self.assertEqual(select_prompt(camera_only=True).name, 'package_front_edge_tolerant_v4.json')
        self.assertEqual(select_prompt(camera_only=False).name, 'package_front_edge_tolerant_v4.json')
        explicit = Path('/tmp/custom-prompt.json')
        self.assertEqual(select_prompt(explicit, True), explicit)

    def test_reference_roi_is_audit_only_not_current_coordinates(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / 'roi.json'
            p.write_text(json.dumps(dict(platform_polygon=[[351,272],[329,437],[539,437],[511,275]],
                                         body_side_edge_indices=[1,2], image_width=848, image_height=480)))
            reference = reference_calibration(p)
            self.assertFalse(reference['used_as_current_pixel_roi'])
            self.assertFalse(reference['sent_as_model_image'])
            prompt = json.loads(select_prompt(camera_only=True).read_text())['skill_prompt']
            self.assertNotIn('437', prompt)
            self.assertNotIn('329', prompt)
            self.assertIn('re-identify', prompt.lower())

    def test_minor_overhang_is_not_an_automatic_recovery_condition(self):
        config = json.loads(select_prompt(camera_only=True).read_text())
        self.assertIn('Crossing the platform edge is NOT sufficient', config['skill_prompt'])
        self.assertIn('explicitly labels the following as NORMAL', config['skill_prompt'])
        self.assertFalse(any(e['sent_to_model'] for e in config['reference_examples']))

    def test_strict_binary_contract(self):
        for skill in ["normal", "recovery"]:
            self.assertEqual(strict_skill(json.dumps({"skill": skill})), {"skill": skill})
        for raw in ['normal', '{"skill":"unknown"}', '{"skill":"normal","reason":"ok"}',
                    '```json\n{"skill":"normal"}\n```', '{"skill":"normal","skill":"recovery"}']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                strict_skill(raw)

    def test_failure_is_never_normal_and_exact_rgb_is_saved(self):
        from io import BytesIO
        from PIL import Image
        image = Image.new("RGB", (848, 480), (220, 30, 80))
        data = BytesIO()
        image.save(data, format="PNG")

        class FakeAdapter:
            def generate_raw(self, image, prompt, *, before_generate, **kwargs):
                before_generate()
                raise RuntimeError("simulated GPU failure")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            row = classify(FakeAdapter(), packet(data=data.getvalue()), "prompt", root,
                           640, lambda p: None, lambda: 100.2)
            self.assertEqual(row["status"], "inference_error")
            self.assertIsNone(row["parsed"])
            self.assertIsNone(row["raw_output"])
            with Image.open(row["input_image"]) as saved:
                self.assertEqual(saved.size, (640, 362))
                self.assertEqual(saved.getpixel((0, 0)), (220, 30, 80))

    def test_vla_log_is_observed_without_replaying_old_activity(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "vla.log"
            log.write_text("Step 99 time: 0.1s\n")
            journal = Journal(root)
            observer = ExecutionEvidence(log, journal)
            observer.poll()
            self.assertEqual(observer.snapshot()["status"], "unknown")
            with log.open("a") as f:
                f.write("Step 1 time: 0.1s\n")
            observer.poll()
            self.assertEqual(observer.snapshot()["last_evidence"]["step"], 1)
            log.write_text("different task starting\n")
            observer.poll()
            self.assertEqual(observer.snapshot()["status"], "unknown")
            journal.stream.close()

    def test_entrypoint_has_no_robot_actuation_calls(self):
        source = Path(__file__).with_name("passive_vlm_monitor.py").read_text()
        calls = {node.func.attr for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(calls & {"Publisher", "ServiceProxy", "publish", "register_publisher",
                                 "register_service", "kill", "Popen", "control_arm_joint_positions"})


if __name__ == "__main__":
    unittest.main()
