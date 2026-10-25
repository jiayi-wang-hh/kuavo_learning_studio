from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import yaml

from .controller import RecoveryController
from .models import CameraFrame, MultiViewObservation
from .observation import ObservationSynchronizer
from .perception import CandidateTracker, PerceptionAdapter, TemporalHistory
from .policies import ActionExecutor, LingBotPolicyAdapter
from .registry import SkillRegistry
from .tactile import BinaryContactProcessor
from .vlm import QwenVLMAdapter, SceneReasoner


def _image(path: str):
    from PIL import Image
    return Image.open(path).convert("RGB")


def _load(config_path: str):
    path = Path(config_path).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.parent.parent
    registry_path = (path.parent / cfg["skill_registry"]).resolve()
    return cfg, root, SkillRegistry.from_yaml(registry_path)


def _vlm(cfg):
    config = cfg["vlm"]
    return QwenVLMAdapter(config["backend"], config["model_path"], device_map=config.get("device_map", "auto"), dtype=config.get("dtype", "bfloat16"))


def _perception(cfg):
    roi_path = Path(cfg["perception"]["roi_config"]).expanduser()
    roi = json.loads(roi_path.read_text(encoding="utf-8")) if roi_path.is_file() else {}
    plugin = cfg["perception"].get("detector_plugin")
    detector = None
    if plugin:
        import importlib
        module_name, function_name = plugin.split(":", 1)
        detector = getattr(importlib.import_module(module_name), function_name)
    tracker = CandidateTracker(detector=detector, max_track_age_seconds=float(cfg["perception"].get("max_track_age_seconds", 1.5)))
    image_size = (roi.get("image_width"), roi.get("image_height")) if roi.get("image_width") and roi.get("image_height") else None
    adapter = PerceptionAdapter(tracker, roi.get("platform_polygon"), roi.get("body_side_edge"), image_size=image_size)
    return adapter


def _reason_observation(reasoner, perception, frame_map, robot_state=None, tactile=None, history=None):
    head = frame_map.get("head")
    evidence = perception.inspect(head.image, view_name="head") if head else {"package_visible": "unknown", "geometry_validity": "unreliable"}
    obs = MultiViewObservation(
        images=frame_map,
        tactile=tactile or {}, robot_state=robot_state or {"valid": False},
        perception=evidence, history=history or [],
    )
    recommendation, latency = reasoner.reason(obs)
    age = (time.monotonic() - head.received_at) if head else None
    return obs, recommendation, latency, age


def _jsonl(path: str):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                record = json.loads(line)
                record["_line"] = line_number
                yield record


def preflight(config_path: str) -> int:
    cfg, root, registry = _load(config_path)
    report = {"preflight_only": True, "vlm_backend": cfg["vlm"]["backend"], "skills": []}
    failed = False
    for skill in registry:
        item = {"skill_id": skill.skill_id, "checkpoint": skill.policy_checkpoint}
        try:
            item["resolved_checkpoint"] = str(registry.resolve_checkpoint(skill))
            item["checkpoint_ready"] = True
        except Exception as exc:
            item.update({"checkpoint_ready": False, "checkpoint_error": str(exc)})
            failed = True
        item["base_model_exists"] = Path(skill.base_model).is_dir()
        item["model_repo_root_exists"] = Path(skill.model_repo_root).is_dir()
        item["tokenizer_exists"] = Path(skill.tokenizer).is_dir()
        item["norm_stats_exists"] = Path(skill.norm_stats).is_file()
        item["training_config_exists"] = Path(skill.training_config).is_file()
        item["robot_config_exists"] = Path(skill.robot_config).is_file()
        if any(item[key] is False for key in ("base_model_exists", "model_repo_root_exists", "tokenizer_exists", "norm_stats_exists", "training_config_exists", "robot_config_exists")):
            failed = True
        report["skills"].append(item)
    report["vlm_model_exists"] = Path(cfg["vlm"]["model_path"]).is_dir()
    if not report["vlm_model_exists"]:
        failed = True
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if failed else 0


def offline(config_path: str, manifest: str, max_frames: int = 10, output_path: str | None = None) -> int:
    cfg, _, registry = _load(config_path)
    reasoner = SceneReasoner(registry, _vlm(cfg))
    perception = _perception(cfg)
    policies = LingBotPolicyAdapter(registry)
    controller = RecoveryController(registry, policies, ActionExecutor(False), max_observation_age=cfg["runtime"]["max_observation_age_seconds"])
    history = TemporalHistory()
    enabled = ["head"] + [name for name in ("left_wrist", "right_wrist") if cfg["cameras"][name]["enabled"]]
    count = 0
    for record in _jsonl(manifest):
        if count >= max_frames:
            break
        frames = {}
        for view in enabled:
            img_path = record.get(f"{view}_image")
            frames[view] = CameraFrame(view, _image(img_path), record.get(f"{view}_timestamp"), time.monotonic(), "manifest") if img_path else None
        obs, recommendation, latency, _ = _reason_observation(reasoner, perception, frames, record.get("robot_state"), record.get("tactile"), history.snapshot())
        decision = controller.decide(recommendation, observation_age=0.0 if frames.get("head") else None,
                                     robot_state=record.get("robot_state", {"valid": False}), dry_run=True)
        output = {"manifest_line": record["_line"], "label": record.get("label"),
                  "input_images": {view: record.get(f"{view}_image") for view in enabled if record.get(f"{view}_image")},
                  "inference_latency_seconds": latency,
                  "vlm_recommendation": recommendation.raw, "controller_decision": decision}
        serialized = json.dumps(output, ensure_ascii=False, default=str)
        print(serialized, flush=True)
        if output_path:
            with Path(output_path).open("a", encoding="utf-8") as stream:
                stream.write(serialized + "\n")
        history.append({"recommendation": recommendation.raw, "perception": obs.perception})
        count += 1
    policies.shutdown()
    return 0


def live_dry_run(config_path: str, *, max_observations: int = 0) -> int:
    import rospy
    cfg, _, registry = _load(config_path)
    rospy.init_node("kuavo_vlm_skill_router_dry_run", anonymous=False, disable_signals=True)
    enabled = ["head"] + [name for name in ("left_wrist", "right_wrist") if cfg["cameras"][name]["enabled"]]
    synchronizer = ObservationSynchronizer(enabled, cfg["runtime"]["max_observation_age_seconds"])
    tactile = BinaryContactProcessor(cfg["tactile"]["finger_force_thresholds"], cfg["tactile"].get("valid_status_values"), cfg["tactile"]["sustain_seconds"])
    from .ros_bridge import KuavoSensorBridge
    bridge = KuavoSensorBridge(cfg, synchronizer, tactile)
    reasoner = SceneReasoner(registry, _vlm(cfg))
    perception, policies = _perception(cfg), LingBotPolicyAdapter(registry)
    controller = RecoveryController(registry, policies, ActionExecutor(False), max_observation_age=cfg["runtime"]["max_observation_age_seconds"])
    history = TemporalHistory()
    count = 0
    try:
        while not rospy.is_shutdown() and (max_observations <= 0 or count < max_observations):
            now = time.monotonic()
            observation = synchronizer.snapshot(perception={})
            freshness = synchronizer.freshness(now=now)
            if not observation.available_views:
                rospy.sleep(0.05)
                continue
            robot_state_for_vlm = dict(synchronizer.robot_state)
            robot_age_before = now - robot_state_for_vlm.get("received_at", 0.0) if robot_state_for_vlm else float("inf")
            if robot_age_before > cfg["runtime"]["max_observation_age_seconds"]:
                robot_state_for_vlm["valid"] = False
            tactile_age_before = now - synchronizer.tactile_received_at if synchronizer.tactile_received_at else float("inf")
            tactile_for_vlm = synchronizer.tactile if tactile_age_before <= cfg["runtime"]["max_observation_age_seconds"] else {"left_multifinger_contact": None, "right_multifinger_contact": None}
            frame_map = observation.images
            obs, recommendation, latency, _ = _reason_observation(reasoner, perception, frame_map,
                robot_state_for_vlm, tactile_for_vlm, history.snapshot())
            # Re-read sensor age after the potentially multi-second VLM call.
            post_inference_freshness = synchronizer.freshness()
            # Check the exact frame used for reasoning; a newer buffer entry cannot make an old
            # recommendation fresh after a long VLM call.
            used_view_ages = {name: time.monotonic() - frame.received_at for name, frame in frame_map.items() if frame is not None}
            head_age = used_view_ages.get("head")
            recommendation_age = max(used_view_ages.values(), default=None)
            current_robot_state = dict(synchronizer.robot_state)
            robot_age = time.monotonic() - current_robot_state.get("received_at", 0.0) if current_robot_state else float("inf")
            if robot_age > cfg["runtime"]["max_observation_age_seconds"]:
                current_robot_state["valid"] = False
            tactile_age = time.monotonic() - synchronizer.tactile_received_at if synchronizer.tactile_received_at else float("inf")
            current_tactile = synchronizer.tactile if tactile_age <= cfg["runtime"]["max_observation_age_seconds"] else {"left_multifinger_contact": None, "right_multifinger_contact": None}
            decision = controller.decide(recommendation, observation_age=recommendation_age,
                robot_state=current_robot_state, dry_run=True)
            print(json.dumps({"inference_latency_seconds": latency, "image_freshness_seconds": post_inference_freshness,
                              "reasoning_frame_ages_seconds": used_view_ages,
                              "robot_state_age_seconds": robot_age, "tactile_age_seconds": tactile_age,
                              "perception": obs.perception, "tactile": current_tactile,
                              "vlm_recommendation": recommendation.raw, "controller_decision": decision},
                             ensure_ascii=False, default=str), flush=True)
            history.append({"recommendation": recommendation.raw, "perception": obs.perception})
            count += 1
            rospy.sleep(cfg["runtime"]["reasoning_interval_seconds"])
    finally:
        bridge.close()
        policies.shutdown()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="VLM-guided Kuavo skill routing (motion disabled by default)")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config" / "package_sorting.yaml"))
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("preflight", help="Check configured checkpoint and resource paths; no models are loaded")
    offline_parser = sub.add_parser("offline", help="Replay a JSONL image manifest; no robot actions are sent")
    offline_parser.add_argument("--manifest", required=True)
    offline_parser.add_argument("--max-frames", type=int, default=10)
    offline_parser.add_argument("--output", help="Append structured results as JSONL to this file")
    live = sub.add_parser("live-dry-run", help="Read ROS sensors and reason; creates no action publishers")
    live.add_argument("--max-observations", type=int, default=0, help="0 means until Ctrl-C")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.mode == "preflight":
        return preflight(args.config)
    if args.mode == "offline":
        if not 1 <= args.max_frames <= 10:
            parser.error("--max-frames must be between 1 and 10 for bounded replay")
        return offline(args.config, args.manifest, max_frames=args.max_frames, output_path=args.output)
    return live_dry_run(args.config, max_observations=args.max_observations)


if __name__ == "__main__":
    raise SystemExit(main())

