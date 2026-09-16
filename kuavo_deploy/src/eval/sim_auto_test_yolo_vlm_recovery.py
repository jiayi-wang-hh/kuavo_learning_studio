#!/usr/bin/env python3
"""YOLO/joint-state stage-1 trigger with external Qwen3.5 stage-2 recovery.

Stage 1 is deliberately the lightweight online YOLO monitor.  Qwen is only
started after a confirmed YOLO event, from its isolated environment, and
returns a failure type plus a deterministic recovery action.
"""
from __future__ import annotations

import csv
import datetime
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import rospy
import torch
from std_msgs.msg import Bool
from std_srvs.srv import Trigger, TriggerRequest

from kuavo_deploy.kuavo_env.KuavoSimEnv import KuavoSimEnv  # noqa: F401
from kuavo_deploy.src.eval import sim_auto_test as base
from kuavo_deploy.src.eval.sim_auto_test_vlm_agentic import (
    TriggerResult,
    VLMFailureVerifier,
    VLMVerifierConfig,
)
from kuavo_deploy.src.eval.sim_auto_test_yolo_robot_state import EePoseBuffer, rgb, state_by_side
from kuavo_deploy.utils.policy_loader import inject_task_prompt, resolve_eval_output_dir
from kuavo_deploy.utils.ros_manager import ROSManager


def _write_clip(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise RuntimeError("Cannot run stage-2 verification without head-camera frames")
    imageio.mimsave(str(path), frames, fps=fps, codec="libx264")


def _stage1_result(episode: int, step: int, event: dict, clip_path: Path) -> TriggerResult:
    sides = event["sides"] or "unknown"
    return TriggerResult(
        episode=episode, source_step=step, start_s=0.0, end_s=event["time_s"],
        completed_at=time.perf_counter(), raw_decision="PAUSE", final_decision="PAUSE",
        confidence="MEDIUM", evidence=(
            f"YOLO/joint-state inconsistency on {sides}: robot joint state moved "
            "while the assigned toy remained visually static."
        ), guard_reason="YOLO_ROBOT_STATE", schema_valid=True, raw_output="",
        inference_seconds=0.0, clip_path=str(clip_path), current_phase="UNKNOWN",
        expected_effect="The grasp/manipulation action should move the task object.",
        observed_effect=f"The {sides} assigned toy remained static while the robot moved.",
    )


def run_episode(config, policy, preprocessor, postprocessor, episode, attempt, output_dir, pause_pub, verifier, events):
    """Return (outcome, success, steps, verification), where outcome is done/retry/stop."""
    cfg = config.inference
    from yolo.online_robot_state_trigger import OnlineRobotStateTrigger

    env = gym.make(config.env.env_name, max_episode_steps=cfg.max_episode_steps, config=config)
    ros_manager = ROSManager()
    ros_manager.register_subscriber("/simulator/success", Bool, base.env_success_callback)
    ee_poses = EePoseBuffer(ros_manager, cfg)
    monitor = OnlineRobotStateTrigger(cfg, output_dir / f"episode_{episode:03d}_attempt_{attempt:02d}_yolo_trigger.csv")
    policy.reset()
    observation, _ = env.reset(seed=cfg.seed + episode)
    rospy.ServiceProxy("/simulator/start", Trigger)(TriggerRequest())
    cam_keys = [key for key in observation if "images" in key or "depth" in key]
    frames = {key: [] for key in cam_keys}
    head_frames: list[np.ndarray] = []
    step, verification = 0, None
    try:
        while True:
            if not base.check_control_signals():
                return "stop", 0, step, verification
            policy_obs = observation
            if cfg.policy_type != "client":
                policy_obs = preprocessor(inject_task_prompt(policy_obs, cfg.task_prompt))
            with torch.inference_mode():
                action = policy.select_action(policy_obs)
            if cfg.policy_type != "client":
                action = postprocessor(action)
            if torch.is_tensor(action):
                action_np = action.squeeze(0).detach().cpu().numpy()
            else:
                action_np = np.asarray(action)
                if action_np.ndim > 1 and action_np.shape[0] == 1:
                    action_np = action_np.squeeze(0)
            observation, _, terminated, truncated, _ = env.step(action_np)
            now_s = rospy.Time.now().to_sec()
            head = rgb(observation, "observation.images.head_cam_h")
            event = monitor.update(head, now_s, ee_poses.by_side(), state_by_side(observation, config.env.which_arm, now_s))
            head_frames.append(head)
            for key in cam_keys:
                frames[key].append(rgb(observation, key))
            step += 1
            if event:
                pause_pub.publish(True)
                clip = output_dir / "stage2_clips" / f"episode_{episode:03d}_attempt_{attempt:02d}_step_{step:05d}.mp4"
                clip.parent.mkdir(parents=True, exist_ok=True)
                _write_clip(clip, head_frames, env.unwrapped.ros_rate)
                trigger = _stage1_result(episode, step, event, clip)
                verification = verifier.verify(trigger)
                record = {"episode": episode, "attempt": attempt, "step": step, "stage1": asdict(trigger), "stage2": asdict(verification)}
                events.write(json.dumps(record, ensure_ascii=False) + "\n"); events.flush()
                base.log_robot.warning("YOLO stage-1 pause: episode=%d step=%d sides=%s; stage-2: type=%s action=%s", episode, step, event["sides"], verification.failure_type, verification.recovery_action)
                if verification.recovery_action == "RESUME":
                    pause_pub.publish(False)
                    monitor.reset_episode()
                    head_frames.clear()
                    continue
                if verification.recovery_action == "RESET_AND_RETRY":
                    return "retry", 0, step, verification
                return "stop", 0, step, verification
            if terminated or truncated or base.success_evt.is_set():
                return "done", int(base.success_evt.is_set()), step, verification
    finally:
        pause_pub.publish(False)
        for key, sequence in frames.items():
            base.save_rollout_video(output_dir / f"episode_{episode:03d}_attempt_{attempt:02d}_{key}.mp4", sequence, env.unwrapped.ros_rate)
        monitor.close()
        env.close()
        ros_manager.close()


def evaluate(config, verifier_config: VLMVerifierConfig) -> None:
    cfg = config.inference
    if not cfg.failure_trigger_enabled or not cfg.failure_trigger_allow_joint_state_fallback:
        raise ValueError("Enable failure_trigger_enabled and failure_trigger_allow_joint_state_fallback for this evaluator")
    checkpoint = Path(cfg.pretrained_path) if cfg.pretrained_path else Path(f"outputs/train/{cfg.task}/{cfg.method}/{cfg.timestamp}/epoch{cfg.epoch}")
    output_dir = resolve_eval_output_dir(checkpoint, Path("outputs/eval")) / datetime.datetime.now().strftime("yolo_vlm_recovery_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    policy, preprocessor, postprocessor, _ = base.setup_policy(checkpoint, cfg.policy_type, torch.device(cfg.device), task_prompt=cfg.task_prompt)
    verifier = VLMFailureVerifier(verifier_config, cfg.task_prompt)
    reset = rospy.ServiceProxy("/simulator/reset", Trigger)
    init_service = rospy.Service("/simulator/init", Trigger, base.env_init_service)
    pause_pub = rospy.Publisher(cfg.failure_trigger_pause_topic, Bool, queue_size=1)
    events_path = output_dir / "stage2_failure_events.jsonl"
    rows = []
    try:
        base.init_evt.clear()
        deadline = time.monotonic() + 30
        while not base.init_evt.is_set() and time.monotonic() < deadline:
            base.log_robot.info("Waiting for initial simulator init"); time.sleep(1)
        if not base.init_evt.is_set():
            raise RuntimeError("Simulator did not call /simulator/init during startup")
        base.safe_reset_service(reset); base.init_evt.clear()
        with events_path.open("w", encoding="utf-8") as events:
            for episode in range(cfg.eval_episodes):
                attempt, finished = 0, False
                while not finished:
                    deadline = time.monotonic() + 30
                    while not base.init_evt.is_set() and time.monotonic() < deadline:
                        if not base.check_control_signals():
                            return
                        base.log_robot.info("Waiting for simulator init for episode %d", episode); time.sleep(1)
                    if not base.init_evt.is_set():
                        raise RuntimeError(f"Simulator did not call /simulator/init for episode {episode}")
                    outcome, success, steps, verdict = run_episode(config, policy, preprocessor, postprocessor, episode, attempt, output_dir, pause_pub, verifier, events)
                    rows.append({"episode": episode, "attempt": attempt, "outcome": outcome, "success": bool(success), "steps": steps, "failure_type": "" if verdict is None else verdict.failure_type, "recovery_action": "" if verdict is None else verdict.recovery_action})
                    if outcome == "retry" and attempt < verifier.config.max_retries_per_episode:
                        attempt += 1
                        base.safe_reset_service(reset); base.init_evt.clear(); base.success_evt.clear(); gc.collect()
                        continue
                    if outcome == "stop":
                        return
                    finished = True
                    base.safe_reset_service(reset); base.init_evt.clear(); base.success_evt.clear(); gc.collect()
    finally:
        init_service.shutdown(); pause_pub.publish(False)
        with (output_dir / "yolo_vlm_recovery_episode_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["episode", "attempt", "outcome", "success", "steps", "failure_type", "recovery_action"])
            writer.writeheader(); writer.writerows(rows)
        (output_dir / "evaluation_yolo_vlm_recovery.log").write_text(f"episodes={len(rows)}\nsuccess_count={sum(row['success'] for row in rows)}\n", encoding="utf-8")
