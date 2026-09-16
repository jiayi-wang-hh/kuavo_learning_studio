#!/usr/bin/env python3
"""Closed-loop checkpoint evaluation with online YOLO + robot-state PAUSE.

This is intentionally a separate evaluator based on ``sim_auto_test.py``.
It loads the normal checkpoint, runs the ordinary simulator loop, and adds a
same-step detector after ``env.step``.  It never parses the text deploy log:
the detector receives the in-memory ``observation.state`` that produces the
``STATE: ...`` log record, avoiding log buffering and timestamp skew.

Run directly:
  python -m kuavo_deploy.src.eval.sim_auto_test_yolo_robot_state --config CFG
"""
from __future__ import annotations

import argparse
import csv
import datetime
import gc
from pathlib import Path
import time

import gymnasium as gym
import imageio
import numpy as np
import rospy
import torch
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from std_srvs.srv import Trigger, TriggerRequest

from kuavo_deploy.config import load_kuavo_config
from kuavo_deploy.src.eval import sim_auto_test as base
from kuavo_deploy.kuavo_env.KuavoSimEnv import KuavoSimEnv  # noqa: F401
from kuavo_deploy.utils.policy_loader import inject_task_prompt, resolve_eval_output_dir
from kuavo_deploy.utils.ros_manager import ROSManager


def rgb(observation, key):
    image = observation[key].squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(image * 255, 0, 255).astype(np.uint8)


def state_by_side(observation, which_arm, timestamp):
    """Return state samples identical to the values logged by KuavoBaseRosEnv."""
    state = observation["observation.state"].squeeze(0).detach().cpu().numpy().reshape(-1)
    result = {"left": None, "right": None}
    if which_arm == "both":
        middle = len(state) // 2
        result["left"] = (timestamp, tuple(float(x) for x in state[:middle]))
        result["right"] = (timestamp, tuple(float(x) for x in state[middle:]))
    elif which_arm in result:
        result[which_arm] = (timestamp, tuple(float(x) for x in state))
    else:
        raise ValueError(f"Unsupported which_arm: {which_arm}")
    return result


class EePoseBuffer:
    """Latest Cartesian EE samples from simulator PoseStamped topics.

    Positions are assumed to share a world/base frame.  The trigger uses XY
    speed for its planar test and logs Z speed independently.
    """
    def __init__(self, ros_manager, cfg):
        self.samples = {"left": None, "right": None}
        for side, topic in (("left", cfg.failure_trigger_ee_pose_topic_left), ("right", cfg.failure_trigger_ee_pose_topic_right)):
            if topic:
                ros_manager.register_subscriber(topic, PoseStamped, self._callback(side))
                base.log_robot.info("YOLO trigger subscribed to %s EE pose: %s", side, topic)

    def _callback(self, side):
        def receive(msg):
            stamp = msg.header.stamp.to_sec()
            if stamp <= 0:
                stamp = rospy.Time.now().to_sec()
            point = msg.pose.position
            self.samples[side] = (stamp, (float(point.x), float(point.y), float(point.z)))
        return receive

    def by_side(self):
        return dict(self.samples)


def run_episode(config, policy, preprocessor, postprocessor, episode, output_dir, pause_pub):
    cfg = config.inference
    if not cfg.failure_trigger_enabled:
        raise ValueError("Set inference.failure_trigger_enabled: true for this evaluator")
    if not cfg.failure_trigger_allow_joint_state_fallback:
        raise ValueError(
            "This evaluator uses observation.state (joint_q + gripper) only; "
            "set failure_trigger_allow_joint_state_fallback: true"
        )

    from yolo.online_robot_state_trigger import OnlineRobotStateTrigger
    env = gym.make(config.env.env_name, max_episode_steps=cfg.max_episode_steps, config=config)
    ros_manager = ROSManager()
    ros_manager.register_subscriber("/simulator/success", Bool, base.env_success_callback)
    ee_poses = EePoseBuffer(ros_manager, cfg)
    monitor = OnlineRobotStateTrigger(cfg, output_dir / f"rollout_{episode}_yolo_robot_state_trigger.csv")

    policy.reset()
    observation, _ = env.reset(seed=cfg.seed + episode)
    rospy.ServiceProxy("/simulator/start", Trigger)(TriggerRequest())
    cam_keys = [key for key in observation if "images" in key or "depth" in key]
    frames = {key: [] for key in cam_keys}
    step, done, yolo_pause = 0, False, False
    try:
        while not done:
            policy_obs = observation
            if cfg.policy_type != "client":
                policy_obs = inject_task_prompt(policy_obs, cfg.task_prompt)
                policy_obs = preprocessor(policy_obs)
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
            event = monitor.update(
                rgb(observation, "observation.images.head_cam_h"), now_s,
                ee_poses.by_side(), state_by_side(observation, config.env.which_arm, now_s),
            )
            for key in cam_keys:
                frames[key].append(rgb(observation, key))
            step += 1
            if event:
                yolo_pause = True
                pause_pub.publish(True)
                base.log_robot.warning("YOLO PAUSE episode=%d step=%d sides=%s", episode, step, event["sides"])
            done = bool(terminated or truncated or event or base.success_evt.is_set())

        for key, sequence in frames.items():
            base.save_rollout_video(output_dir / f"rollout_{episode}_{key}.mp4", sequence, env.unwrapped.ros_rate)
        return int(base.success_evt.is_set()), int(yolo_pause), step
    finally:
        monitor.close()
        pause_pub.publish(False)
        env.close()
        ros_manager.close()


def evaluate(config):
    cfg = config.inference
    checkpoint = Path(cfg.pretrained_path) if cfg.pretrained_path else Path(f"outputs/train/{cfg.task}/{cfg.method}/{cfg.timestamp}/epoch{cfg.epoch}")
    output_dir = resolve_eval_output_dir(checkpoint, Path("outputs/eval")) / datetime.datetime.now().strftime("yolo_robot_state_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    policy, preprocessor, postprocessor, _ = base.setup_policy(checkpoint, cfg.policy_type, torch.device(cfg.device), task_prompt=cfg.task_prompt)
    reset = rospy.ServiceProxy("/simulator/reset", Trigger)
    init_service = rospy.Service("/simulator/init", Trigger, base.env_init_service)
    pause_pub = rospy.Publisher(cfg.failure_trigger_pause_topic, Bool, queue_size=1)
    rows = []
    try:
        # Match sim_auto_test.py exactly: wait for the simulator's initial
        # ready notification, then reset, then wait for the post-reset ready
        # notification before creating KuavoSimEnv.  The two phases matter:
        # the gait-switch controller may not exist during early simulator boot.
        base.init_evt.clear()
        initial_deadline = time.monotonic() + 30.0
        while not base.init_evt.is_set() and time.monotonic() < initial_deadline:
            base.log_robot.info("Waiting for initial simulator init")
            time.sleep(1)
        if not base.init_evt.is_set():
            raise RuntimeError(
                "Simulator did not call /simulator/init during initial startup. "
                "Start the paired kuavo-ros-opensource auto-test simulator."
            )
        base.safe_reset_service(reset)
        base.init_evt.clear()
        for episode in range(cfg.eval_episodes):
            deadline = time.monotonic() + 30.0
            while not base.init_evt.is_set() and time.monotonic() < deadline:
                base.log_robot.info("Waiting for simulator init for episode %d", episode)
                time.sleep(1)
            if not base.init_evt.is_set():
                raise RuntimeError(
                    "Simulator did not call /simulator/init within 30 seconds. "
                    "Start the paired kuavo-ros-opensource auto-test simulator "
                    "and verify the reset/init services use the same ROS master."
                )
            success, paused, steps = run_episode(config, policy, preprocessor, postprocessor, episode, output_dir, pause_pub)
            rows.append({"episode": episode, "success": bool(success), "yolo_pause": bool(paused), "steps": steps})
            base.log_robot.info("Episode %d finished: success=%s yolo_pause=%s steps=%d", episode, bool(success), bool(paused), steps)
            base.safe_reset_service(reset)
            base.init_evt.clear(); base.success_evt.clear(); base.pause_flag.clear()
            gc.collect()
    finally:
        init_service.shutdown()
        pause_pub.publish(False)
    with (output_dir / "yolo_robot_state_episode_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "success", "yolo_pause", "steps"])
        writer.writeheader(); writer.writerows(rows)
    successes, pauses = sum(r["success"] for r in rows), sum(r["yolo_pause"] for r in rows)
    (output_dir / "evaluation_yolo_robot_state.log").write_text(f"checkpoint={checkpoint}\nepisodes={len(rows)}\nsuccess_count={successes}\nyolo_pause_count={pauses}\n", encoding="utf-8")
    base.log_robot.info("YOLO evaluation complete: success=%d/%d pauses=%d output=%s", successes, len(rows), pauses, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_kuavo_config(args.config)
    rospy.init_node("kuavo_yolo_robot_state_eval", anonymous=True)
    evaluate(config)


if __name__ == "__main__":
    main()
