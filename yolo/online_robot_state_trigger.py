"""Online YOLO-World + end-effector cross-signal PAUSE trigger.

Imported only by the simulator evaluator when ``failure_trigger_enabled`` is
true, keeping Ultralytics out of the ordinary evaluation path.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import tempfile
from collections import deque
from pathlib import Path
from types import SimpleNamespace


RELIABLE = {"DETECTED", "TRACKED"}


class ExternalYoloWorldWorker:
    """YOLO-World subprocess for a separate Python/Conda environment."""
    def __init__(self, python_path, model, device, conf, iou):
        worker = Path(__file__).with_name("yolo_world_worker.py")
        self.temp_dir = Path(tempfile.mkdtemp(prefix="kuavo_yolo_worker_"))
        self.image_path = self.temp_dir / "frame.jpg"
        self.process = subprocess.Popen(
            [str(python_path), str(worker), "--model", str(model), "--device", str(device), "--conf", str(conf), "--iou", str(iou)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
        )

    def predict(self, image):
        import cv2
        if not cv2.imwrite(str(self.image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Unable to write YOLO worker image: {self.image_path}")
        if self.process.poll() is not None:
            stderr = self.process.stderr.read()
            raise RuntimeError(f"YOLO worker exited ({self.process.returncode}): {stderr}")
        self.process.stdin.write(json.dumps({"image_path": str(self.image_path)}) + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError("YOLO worker returned no response")
        parsed = json.loads(response)
        if "error" in parsed:
            raise RuntimeError(f"YOLO worker error: {parsed['error']}")
        return parsed["detections"]

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def _distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _past(history, target):
    values = [item for item in history if item["t"] <= target]
    return values[-1] if values else None


class OnlineRobotStateTrigger:
    """Per-episode monitor. ``update`` returns an event dict or ``None``."""

    def __init__(self, cfg, output_csv: Path):
        # Import lazily: normal simulation evaluation does not require YOLO.
        from yolo.test_yolo_world_roi_filter import (
            CLASSES, SideTrack, box_center_norm, side_assignment,
            toy_rejection_reason, update_side_track,
        )
        self.CLASSES = CLASSES
        self.SideTrack = SideTrack
        self.box_center_norm = box_center_norm
        self.side_assignment = side_assignment
        self.toy_rejection_reason = toy_rejection_reason
        self.update_side_track = update_side_track
        self.external_worker = None
        if cfg.failure_trigger_yolo_python:
            self.external_worker = ExternalYoloWorldWorker(
                cfg.failure_trigger_yolo_python, cfg.failure_trigger_yolo_model,
                cfg.failure_trigger_yolo_device, cfg.failure_trigger_yolo_conf,
                cfg.failure_trigger_yolo_iou,
            )
            self.model = None
        else:
            from ultralytics import YOLOWorld
            self.model = YOLOWorld(cfg.failure_trigger_yolo_model)
            self.model.set_classes(CLASSES)
        self.device = cfg.failure_trigger_yolo_device
        self.conf = cfg.failure_trigger_yolo_conf
        self.iou = cfg.failure_trigger_yolo_iou
        self.window_s = cfg.failure_trigger_motion_window_s
        self.ee_min = cfg.failure_trigger_ee_motion_min
        self.ee_z_min = cfg.failure_trigger_ee_vertical_motion_min
        self.toy_max = cfg.failure_trigger_toy_motion_max
        self.duration_s = cfg.failure_trigger_duration_s
        self.confirm_frames = cfg.failure_trigger_confirm_frames
        self.pose_max_age_s = cfg.failure_trigger_pose_max_age_s
        self.start_after_s = cfg.failure_trigger_start_after_s
        self.allow_joint_state_fallback = cfg.failure_trigger_allow_joint_state_fallback
        self.episode_start_s = None
        self.frame = 0
        self.streak = 0
        self.triggered = False
        self.tracks = {"left": SideTrack("left_toy"), "right": SideTrack("right_toy")}
        self.state = {
            side: {"toy": deque(maxlen=1000), "ee": deque(maxlen=1000), "candidate_s": 0.0, "last_t": None}
            for side in self.tracks
        }
        # Same tracking/filter defaults as the offline visual tracker.
        self.args = SimpleNamespace(
            toy_max_area_ratio=0.07, toy_max_width_ratio=0.35, toy_max_height_ratio=0.35,
            toy_bottom_limit=0.96, roi_xmin=0.05, roi_xmax=0.95, roi_ymin=0.30, roi_ymax=0.88,
            gripper_overlap_reject=0.55, side_split_x=0.50, side_margin=0.03,
            left_roi_xmax=None, right_roi_xmin=None, visual_tracker="auto",
            tracker_max_jump=0.12, tracker_max_area_change=2.5, tracker_min_area_change=0.35,
            reinit_tracker_every_detection=True, max_missing=5, assoc_max_dist=0.20,
            velocity_alpha=0.60, box_alpha=0.80, prediction_decay=0.85,
        )
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.file = output_csv.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=(
            "frame", "time_s", "left_source", "right_source", "left_motion_source", "right_motion_source", "left_ee_age_s", "right_ee_age_s",
            "left_toy_motion", "right_toy_motion", "left_toy_area_change", "right_toy_area_change",
            "left_ee_speed", "right_ee_speed", "left_ee_xy_speed", "right_ee_xy_speed", "left_ee_z_speed", "right_ee_z_speed", "left_ee_z_moving", "right_ee_z_moving",
            "left_candidate_s", "right_candidate_s", "raw_trigger", "trigger_streak",
            "final_pause", "trigger_sides", "left_reason", "right_reason",
        ))
        self.writer.writeheader()

    def close(self):
        self.file.close()
        if self.external_worker is not None:
            self.external_worker.close()

    def reset_episode(self):
        """Clear temporal evidence after a stage-2 FALSE_ALARM recovery.

        The YOLO worker and CSV stay open, but stale pre-pause motion history
        must not immediately trigger the resumed rollout again.
        """
        self.episode_start_s = None
        self.streak = 0
        self.triggered = False
        self.tracks = {"left": self.SideTrack("left_toy"), "right": self.SideTrack("right_toy")}
        self.state = {
            side: {"toy": deque(maxlen=1000), "ee": deque(maxlen=1000), "candidate_s": 0.0, "last_t": None}
            for side in self.tracks
        }

    def _detect(self, image):
        h, w = image.shape[:2]
        if self.external_worker is not None:
            raw = [{**det, "box": tuple(det["box"])} for det in self.external_worker.predict(image)]
        else:
            result = self.model.predict(image, conf=self.conf, iou=self.iou, device=self.device, verbose=False)[0]
            raw = []
            if result.boxes is not None:
                for det in result.boxes:
                    cls = int(det.cls[0].item())
                    if 0 <= cls < len(self.CLASSES):
                        raw.append({"class": self.CLASSES[cls], "conf": float(det.conf[0].item()),
                                    "box": tuple(float(x) for x in det.xyxy[0].tolist())})
        grippers = [d["box"] for d in raw if d["class"] == "robot gripper"]
        candidates = {"left": [], "right": []}
        for det in raw:
            if det["class"] != "toy":
                continue
            if self.toy_rejection_reason(det["box"], w, h, grippers, self.args) is not None:
                continue
            side, _ = self.side_assignment(det["box"], w, h, self.args)
            if side:
                candidates[side.split("_")[0]].append(det)
        output = {}
        for side in ("left", "right"):
            self.update_side_track(self.tracks[side], candidates[side], image, w, h, self.args)
            track = self.tracks[side]
            if track.box is None:
                output[side] = None
            else:
                x1, y1, x2, y2 = track.box
                output[side] = {
                    "xy": self.box_center_norm(track.box, w, h),
                    "area": max(0.0, (x2 - x1) * (y2 - y1)) / float(w * h),
                    "source": track.source,
                }
        return output

    def update(self, image, time_s, poses, joint_states=None):
        """Update from synchronized image, EE pose, and optional joint-state proxy.

        ``poses`` and ``joint_states`` map a side to ``(timestamp_s, vector)``.
        Joint states are used only when explicitly enabled in configuration.
        """
        if self.episode_start_s is None:
            self.episode_start_s = time_s
        in_pregrasp_grace = (time_s - self.episode_start_s) < self.start_after_s
        toys, details, raw_sides = self._detect(image), {}, []
        for side in ("left", "right"):
            st, toy, pose = self.state[side], toys[side], poses.get(side)
            motion_source = "ee_pose"
            if pose is None and self.allow_joint_state_fallback and joint_states:
                pose = joint_states.get(side)
                motion_source = "joint_state_proxy"
            if pose is None:
                motion_source = ""
            source = "" if toy is None else toy["source"]
            age = None if pose is None else max(0.0, time_s - pose[0])
            detail = {"source": source, "age": age, "toy_motion": None, "toy_area_change": None,
                      "ee_speed": None, "ee_xy_speed": None, "ee_z_speed": None, "ee_z_moving": None,
                      "candidate_s": st["candidate_s"], "raw": False, "reason": ""}
            if in_pregrasp_grace:
                st["candidate_s"] = 0.0
                detail["reason"] = "pregrasp_grace"
            elif toy is None or source not in RELIABLE:
                st["candidate_s"] = 0.0; detail["reason"] = "vision_uncertain"
            elif pose is None or age > self.pose_max_age_s:
                st["candidate_s"] = 0.0; detail["reason"] = "ee_pose_missing_or_stale"
            else:
                st["toy"].append({"t": time_s, "xy": toy["xy"], "area": toy["area"]})
                st["ee"].append({"t": time_s, "xyz": pose[1]})
                old_toy, old_ee = _past(st["toy"], time_s - self.window_s), _past(st["ee"], time_s - self.window_s)
                if old_toy is None or old_ee is None:
                    st["candidate_s"] = 0.0; detail["reason"] = "warming_window"
                else:
                    toy_motion = _distance(toy["xy"], old_toy["xy"])
                    # ``area`` was added after the original tracker format.
                    # Keep this tolerant of a pre-existing in-memory history
                    # entry when a long-running process reloads the module.
                    old_area = old_toy.get("area")
                    toy_area_change = abs(toy["area"] - old_area) if old_area is not None else None
                    elapsed = time_s - old_ee["t"]
                    delta = tuple(a - b for a, b in zip(pose[1], old_ee["xyz"]))
                    ee_xy_speed = math.hypot(delta[0], delta[1]) / elapsed if elapsed > 0 else 0.0
                    ee_z_speed = abs(delta[2]) / elapsed if elapsed > 0 and len(delta) >= 3 else 0.0
                    # A pose sample is Cartesian XYZ: use world-plane XY for
                    # the trigger. Z is intentionally separate because a lift
                    # can be invisible in the image plane or camera-depth axis.
                    # Joint-state fallback remains a binary actuation proxy.
                    ee_speed = _distance(pose[1], old_ee["xyz"]) / elapsed if elapsed > 0 else 0.0
                    robot_moves = ee_xy_speed >= self.ee_min if motion_source == "ee_pose" else ee_speed >= self.ee_min
                    inconsistent = robot_moves and toy_motion <= self.toy_max
                    step_dt = 0.0 if st["last_t"] is None else max(0.0, time_s - st["last_t"])
                    st["candidate_s"] = st["candidate_s"] + step_dt if inconsistent else 0.0
                    detail.update(toy_motion=toy_motion, toy_area_change=toy_area_change, ee_speed=ee_speed,
                                  ee_xy_speed=ee_xy_speed if motion_source == "ee_pose" else None,
                                  ee_z_speed=ee_z_speed if motion_source == "ee_pose" else None,
                                  ee_z_moving=ee_z_speed >= self.ee_z_min if motion_source == "ee_pose" else None,
                                  candidate_s=st["candidate_s"],
                                  raw=inconsistent and st["candidate_s"] >= self.duration_s,
                                  reason="ee_xy_moves_toy_static" if inconsistent and motion_source == "ee_pose"
                                  else "joint_proxy_moves_toy_static" if inconsistent else "signals_consistent")
            detail["candidate_s"] = st["candidate_s"]
            st["last_t"] = time_s
            details[side] = detail
            if detail["raw"]:
                raw_sides.append(side)
        raw = bool(raw_sides)
        self.streak = self.streak + 1 if raw else 0
        final = raw and self.streak >= self.confirm_frames and not self.triggered
        self.triggered = self.triggered or final
        self.writer.writerow({
            "frame": self.frame, "time_s": f"{time_s:.6f}", "left_source": details["left"]["source"], "right_source": details["right"]["source"],
            "left_motion_source": "ee_pose" if poses.get("left") is not None else "joint_state_proxy" if self.allow_joint_state_fallback else "",
            "right_motion_source": "ee_pose" if poses.get("right") is not None else "joint_state_proxy" if self.allow_joint_state_fallback else "",
            "left_ee_age_s": details["left"]["age"], "right_ee_age_s": details["right"]["age"],
            "left_toy_motion": details["left"]["toy_motion"], "right_toy_motion": details["right"]["toy_motion"],
            "left_toy_area_change": details["left"]["toy_area_change"], "right_toy_area_change": details["right"]["toy_area_change"],
            "left_ee_speed": details["left"]["ee_speed"], "right_ee_speed": details["right"]["ee_speed"],
            "left_ee_xy_speed": details["left"]["ee_xy_speed"], "right_ee_xy_speed": details["right"]["ee_xy_speed"],
            "left_ee_z_speed": details["left"]["ee_z_speed"], "right_ee_z_speed": details["right"]["ee_z_speed"],
            "left_ee_z_moving": details["left"]["ee_z_moving"], "right_ee_z_moving": details["right"]["ee_z_moving"],
            "left_candidate_s": details["left"]["candidate_s"], "right_candidate_s": details["right"]["candidate_s"],
            "raw_trigger": raw, "trigger_streak": self.streak, "final_pause": final,
            "trigger_sides": "+".join(raw_sides), "left_reason": details["left"]["reason"], "right_reason": details["right"]["reason"],
        })
        self.file.flush(); self.frame += 1
        return {"time_s": time_s, "sides": "+".join(raw_sides)} if final else None
