#!/usr/bin/env python3
"""Read-only ROS RGB observer. Never imports deployment/control entrypoints.

Run from the repository root with ``python -m vlm_test.passive_vlm_monitor``.
stdout: current, valid {"skill": ...} results only. Diagnostics go to stderr.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import sys
import threading
import time

from recovery_harness.observation import ros_stamp_seconds
from recovery_harness.ros_bridge import decode_compressed_rgb, KuavoControlSignalBridge

ROOT = Path(__file__).resolve().parents[1]


def utc(seconds=None):
    return datetime.fromtimestamp(time.time() if seconds is None else seconds, timezone.utc).isoformat()


def strict_skill(raw):
    """No fences, explanations, extra keys, unknown, or fallback-to-normal."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(raw.strip(), object_pairs_hook=unique)
    if not isinstance(value, dict) or set(value) != {"skill"} or value["skill"] not in ["normal", "recovery"]:
        raise ValueError('Expected exactly {"skill":"normal"} or {"skill":"recovery"}')
    return value


@dataclass(frozen=True)
class RGBPacket:
    generation: int
    stamp_s: float | None
    stamp_ns: int
    epoch: int
    received_mono: float
    received_wall: float
    received_ros: float
    frame_id: str
    seq: int
    format: str
    publisher: str | None
    data: bytes
    raw_header_secs: int | None = None
    raw_header_nsecs: int | None = None
    stamp_error: str | None = None

    def metadata(self):
        return {k: v for k, v in vars(self).items() if k != "data"}


def image_header_stamp(stamp):
    """Read ROS1 wire fields directly; never substitute receipt time for capture time."""
    secs, nsecs = getattr(stamp, "secs", None), getattr(stamp, "nsecs", None)
    try:
        if secs is not None and nsecs is not None:
            secs, nsecs = int(secs), int(nsecs)
            if secs < 0 or not 0 <= nsecs < 1_000_000_000:
                raise ValueError("invalid ROS1 secs/nsecs")
            nanoseconds = secs * 1_000_000_000 + nsecs
        else:
            value, _ = ros_stamp_seconds(stamp)
            if value is None:
                return None, 0, secs, nsecs, "no positive header stamp or readable ROS1 fields"
            nanoseconds = int(stamp.to_nsec())
        if nanoseconds <= 0:
            return None, 0, secs, nsecs, "zero header stamp"
        return nanoseconds / 1_000_000_000, nanoseconds, secs, nsecs, None
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        return None, 0, secs, nsecs, f"{type(exc).__name__}: {exc}"


class LatestRGB:
    """One atomic latest packet; decoding is deferred until the 1Hz sample."""
    def __init__(self, use_sim_time=False):
        self.lock = threading.Lock()
        self.packet = None
        self.generation = 0
        self.epoch = 0
        self.last_ros = None
        self.last_advance_mono = None
        self.last_clock_mono = None
        self.last_stamp_ns = -1
        self.use_sim_time = use_sim_time
        self.rejected = 0

    def _clock(self, ros_now, mono_now):
        if self.last_clock_mono is not None and mono_now < self.last_clock_mono:
            return
        if self.last_ros is not None and ros_now < self.last_ros - 0.05:
            self.epoch += 1
            self.packet = None
            self.last_stamp_ns = -1
        if self.last_ros is None or ros_now != self.last_ros:
            self.last_advance_mono = mono_now
        self.last_ros = ros_now
        self.last_clock_mono = mono_now

    def receive(self, msg, ros_now, mono_now, wall_now):
        stamp, stamp_ns, secs, nsecs, stamp_error = image_header_stamp(msg.header.stamp)
        with self.lock:
            self._clock(ros_now, mono_now)
            if stamp_ns > 0 and stamp_ns <= self.last_stamp_ns:
                self.rejected += 1
                return
            if stamp_ns > 0:
                self.last_stamp_ns = stamp_ns
            self.generation += 1
            self.packet = RGBPacket(
                self.generation, stamp, stamp_ns, self.epoch, mono_now, wall_now, ros_now,
                str(msg.header.frame_id), int(msg.header.seq), str(msg.format),
                (getattr(msg, "_connection_header", None) or {}).get("callerid"), bytes(msg.data),
                secs, nsecs, stamp_error)

    def snapshot(self, ros_now, mono_now):
        with self.lock:
            self._clock(ros_now, mono_now)
            return self.packet, dict(epoch=self.epoch, use_sim_time=self.use_sim_time,
                                     last_advance_mono=self.last_advance_mono,
                                     duplicate_or_out_of_order=self.rejected)


def freshness(packet, ros_now, mono_now, clock, max_age, future_tolerance=0.05):
    if not math.isfinite(ros_now) or ros_now <= 0:
        return "ros_time_unavailable"
    if clock["use_sim_time"] and (clock["last_advance_mono"] is None or mono_now - clock["last_advance_mono"] > max_age):
        return "ros_clock_stalled"
    if packet is None:
        return "no_frame"
    if packet.epoch != clock["epoch"]:
        return "ros_clock_epoch_changed"
    if packet.stamp_s is None or not math.isfinite(packet.stamp_s) or packet.stamp_s <= 0:
        return "missing_header_stamp"
    if ros_now >= 1_000_000_000 and packet.stamp_s < 1_000_000_000:
        return "header_clock_domain_mismatch"
    if ros_now - packet.stamp_s < -future_tolerance:
        return "future_header_stamp"
    if mono_now < packet.received_mono:
        return "invalid_monotonic_time"
    if ros_now - packet.stamp_s > max_age or mono_now - packet.received_mono > max_age:
        return "stale_frame"
    return None


class Journal:
    def __init__(self, directory):
        self.directory = directory
        self.stream = (directory / "events.jsonl").open("x", encoding="utf-8")

    def emit(self, event, **fields):
        self.stream.write(json.dumps(dict(event=event, logged_at_utc=utc(), logged_wall_s=time.time(),
                                          logged_mono_s=time.monotonic(), **fields),
                                     ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()

    def status(self, value):
        temp = self.directory / "latest_status.tmp"
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temp.replace(self.directory / "latest_status.json")


class ExecutionEvidence:
    """Follow existing eval log read-only. Observations, not physical acknowledgements."""
    def __init__(self, path, journal, ttl=3.0):
        self.path, self.journal, self.ttl = path, journal, ttl
        self.identity = None
        self.offset = 0
        self.prefix = b""
        self.partial = b""
        self.last = None
        self.started = False

    def poll(self):
        try:
            with self.path.open("rb") as f:
                stat = f.fileno()
                import os
                info = os.fstat(stat)
                identity = (info.st_dev, info.st_ino)
                prefix = f.read(128)
                reset = self.identity != identity or info.st_size < self.offset or not prefix.startswith(self.prefix)
                if reset:
                    self.offset = 0 if self.started else info.st_size
                    self.partial = b""
                    self.last = None
                    self.journal.emit("vla_log_attached" if not self.started else "vla_log_reset",
                                      path=str(self.path), skipped_preexisting_bytes=self.offset)
                self.identity, self.prefix, self.started = identity, prefix, True
                f.seek(self.offset)
                data = f.read(65536)
                self.offset = f.tell()
        except FileNotFoundError:
            self.last = None
            return
        except OSError as exc:
            self.journal.emit("vla_log_read_error", error=str(exc), path=str(self.path))
            self.last = None
            return
        parts = (self.partial + data).split(b"\n")
        self.partial = parts.pop()[-16384:]
        for part in parts:
            line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", part.decode("utf-8", errors="replace")).strip()
            if not line:
                continue
            self.journal.emit("vla_log_line", path=str(self.path), line=line)
            status = None
            step = re.search(r"Step (\d+) time:", line)
            if step or "action infer time:" in line:
                status = "running_observed"
            elif "Paused. Waiting for resume" in line:
                status = "paused_observed"
            elif "Stop flag detected" in line:
                status = "stop_observed_in_vla_log"
            elif re.search(r"Episode \d+: (Success|Failed)", line):
                status = "episode_finished_observed"
            elif "Model loaded from" in line:
                status = "model_loaded_observed"
            if status:
                self.last = dict(status=status, received_mono=time.monotonic(), received_at_utc=utc(),
                                 step=int(step.group(1)) if step else None, evidence=line)

    def snapshot(self):
        age = time.monotonic() - self.last["received_mono"] if self.last else None
        return dict(status=self.last["status"] if age is not None and age <= self.ttl else "unknown",
                    evidence_age_seconds=age, last_evidence=self.last, source=str(self.path),
                    meaning="Observed deployment log activity, not confirmed motion/skill acknowledgement")


class ObservedSignals(KuavoControlSignalBridge):
    def __init__(self):
        self.evidence_lock = threading.Lock()
        self.observed = {}
        super().__init__({})

    def _pause_cb(self, msg):
        with self.evidence_lock:
            super()._pause_cb(msg)
            self.observed["pause"] = dict(value=bool(msg.data), received_wall_s=time.time())

    def _stop_cb(self, msg):
        with self.evidence_lock:
            super()._stop_cb(msg)
            self.observed["stop"] = dict(value=bool(msg.data), received_wall_s=time.time())

    def snapshot(self):
        with self.evidence_lock:
            return dict(observed_requests=dict(self.observed),
                        meaning="Operator pause/stop requests only; absent messages mean unknown, not running")


def camera_source(config_path, camera_key):
    # This loader is pure configuration parsing. Do not instantiate gym/ArmMove/SDK.
    import yaml
    from kuavo_deploy.config import load_kuavo_config
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if "adapters" in raw:
        raise ValueError("--deploy-config needs the eval.py deployment YAML (env/obs_key_map), not the server checkpoint launcher YAML")
    cfg = load_kuavo_config(str(config_path))
    info = cfg.env.obs_key_map.get(camera_key)
    if not info or info.get("msg_type") != "CompressedImage" or "depth" in camera_key:
        raise ValueError(f"{camera_key} is not a configured RGB CompressedImage observation")
    return cfg, info


def reference_calibration(path):
    """Audit a historical calibration; coordinates NEVER enter the current-frame prompt."""
    raw = path.read_bytes()
    data = json.loads(raw)
    polygon, indices = data.get("platform_polygon"), data.get("body_side_edge_indices")
    if (not isinstance(polygon, list) or len(polygon) < 3 or not isinstance(indices, list)
            or len(indices) != 2 or indices[0] == indices[1]
            or any(type(i) is not int or not 0 <= i < len(polygon) for i in indices)):
        raise ValueError("ROI reference requires a platform polygon and two valid body-side edge indices")
    if any(not isinstance(p, list) or len(p) != 2 or
           any(type(v) not in (int, float) or not math.isfinite(v) for v in p) for p in polygon):
        raise ValueError("ROI reference polygon must contain finite image-pixel coordinates")
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), calibration=data,
                used_as_current_pixel_roi=False, sent_as_model_image=False,
                meaning="Historical manual reference only; current platform edge is inferred visually from each RGB frame")


def select_prompt(explicit_path=None, camera_only=False):
    if explicit_path is not None:
        return explicit_path
    # Task semantics are the same with/without VLA; older versions remain explicit opt-ins.
    return ROOT / "vlm_test/prompts/package_front_edge_tolerant_v4.json"


class ExpiredInput(ValueError):
    pass


def classify(adapter, packet, prompt, directory, width, check_current, ros_clock):
    row = dict(frame=packet.metadata(), input_image=None, raw_output=None, parsed=None,
               sampled_wall_s=time.time(), worker_started_utc=utc())
    start = time.monotonic()
    try:
        check_current(packet)
        image = decode_compressed_rgb(packet.data)
        row["source_rgb_size"] = list(image.size)
        if width and image.width != width:
            from PIL import Image
            height = max(2, int(round(image.height * width / image.width / 2)) * 2)
            image = image.resize((width, height), Image.Resampling.BICUBIC)
        path = directory / "images" / f"{packet.epoch}_{packet.generation:08d}_{packet.stamp_ns}.png"
        image.save(path, compress_level=0)
        row.update(input_image=str(path), input_rgb_size=list(image.size))
        check_current(packet)
        with redirect_stdout(sys.stderr):
            raw, timings = adapter.generate_raw(image, prompt, max_new_tokens=24,
                                                before_generate=lambda: check_current(packet))
        row.update(raw_output=raw, timing=timings)
        try:
            row["parsed"] = strict_skill(raw)
            row["status"] = "parsed"
        except (ValueError, TypeError) as exc:
            row.update(status="invalid_output", error=str(exc))
    except ExpiredInput as exc:
        row.update(status="stale_before_inference", error=str(exc))
    except Exception as exc:
        row.update(status="inference_error", error=f"{type(exc).__name__}: {exc}")
    row.update(completed_wall_s=time.time(), completed_mono_s=time.monotonic(), completed_at_utc=utc(),
               completed_ros_s=ros_clock(),
               worker_total_seconds=time.monotonic() - start)
    return row


def load_adapter(model_path, precision, service_url=None):
    if service_url:
        from vlm_test.qwen_service import RemoteQwenAdapter
        return RemoteQwenAdapter(service_url, model_path, precision)
    with redirect_stdout(sys.stderr):
        import torch
        from recovery_harness.vlm import Qwen35SingleFrameAdapter
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("A BF16-capable CUDA GPU is required; no CPU/offload fallback")
        return Qwen35SingleFrameAdapter(str(model_path), precision=precision)


def adapter_ready_metadata(adapter):
    if hasattr(adapter, 'url'):
        return dict(backend='persistent_qwen_service', service_url=adapter.url,
                    model_loaded_in_monitor=False)
    quant = getattr(adapter.model.config, 'quantization_config', None)
    return dict(backend='local', memory_footprint_bytes=adapter.model.get_memory_footprint(),
                quantization_config=quant.to_dict() if hasattr(quant, 'to_dict') else quant,
                torch_version=adapter._torch.__version__,
                transformers_version=getattr(sys.modules.get('transformers'), '__version__', None),
                gpu=adapter._torch.cuda.get_device_name(0))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-config", required=True, type=Path)
    parser.add_argument("--camera-key", default="head_cam_h")
    parser.add_argument("--prompt", type=Path, help="Default: v4 tolerates normal-operable edge overhang; older prompts remain available explicitly")
    parser.add_argument("--camera-only", action="store_true", help="Manual placement test: RGB only, no VLA log or pause/stop subscriptions")
    parser.add_argument("--roi-reference", type=Path, default=Path("/media/data/jiayi/dataset/roi_config.json"),
                        help="Camera-only historical calibration for audit, never applied as a current pixel ROI")
    parser.add_argument("--model-path", type=Path, default=Path("/media/data/jiayi/hf_model/Qwen3.5-9B"))
    parser.add_argument("--precision", choices=["nf4", "bf16"], default="nf4")
    parser.add_argument("--qwen-service-url", help="Reuse a separately loaded local Qwen service; never load a second model")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/passive_vlm_monitor")
    parser.add_argument("--vla-log", type=Path, default=ROOT / "log/kuavo_deploy/kuavo_deploy.log")
    parser.add_argument("--max-frame-age", type=float, default=0.5)
    parser.add_argument("--max-result-age", type=float, default=1.5)
    parser.add_argument("--resize-width", type=int, default=640, help="0 preserves source RGB dimensions")
    parser.add_argument("--probe-only", action="store_true", help="Read camera headers for 5s; no Torch/model load")
    parser.add_argument("--max-samples", type=int, help="Exit after this many completed inference attempts")
    args = parser.parse_args(argv)
    if (not all(math.isfinite(x) and x > 0 for x in [args.max_frame_age, args.max_result_age])
            or args.max_result_age < args.max_frame_age or args.resize_width < 0
            or args.max_samples is not None and args.max_samples < 1):
        parser.error("Require positive finite ages, result age >= frame age, width >=0, samples >=1")
    cfg, source = camera_source(args.deploy_config.resolve(), args.camera_key)
    args.prompt = select_prompt(args.prompt, args.camera_only)
    roi_reference = reference_calibration(args.roi_reference) if args.camera_only else None
    prompt_data = json.loads(args.prompt.read_text(encoding="utf-8"))
    prompt = prompt_data["skill_prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt config requires a nonempty skill_prompt")
    directory = args.output_root.resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    (directory / "images").mkdir(parents=True, exist_ok=False)
    journal = Journal(directory)
    from kuavo_deploy.utils.logging_utils import setup_logger
    logger = setup_logger("passive_vlm", logging.INFO, log_file=str(directory / "monitor.log"))
    import rospy
    from sensor_msgs.msg import CompressedImage
    from kuavo_deploy.utils.ros_manager import ROSManager
    rospy.init_node("passive_vlm_monitor", anonymous=True, disable_rosout=True)
    slot = LatestRGB(bool(rospy.get_param("/use_sim_time", False)))
    manager = ROSManager()
    manager.register_subscriber(source["topic"], CompressedImage,
                                lambda msg: slot.receive(msg, rospy.Time.now().to_sec(), time.monotonic(), time.time()))
    signals = None if args.camera_only else ObservedSignals()
    evidence = None if args.camera_only else ExecutionEvidence(args.vla_log.resolve(), journal)
    session = dict(deploy_config=str(args.deploy_config.resolve()), resolved_camera_key=args.camera_key,
                   resolved_camera=source, timestamp_source="CompressedImage.header.stamp (required, zero rejected)",
                   prompt=prompt_data, prompt_sha256=hashlib.sha256(args.prompt.read_bytes()).hexdigest(),
                   inference_config=None if args.camera_only else vars(cfg.inference), use_sim_time=slot.use_sim_time,
                   roi_reference=roi_reference, camera_only=args.camera_only,
                   geometry_mode="per_frame_vlm_visual_relocalization_no_fixed_roi",
                   reference_image_input_count=0,
                   checkpoint_note="For policy_type=client the checkpoint is selected by the external LingBot server; this monitor does not resolve or validate it.",
                   model_path=str(args.model_path.resolve()), precision=args.precision, qwen_service_url=args.qwen_service_url,
                   period_seconds=1.0, max_frame_age=args.max_frame_age, max_result_age=args.max_result_age,
                   resize_width=args.resize_width, max_new_tokens=24, enable_thinking=False, do_sample=False,
                   vla_log=None if args.camera_only else str(args.vla_log.resolve()),
                   mode="camera_only_manual_placement" if args.camera_only else "passive_read_only",
                   probe_only=args.probe_only)
    (directory / "session.json").write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    journal.emit("session_started", directory=str(directory), **session)
    logger.info("Read-only monitor: topic=%s; logs=%s", source["topic"], directory)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="passive-vlm")
    future = None if args.probe_only else executor.submit(load_adapter, args.model_path, args.precision, args.qwen_service_url)
    loading, adapter = not args.probe_only, None
    accepted = None
    health = "probe" if args.probe_only else "loading_model"
    last_generation, completed, next_tick = -1, 0, time.monotonic()
    started, seen_camera, valid_probe_seen = time.monotonic(), False, False
    last_status_key, last_status_mono = None, float("-inf")

    def clock_snapshot():
        ros_now, mono_now = rospy.Time.now().to_sec(), time.monotonic()
        packet, clock = slot.snapshot(ros_now, mono_now)
        return packet, clock, ros_now, mono_now

    def check_current(packet):
        _, clock, ros_now, mono_now = clock_snapshot()
        reason = freshness(packet, ros_now, mono_now, clock, args.max_frame_age)
        if reason:
            raise ExpiredInput(reason)

    def vla_snapshot():
        if args.camera_only:
            return dict(log=dict(status="not_monitored", source=None),
                        control_requests=dict(meaning="Disabled: camera-only manual-placement test"))
        return dict(log=evidence.snapshot(), control_requests=signals.snapshot())

    def finish(row, stopping=False):
        nonlocal accepted, health, completed
        latest, clock, ros_now, mono_now = clock_snapshot()
        packet = RGBPacket(**row.pop("_packet"))
        reason = freshness(packet, ros_now, mono_now, clock, args.max_result_age)
        reason = reason or freshness(latest, ros_now, mono_now, clock, args.max_frame_age)
        row.update(processed_ros_s=ros_now,
                   header_to_completion_ros_seconds=row["completed_ros_s"] - packet.stamp_s if packet.stamp_s else None,
                   completion_to_log_seconds=mono_now - row["completed_mono_s"],
                   receipt_to_completion_seconds=row["completed_mono_s"] - packet.received_mono,
                   vla_at_completion=vla_snapshot(), accepted_current=False)
        if stopping:
            reason = "monitor_stopped"
        if row["status"] == "parsed" and not reason:
            row.update(accepted_current=True, status="accepted")
            accepted, health = packet, row["parsed"]["skill"]
        else:
            accepted = None
            if row["status"] == "parsed":
                row.update(status="stale_result", rejection_reason=reason)
            health = row["status"]
            logger.warning("No current VLM label: %s (%s)", health, row.get("error") or reason)
        journal.emit("judgment", **row)
        if row["accepted_current"]:
            latest, clock, ros_now, mono_now = clock_snapshot()
            display_error = (freshness(packet, ros_now, mono_now, clock, args.max_result_age)
                             or freshness(latest, ros_now, mono_now, clock, args.max_frame_age))
            if display_error:
                accepted, health = None, "expired_before_display"
                journal.emit("presentation_dropped", reason=display_error, frame=packet.metadata())
            else:
                print(json.dumps(row["parsed"], separators=(",", ":")), flush=True)
                if health == "recovery":
                    logger.warning("RECOVERY observation emitted; this monitor sends no robot commands. Skill routing is handled by the external controller. frame=%s", packet.stamp_ns)
        completed += 1

    pending_packet, pending_vla = None, None
    try:
        while not rospy.is_shutdown():
            packet, clock, ros_now, mono_now = clock_snapshot()
            if future is not None and future.done():
                if loading:
                    adapter = future.result()
                    loading, health, future = False, "waiting_for_fresh_frame", None
                    journal.emit("model_ready", **adapter_ready_metadata(adapter))
                    logger.info("VLM ready. Taking new frames only; no startup backlog.")
                else:
                    row = future.result()
                    row.update(_packet=vars(pending_packet), vla_at_sample=pending_vla)
                    finish(row)
                    future, pending_packet, pending_vla = None, None, None
            if args.max_samples is not None and completed >= args.max_samples:
                break
            packet, clock, ros_now, mono_now = clock_snapshot()
            if accepted is not None and freshness(accepted, ros_now, mono_now, clock, args.max_result_age):
                journal.emit("current_label_cleared", reason="expired_result", frame=accepted.metadata())
                logger.warning("CURRENT=UNAVAILABLE: previous result expired")
                accepted, health = None, "expired_result"
            reason = freshness(packet, ros_now, mono_now, clock, args.max_frame_age)
            if reason and accepted is not None:
                journal.emit("current_label_cleared", reason=reason, frame=accepted.metadata())
                logger.warning("CURRENT=UNAVAILABLE: %s", reason)
                accepted, health = None, reason
            state = dict(updated_at_utc=utc(), current_skill=health if accepted is not None else None,
                         health=health if not reason else reason, busy=future is not None,
                         frame=accepted.metadata() if accepted else None,
                         camera_latest=packet.metadata() if packet else None,
                         camera_rejection=reason, vla=vla_snapshot(), clock=clock)
            state["current_valid_until_ros_s"] = accepted.stamp_s + args.max_result_age if accepted else None
            # A reader must also expire this lease if the monitor dies/stalls without clearing the file.
            state["lease_expires_wall_s"] = time.time() + 0.25 if accepted else None
            status_key = (state["current_skill"], state["health"], state["busy"], accepted.generation if accepted else None)
            # Invalidate current labels on the 50ms watchdog, not only the 1Hz sampling tick.
            if status_key != last_status_key or mono_now - last_status_mono >= (0.1 if accepted else 1.0):
                journal.status(state)
                last_status_key, last_status_mono = status_key, mono_now
            if mono_now >= next_tick:
                next_tick = mono_now + 1.0  # no catch-up bursts, no queued inference requests
                if evidence is not None:
                    evidence.poll()
                state["vla"] = vla_snapshot()
                reason = freshness(packet, ros_now, mono_now, clock, args.max_frame_age)
                if not seen_camera and packet is not None:
                    journal.emit("camera_source_confirmed", topic=source["topic"], frame=packet.metadata())
                    logger.info("Camera header confirmed: %s", json.dumps(packet.metadata(), ensure_ascii=False))
                    seen_camera = True
                if args.probe_only:
                    journal.emit("camera_probe", frame=packet.metadata() if packet else None, rejection=reason, clock=clock)
                    valid_probe_seen = valid_probe_seen or reason is None
                elif future is not None:
                    journal.emit("sample_skipped", reason="model_loading" if loading else "inference_busy",
                                 frame=packet.metadata() if packet else None)
                elif reason:
                    journal.emit("sample_skipped", reason=reason, frame=packet.metadata() if packet else None)
                elif packet.generation == last_generation:
                    journal.emit("sample_skipped", reason="no_new_frame", frame=packet.metadata())
                else:
                    last_generation = packet.generation
                    pending_packet, pending_vla = packet, vla_snapshot()
                    journal.emit("sample_submitted", frame=packet.metadata(), vla_at_sample=pending_vla)
                    future = executor.submit(classify, adapter, packet, prompt, directory, args.resize_width,
                                             check_current, lambda: rospy.Time.now().to_sec())
                state["busy"] = future is not None
                journal.emit("heartbeat", **state)
                logger.info("CURRENT=%s health=%s busy=%s VLA=%s",
                            state["current_skill"] or "UNAVAILABLE", state["health"], state["busy"],
                            state["vla"]["log"]["status"])
            if args.probe_only and mono_now - started >= 5:
                if not valid_probe_seen:
                    raise RuntimeError("No fresh, positive-header-stamped camera frame during 5s probe")
                break
            time.sleep(0.05)
    finally:
        # Drain only our own in-flight inference for the audit log. No VLA signal is sent.
        if future is not None:
            try:
                result = future.result()
                if not loading:
                    result.update(_packet=vars(pending_packet), vla_at_sample=pending_vla)
                    finish(result, stopping=True)
            except Exception as exc:
                journal.emit("worker_error", error=f"{type(exc).__name__}: {exc}")
        executor.shutdown(wait=True)
        if signals is not None:
            signals.close()
        manager.close()
        journal.status(dict(updated_at_utc=utc(), current_skill=None, health="monitor_stopped"))
        journal.emit("session_finished", judgments=completed)
        journal.stream.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Passive monitor failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
