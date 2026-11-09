"""Pure configuration and vote logic for the automatic package demo (no ROS/SDK imports)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import threading

import yaml

ROOT = Path(__file__).resolve().parents[2]


def root_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_demo(path):
    spec = yaml.safe_load(root_path(path).read_text())
    from kuavo_deploy.utils.recovery_entry import validate_entry_config
    spec.setdefault('recovery_right_arm_entry', {'enabled': False})
    validate_entry_config(spec['recovery_right_arm_entry'])
    for name in ["recovery_hits", "normal_hits", "max_recoveries", "rpc_timeout_ms"]:
        if type(spec.get(name)) is not int or spec[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    spec.setdefault('monitor_recovery_timeout', 15.0)
    spec.setdefault('left_grasp_command_threshold', 0.5)
    spec.setdefault('left_grasp_feedback_threshold', 0.25)
    spec.setdefault('grasp_stable_seconds', 0.3)
    for name in ["max_vote_gap_seconds", "max_vlm_result_age", "max_observation_age", "monitor_loss_timeout", "monitor_recovery_timeout",
                 "startup_timeout", "settle_timeout", "settle_window", "settle_max_joint_delta",
                 "settle_max_joint_velocity", "recovery_timeout", "verify_timeout", "minimum_recovery_seconds",
                 "demo_timeout", "left_release_threshold", "release_stable_seconds",
                 "left_grasp_command_threshold", "left_grasp_feedback_threshold", "grasp_stable_seconds"]:
        if not isinstance(spec.get(name), (int, float)) or not math.isfinite(spec[name]) or spec[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if spec['left_release_threshold'] >= 1 or spec['settle_window'] >= spec['settle_timeout']:
        raise ValueError("Invalid release threshold or settle window")
    if any(not spec['left_release_threshold'] < spec[k] <= 1 for k in
           ['left_grasp_command_threshold', 'left_grasp_feedback_threshold']):
        raise ValueError('Grasp thresholds must exceed release threshold and be <=1')
    profiles = yaml.safe_load(root_path(spec['server_config']).read_text())
    expected = {}
    for skill in ['normal', 'recovery']:
        block = profiles['adapters'][spec[skill + '_profile']]
        if block.get('adapter') != 'lingbot_vla':
            raise ValueError('Only the existing LingBot VLA adapter is supported')
        args = {**profiles.get('common', {}), **block.get('args', {})}
        if args.get('which_arm') != 'both' or args.get('execution_horizon') != 25:
            raise ValueError('Package checkpoints require the configured both-arm 25-step server horizon')
        for name in ['checkpoint', 'training_config_path', 'robot_norm_path', 'qwen25_path']:
            p = root_path(args[name])
            if not p.exists():
                raise FileNotFoundError(f'{skill}: {name} not found: {p}')
            args[name] = str(p)
        checkpoint = Path(args['checkpoint'])
        if not (checkpoint / 'config.json').is_file() or not list(checkpoint.glob('*.safetensors')):
            raise ValueError(f'{skill} is not an exported HF checkpoint')
        json.loads((checkpoint / 'config.json').read_text())
        yaml.safe_load(Path(args['training_config_path']).read_text())
        json.loads(Path(args['robot_norm_path']).read_text())
        expected[skill] = {k: args[k] for k in ['checkpoint', 'training_config_path', 'robot_norm_path', 'port']}
    if expected['normal']['port'] == expected['recovery']['port']:
        raise ValueError('Normal and Recovery servers need different ports')
    if expected['normal']['robot_norm_path'] == expected['recovery']['robot_norm_path']:
        raise ValueError('Normal and Recovery must use their separate normalization statistics')
    for key in ['vlm_python', 'vlm_prompt', 'vlm_model']:
        if not root_path(spec[key]).exists():
            raise FileNotFoundError(spec[key])
    spec['expected_servers'] = expected
    return spec


def validate_deployment(path):
    from kuavo_deploy.config import load_kuavo_config
    resolved = root_path(path)
    cfg = load_kuavo_config(str(resolved))
    required = {
        'inference.policy_type': 'client',
        'inference.async_inference': False,
        'env.which_arm': 'both',
        'env.eef_type': 'qiangnao',
        'env.qiangnao_dof_needed': 1,
        'env.frame_alignment': False,
        'inference.safety_enabled': True,
    }
    mismatches = []
    for field, expected in required.items():
        section, key = field.split('.')
        actual = getattr(getattr(cfg, section), key)
        if type(actual) is not type(expected) or actual != expected:
            mismatches.append(f'  {field}: current={actual!r}, required={expected!r}')
    if mismatches:
        raise ValueError(f'Demo deployment config incompatible: {resolved}\n' + '\n'.join(mismatches))
    if not cfg.inference.go_bag_path or not Path(cfg.inference.go_bag_path).is_file():
        raise FileNotFoundError('The existing ArmMove startup requires a valid inference.go_bag_path')
    return cfg


class FreshVotes:
    """Consume every judgment in order, not every UI poll or every repeated status file."""
    def __init__(self, max_age=1.5, max_gap=2.5):
        self.max_age, self.max_gap = max_age, max_gap
        self.lock = threading.RLock()
        self.last = None
        self.last_seen_mono = None
        self.streak = 0
        self.last_stamp_ns = -1
        self.epoch = None
        self.not_before_ros = float('-inf')
        self.last_event = 'waiting'

    def reset(self, not_before_ros=None):
        with self.lock:
            self.streak, self.last = 0, None
            if not_before_ros is not None:
                self.not_before_ros = not_before_ros

    def feed(self, row, ros_now, mono_now):
        if row.get('event') != 'judgment':
            return
        with self.lock:
            self.last_event = row.get('status', 'invalid')
            frame = row.get('frame') or {}
            parsed = row.get('parsed')
            if (not row.get('accepted_current') or row.get('status') != 'accepted'
                    or not isinstance(parsed, dict) or set(parsed) != {'skill'}
                    or parsed['skill'] not in ['normal', 'recovery']):
                self.streak, self.last = 0, None
                return
            stamp, ns, received = frame.get('stamp_s'), frame.get('stamp_ns'), frame.get('received_mono')
            if (not all(isinstance(x, (int, float)) and math.isfinite(x) for x in [stamp, ns, received])
                    or stamp <= 0 or not -0.05 <= ros_now - stamp <= self.max_age
                    or not 0 <= mono_now - received <= self.max_age):
                self.streak, self.last = 0, None
                self.last_event = 'stale_at_consumer'
                return
            epoch = frame.get('epoch')
            if type(epoch) is not int or self.epoch is not None and epoch < self.epoch:
                self.streak, self.last = 0, None
                return
            if self.epoch != epoch:
                self.streak, self.last, self.last_stamp_ns = 0, None, -1
                self.epoch = epoch
            if ns <= self.last_stamp_ns:
                return  # repeated logs/polls are not new votes
            self.last_stamp_ns = ns
            if stamp < self.not_before_ros:
                return
            same = (self.last is not None and self.last['skill'] == parsed['skill']
                    and stamp - self.last['stamp_s'] <= self.max_gap)
            self.streak = self.streak + 1 if same else 1
            self.last = dict(skill=parsed['skill'], stamp_s=stamp, stamp_ns=ns,
                             received_mono=received, completed_at_utc=row.get('completed_at_utc'))
            self.last_seen_mono = mono_now

    def snapshot(self, ros_now, mono_now):
        with self.lock:
            fresh = (self.last is not None
                     and -0.05 <= ros_now - self.last['stamp_s'] <= self.max_age
                     and 0 <= mono_now - self.last['received_mono'] <= self.max_age)
            return dict(last=dict(self.last) if self.last else None, streak=self.streak,
                        fresh=fresh, last_seen_mono=self.last_seen_mono, last_event=self.last_event)


def expected_metadata(response, expected):
    if not isinstance(response, dict) or response.get('error') or response.get('adapter') != 'lingbot_vla':
        raise RuntimeError(f'Unexpected model server: {response}')
    for name in ['checkpoint', 'training_config_path', 'robot_norm_path']:
        if root_path(response.get(name, '')) != root_path(expected[name]):
            raise RuntimeError(f'Server {name} does not match the selected skill')
    if response.get('which_arm') != 'both' or response.get('execution_horizon') != 25:
        raise RuntimeError('Server arm mode/horizon mismatch')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_monitor_vote(snapshot, normal_hits, recovery_hits):
    last = snapshot.get('last')
    return bool(snapshot.get('fresh') and last and last.get('skill') in ['normal', 'recovery']
                and snapshot['streak'] >= (normal_hits if last['skill'] == 'normal' else recovery_hits))


class PostStopDecision:
    """Require fresh stable votes after the caller has reset the capture boundary."""
    def __init__(self, deadline, normal_hits, recovery_hits):
        self.deadline, self.normal_hits, self.recovery_hits = deadline, normal_hits, recovery_hits

    def evaluate(self, snapshot, now):
        if now >= self.deadline:
            raise TimeoutError('No stable post-stop normal/recovery classification before timeout')
        if stable_monitor_vote(snapshot, self.normal_hits, self.recovery_hits):
            return snapshot['last']['skill']
        return None


class RecoveryGripperCycle:
    """Per-attempt closure then release evidence; joint position does not prove object contact."""
    def __init__(self, spec):
        self.spec = spec
        self.stage = 'waiting_for_grasp'
        self.grasp_since = self.release_since = self.previous_time = None

    def update(self, command, feedback, now):
        if self.previous_time is not None and now - self.previous_time > self.spec['max_observation_age']:
            self.grasp_since = self.release_since = None
        self.previous_time = now
        if command is None or not all(math.isfinite(x) for x in [command, feedback]):
            self.grasp_since = self.release_since = None
            if self.stage == 'released':
                self.stage = 'grasp_observed'
            return self.stage
        if self.stage == 'waiting_for_grasp':
            closed = (command >= self.spec['left_grasp_command_threshold']
                      and feedback >= self.spec['left_grasp_feedback_threshold'])
            self.grasp_since = (now if self.grasp_since is None else self.grasp_since) if closed else None
            if self.grasp_since is not None and now - self.grasp_since >= self.spec['grasp_stable_seconds']:
                self.stage = 'grasp_observed'
        else:
            opened = command <= self.spec['left_release_threshold'] and feedback <= self.spec['left_release_threshold']
            self.release_since = (now if self.release_since is None else self.release_since) if opened else None
            self.stage = ('released' if self.release_since is not None
                          and now - self.release_since >= self.spec['release_stable_seconds'] else 'grasp_observed')
        return self.stage
