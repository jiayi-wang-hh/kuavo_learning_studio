#!/usr/bin/env python3
"""Automatic Normal -> Recovery -> Normal demo. Preflight never imports ROS/SDK.

Robot execution is only entered with --execute. No model/robot is started on import.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kuavo_deploy.utils.skill_demo import FreshVotes, expected_metadata, load_demo, root_path, validate_deployment, stable_monitor_vote, RecoveryGripperCycle, PostStopDecision


class Audit:
    def __init__(self, directory):
        self.directory = directory
        self.lock = threading.Lock()
        self.file = (directory / 'decisions.jsonl').open('x')

    def emit(self, event, **fields):
        with self.lock:
            row = dict(event=event, wall_s=time.time(), mono_s=time.monotonic(), **fields)
            self.file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            self.file.flush()


class OwnedProcesses:
    """Only processes started by this demo are terminated; never kill an existing server."""
    def __init__(self, audit):
        self.items, self.audit = [], audit

    def start(self, name, command, env):
        log = (self.audit.directory / (name + '.log')).open('x')
        try:
            proc = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            log.close()
            raise
        self.items.append((name, proc, log))
        self.audit.emit('process_started', name=name, pid=proc.pid)  # token is not logged

    def check(self):
        for name, proc, _ in self.items:
            if proc.poll() is not None:
                raise RuntimeError(f'{name} exited ({proc.returncode}); see its demo log')

    def close(self):
        for _, proc, _ in self.items:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for name, proc, log in self.items:
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=3)
            log.close()
            self.audit.emit('process_finished', name=name, returncode=proc.returncode)


class MonitorReader(threading.Thread):
    def __init__(self, root, votes, ros_clock, stop, audit):
        super().__init__(name='skill-vlm-reader', daemon=True)
        self.root, self.votes, self.ros_clock, self.stop, self.audit = root, votes, ros_clock, stop, audit
        self.error = None

    def run(self):
        try:
            path, offset, partial = None, 0, b''
            while not self.stop.wait(.05):
                if path is None:
                    files = list(self.root.glob('*/events.jsonl'))
                    if not files:
                        continue
                    if len(files) != 1:
                        raise RuntimeError('More than one monitor session in this demo directory')
                    path = files[0]
                with path.open('rb') as f:
                    f.seek(offset)
                    data = f.read(1 << 20)
                    offset = f.tell()
                parts = (partial + data).split(b'\n')
                partial = parts.pop()
                for line in parts:
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get('event') == 'judgment':
                        self.votes.feed(row, self.ros_clock(), time.monotonic())
                        self.audit.emit('vlm_vote', source=str(path), frame=row.get('frame'),
                                        parsed=row.get('parsed'), accepted_by_monitor=row.get('accepted_current'),
                                        votes=self.votes.snapshot(self.ros_clock(), time.monotonic()))
                    elif row.get('event') == 'heartbeat' and row.get('camera_rejection'):
                        self.votes.reset()
                        self.audit.emit('vlm_votes_reset', reason=row['camera_rejection'])
                    elif row.get('event') == 'presentation_dropped':
                        self.votes.reset()
                        self.audit.emit('vlm_votes_reset', reason='result_expired_before_presentation')
                    elif row.get('event') in ['worker_error', 'session_finished']:
                        raise RuntimeError(f"Monitor terminated: {row.get('error') or row['event']}")
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            self.audit.emit('monitor_reader_error', error=self.error)


class SkillClients:
    """Existing PolicyClient on private demo servers, with strict reset/metadata checks."""
    def __init__(self, spec, token):
        import zmq
        from kuavo_deploy.kuavo_service.client import PolicyClient
        self.zmq, self.spec, self.clients = zmq, spec, {}
        for skill in ['normal', 'recovery']:
            client = PolicyClient(host='127.0.0.1', port=spec['expected_servers'][skill]['port'],
                                  task_prompt=spec[skill + '_instruction'])
            client.policy.api_token = token
            self.clients[skill] = client
            self.socket_options(client, 2000)

    def socket_options(self, client, timeout):
        for option, value in [(self.zmq.LINGER, 0), (self.zmq.SNDTIMEO, 2000), (self.zmq.RCVTIMEO, timeout)]:
            client.policy.socket.setsockopt(option, value)

    def rpc(self, skill, endpoint):
        client = self.clients[skill]
        response = client.policy.call_endpoint(endpoint, requires_input=False)
        if not isinstance(response, dict) or response.get('error') or response.get('status') != 'ok':
            raise RuntimeError(f'{skill} {endpoint} failed: {response}')
        return response

    def ready(self, skill):
        client = self.clients[skill]
        try:
            response = self.rpc(skill, 'metadata')
            expected_metadata(response, self.spec['expected_servers'][skill])
            return True
        except self.zmq.ZMQError:
            client.policy.socket.close(linger=0)
            client.policy._init_socket()
            self.socket_options(client, 2000)
            return False

    def arm(self):
        for skill, client in self.clients.items():
            self.socket_options(client, self.spec['rpc_timeout_ms'])
            self.rpc(skill, 'reset')

    def reset(self, skill):
        self.rpc(skill, 'reset')  # unlike PolicyClient.reset(), errors MUST NOT be swallowed

    def action(self, skill, observation):
        return self.clients[skill].select_action(observation)

    def close(self):
        for client in self.clients.values():
            client.policy.socket.close(linger=0)
            client.policy.context.term()


def execute(deploy_config, spec):
    """All ROS, model-server, and actuator work is confined to this explicit path."""
    from datetime import datetime, timezone
    cfg = validate_deployment(deploy_config)
    directory = ROOT / 'outputs/skill_demo' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    directory.mkdir(parents=True, exist_ok=False)
    audit = Audit(directory)
    (directory / 'settings.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2))
    children = OwnedProcesses(audit)
    stop, paused, reader_stop = threading.Event(), threading.Event(), threading.Event()
    clients, reader, arm = None, None, None
    service_lease = None
    token = uuid.uuid4().hex
    votes = FreshVotes(spec['max_vlm_result_age'], spec['max_vote_gap_seconds'])
    print(f'AUTOMATIC SKILL DEMO logs: {directory}', flush=True)

    def stop_handler(*_):
        stop.set()
        if arm is not None:
            for manager in [arm.env.control_signal_manager, arm.env.obs_buffer.control_signal_manager]:
                manager.stop_flag.set()

    def pause_handler(*_):
        paused.clear() if paused.is_set() else paused.set()

    def install_signals():
        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGUSR2]:
            signal.signal(sig, stop_handler)
        signal.signal(signal.SIGUSR1, pause_handler)

    install_signals()
    try:
        entry_cfg = spec['recovery_right_arm_entry']
        entry_reference = None
        if entry_cfg['enabled']:
            from kuavo_deploy.utils.recovery_entry import load_bag_right_reference
            joint_source = cfg.env.obs_key_map['joint_q']
            entry_reference = load_bag_right_reference(entry_cfg['bag_path'], joint_source['topic'],
                joint_source['handle']['params']['slice'], cfg.env.limits['joint_q']['min'],
                cfg.env.limits['joint_q']['max'])
            audit.emit('right_arm_entry_reference', **entry_reference)
        if spec.get('qwen_service_url'):
            from vlm_test.qwen_service import RemoteQwenAdapter
            RemoteQwenAdapter(spec['qwen_service_url'], root_path(spec['vlm_model']), spec['vlm_precision'])
            audit.emit('qwen_service_ready', url=spec['qwen_service_url'], owned_by_demo=False)
        if spec.get('skill_service_manifest'):
            import fcntl
            manifest = root_path(spec['skill_service_manifest'])
            if not manifest.is_file():
                raise RuntimeError('Start python -m vlm_test.skill_service and wait for Skills ready first')
            service_lease = manifest.with_suffix('.client.lock').open('a')
            fcntl.flock(service_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            service = json.loads(manifest.read_text())
            if service['expected_servers'] != spec['expected_servers']:
                raise RuntimeError('Persistent checkpoint configuration differs from Demo configuration')
            token = service['token']
            clients = SkillClients(spec, token)
            if not all(clients.ready(skill) for skill in ['normal', 'recovery']):
                raise RuntimeError('Persistent VLA service unavailable; restart skill_service')
            audit.emit('skill_service_connected', manifest=str(manifest), owned_by_demo=False)
        else:
            from kuavo_server.launch import build_serve_argv, load_config, wrap_with_env
            launch_cfg = load_config(root_path(spec['server_config']))
            for skill in ['normal', 'recovery']:
                profile = spec[skill + '_profile']
                command = build_serve_argv(profile, launch_cfg, ['--api_token', token])
                command = wrap_with_env(command, launch_cfg['adapters'][profile].get('env', {}))
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(spec['vla_gpu']))
                children.start(skill + '_server', command, env)
        monitor_root = directory / 'monitor'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(spec['vlm_gpu']), OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
        monitor_extra = ['--qwen-service-url', spec['qwen_service_url']] if spec.get('qwen_service_url') else []
        children.start('vlm_monitor', [str(root_path(spec['vlm_python'])), '-u', '-m', 'vlm_test.passive_vlm_monitor',
                                     '--deploy-config', str(deploy_config), '--output-root', str(monitor_root),
                                     '--prompt', str(root_path(spec['vlm_prompt'])), '--precision', spec['vlm_precision'],
                                     '--model-path', str(root_path(spec['vlm_model'])),
                                     '--max-result-age', str(spec['max_vlm_result_age'])] + monitor_extra, env)
        if clients is None:
            clients = SkillClients(spec, token)
        deadline = time.monotonic() + spec['startup_timeout']
        ready = set()
        while not stop.is_set() and len(ready) != 2:
            children.check()
            for skill in ['normal', 'recovery']:
                if skill not in ready and clients.ready(skill):
                    ready.add(skill)
                    audit.emit('server_ready', skill=skill, expected=spec['expected_servers'][skill])
            if time.monotonic() > deadline:
                raise TimeoutError('Model servers did not become ready before startup timeout')
            stop.wait(.2)
        if stop.is_set():
            return
        clients.arm()
        # Reuse exactly the normal deployment environment; never reset it on a skill switch.
        import rospy
        import numpy as np
        from kuavo_deploy.src.scripts.script import ArmMove
        from kuavo_deploy.utils.logging_utils import setup_logger
        arm = ArmMove(cfg)
        install_signals()  # ArmMove installs its own USR handlers; this task owns its lifecycle.
        env = arm.env
        log = setup_logger('model', logging.INFO)
        ros_now = lambda: rospy.Time.now().to_sec()
        reader = MonitorReader(monitor_root, votes, ros_now, reader_stop, audit)
        reader.start()
        monitor_resume_serial = 0
        active = 'normal'

        def snapshot():
            return votes.snapshot(ros_now(), time.monotonic())

        def guard(require_monitor=True):
            nonlocal monitor_resume_serial
            if stop.is_set() or rospy.is_shutdown() or env.control_signal_manager.stop_flag.is_set():
                raise InterruptedError('Demo stop requested')
            children.check()
            if reader.error:
                raise RuntimeError(reader.error)
            s = snapshot()
            if require_monitor and (s['last_seen_mono'] is None or time.monotonic() - s['last_seen_mono'] > spec['monitor_loss_timeout']):
                audit.emit('state', state='WAITING_FOR_VLM', active_skill=active)
                log.warning('VLM unavailable: pausing new action submissions; waiting for fresh stable judgments')
                clients.reset(active)
                votes.reset(ros_now())
                deadline_vlm = time.monotonic() + spec['monitor_recovery_timeout']
                while True:
                    guard(require_monitor=False)  # stop/process failures still exit immediately
                    s = snapshot()
                    if stable_monitor_vote(s, spec['normal_hits'], spec['recovery_hits']):
                        monitor_resume_serial += 1
                        audit.emit('state', state='VLM_AVAILABLE', active_skill=active, evidence=s)
                        log.info('Fresh stable VLM judgments restored; re-evaluating skill before new inference')
                        break
                    if time.monotonic() >= deadline_vlm:
                        raise TimeoutError('VLM did not recover while action submission was paused')
                    stop.wait(.05)
            return s

        def observations():
            # Source timestamps are checked before reading and again before action submission.
            now = ros_now()
            for key in ['head_cam_h', 'wrist_cam_l', 'wrist_cam_r', 'joint_q', 'gripper']:
                buf = env.obs_buffer.obs_buffer_data.get(key)
                if not buf or not buf['timestamp']:
                    raise RuntimeError(f'Missing observation: {key}')
                stamp = float(buf['timestamp'][-1])
                if stamp <= 0 or not -0.05 <= now - stamp <= spec['max_observation_age']:
                    raise RuntimeError(f'Stale/foreign-clock observation: {key}, stamp={stamp}, ros_now={now}')
            obs = env.get_obs()
            state = obs['observation.state'].detach().cpu().numpy().reshape(-1)
            if state.shape != (16,) or not np.all(np.isfinite(state)):
                raise RuntimeError('Expected finite 16-D measured state')
            return obs, state

        def still():
            samples = deque()
            end = time.monotonic() + spec['settle_timeout']
            while time.monotonic() < end:
                guard()
                _, state = observations()
                q = np.concatenate([state[:7], state[8:15]])
                t = time.monotonic()
                samples.append((t, q))
                while len(samples) > 2 and t - samples[1][0] >= spec['settle_window']:
                    samples.popleft()
                if t - samples[0][0] >= spec['settle_window']:
                    qs = np.stack([v for _, v in samples])
                    speed = max(float(np.max(np.abs(b[1] - a[1]))) / max(b[0] - a[0], .001)
                                for a, b in zip(samples, list(samples)[1:]))
                    if float(np.max(np.ptp(qs, axis=0))) <= spec['settle_max_joint_delta'] and speed <= spec['settle_max_joint_velocity']:
                        audit.emit('measured_arm_settled', window=spec['settle_window'], max_velocity=speed)
                        return
                stop.wait(.05)
            raise TimeoutError('Measured arm did not settle; skill switch refused')

        def right_arm_entry():
            from kuavo_deploy.utils.recovery_entry import move_right_arm

            def read_arm():
                buf = env.obs_buffer.obs_buffer_data['joint_q']
                if not buf['data'] or not buf['timestamp']:
                    raise RuntimeError('No arm feedback during right-arm entry')
                stamp, q = float(buf['timestamp'][-1]), list(buf['data'][-1])
                if not np.isfinite(stamp) or not -0.05 <= ros_now() - stamp <= spec['max_observation_age']:
                    raise RuntimeError('Stale/foreign-clock arm feedback during right-arm entry')
                if len(q) != 14 or any(not lo <= value <= hi for value, lo, hi in
                                      zip(q, cfg.env.limits['joint_q']['min'], cfg.env.limits['joint_q']['max'])):
                    raise RuntimeError('Measured arm outside configured joint limits during right-arm entry')
                return stamp, q

            def check_motion():
                guard()
                while paused.is_set():
                    guard()
                    stop.wait(.05)

            audit.emit('state', state='RIGHT_ARM_PREPOSITIONING')
            log.info('Moving right arm to first measured bag pose; left target held, no hand commands')
            def entry_event(event, **fields):
                audit.emit(event, **fields)
                if event in ['right_arm_entry_feedback', 'right_arm_entry_timeout']:
                    log.info('Right-arm arrival: joint=%s max_error=%.4f rad tolerance=%.4f rad remaining=%.1fs',
                             fields['worst_joint'], fields['max_error_rad'], fields['arrival_tolerance'],
                             fields.get('remaining_seconds', 0.))
            move_right_arm(entry_reference, entry_cfg, read_arm, env.safe_control_arm, check_motion,
                           stop.wait, time.monotonic, entry_event,
                           previous_command=env.last_safety_sent_arm_command)
            still()
            audit.emit('state', state='RIGHT_ARM_READY')

        # Do not initialize/reset the robot until all inference services are ready and seeing fresh frames.
        while not stop.is_set():
            initial = snapshot()
            required = spec['normal_hits'] if initial['last'] and initial['last']['skill'] == 'normal' else spec['recovery_hits']
            if initial['fresh'] and initial['streak'] >= required:
                break
            guard(require_monitor=False)
            if time.monotonic() > deadline:
                raise TimeoutError('No stable fresh VLM scene after startup')
            stop.wait(.05)
        if stop.is_set():
            return
        guard()
        while paused.is_set():
            guard()
            stop.wait(.05)
        env.reset()  # Same initial external-control/head/gripper initialization as real_single_test.
        observations()
        active, recoveries, steps, normal_steps, release_since = 'normal', 0, 0, 0, None
        started = phase_start = time.monotonic()
        last_action = None
        recovery_cycle = None
        votes.reset(ros_now())
        # Re-evaluate AFTER initialization, which can change the scene/gripper state.
        while True:
            s = guard()
            required = spec['normal_hits'] if s['last'] and s['last']['skill'] == 'normal' else spec['recovery_hits']
            if s['fresh'] and s['streak'] >= required:
                break
            if time.monotonic() - started > spec['verify_timeout']:
                raise TimeoutError('No stable scene after environment initialization')
            stop.wait(.05)
        audit.emit('state', state='NORMAL', checkpoint=spec['expected_servers']['normal']['checkpoint'])
        while not stop.is_set():
            if time.monotonic() - started > spec['demo_timeout']:
                raise TimeoutError('Demo duration limit reached')
            s = guard()
            if paused.is_set():
                clients.reset(active)
                audit.emit('state', state='OPERATOR_PAUSED', active_skill=active)
                while paused.is_set() and not stop.is_set():
                    guard()
                    stop.wait(.05)
                votes.reset(ros_now())
                continue
            if active == 'normal' and s['fresh'] and s['last']['skill'] == 'recovery' and s['streak'] >= spec['recovery_hits']:
                audit.emit('state', state='STOPPING_NORMAL', trigger=s)
                clients.reset('normal')  # Synchronous actor has no local action queue or in-flight request here.
                still()
                votes.reset(ros_now())  # Only frames captured after settling may decide the switch.
                decision = PostStopDecision(time.monotonic() + spec['verify_timeout'],
                                            spec['normal_hits'], spec['recovery_hits'])
                audit.emit('state', state='VERIFY_SWITCH')
                log.info('Waiting up to %.0fs for stable post-stop VLM classification; no new actions', spec['verify_timeout'])
                while True:
                    s = guard(require_monitor=False)  # Bounded wait here handles absent/unstable results.
                    observations()
                    selected = decision.evaluate(s, time.monotonic())
                    if selected is not None:
                        break
                    stop.wait(.05)
                if selected == 'normal':
                    clients.reset('normal')
                    last_action = None
                    votes.reset(ros_now())
                    audit.emit('state', state='RECOVERY_CANCELLED_NORMAL_RESUMED', evidence=s)
                    log.info('Post-stop scene is stably normal; Recovery cancelled, Normal resumes')
                    continue
                if recoveries >= spec['max_recoveries']:
                    raise RuntimeError('Recovery attempt limit reached')
                if entry_cfg['enabled']:
                    right_arm_entry()
                    # The transition changed the image/state: never reuse the pre-motion votes.
                    votes.reset(ros_now())
                    decision = PostStopDecision(time.monotonic() + spec['verify_timeout'],
                                               spec['normal_hits'], spec['recovery_hits'])
                    audit.emit('state', state='VERIFY_AFTER_RIGHT_ARM_ENTRY')
                    while True:
                        s = guard(require_monitor=False)
                        observations()
                        selected = decision.evaluate(s, time.monotonic())
                        if selected is not None:
                            break
                        stop.wait(.05)
                    if selected == 'normal':
                        clients.reset('normal')
                        last_action = None
                        votes.reset(ros_now())
                        audit.emit('state', state='RECOVERY_CANCELLED_NORMAL_RESUMED', evidence=s,
                                   reason='Scene normal after right-arm entry')
                        continue
                clients.reset('recovery')
                active, phase_start, release_since, last_action = 'recovery', time.monotonic(), None, None
                recoveries += 1
                recovery_cycle = RecoveryGripperCycle(spec)
                votes.reset(ros_now())
                audit.emit('state', state='RECOVERING', attempt=recoveries, checkpoint=spec['expected_servers']['recovery']['checkpoint'])
                log.info('Executing Recovery; completion requires gripper closure/release and scene verification; timeout=%.0fs', spec['recovery_timeout'])
                continue
            if active == 'recovery' and time.monotonic() - phase_start > spec['recovery_timeout']:
                raise TimeoutError(f'Recovery did not complete before timeout; gripper_stage={recovery_cycle.stage}')
            obs, state = observations()
            if active == 'recovery':
                previous_stage = recovery_cycle.stage
                stage = recovery_cycle.update(float(last_action[7]) if last_action is not None else None,
                                              float(state[7]), time.monotonic())
                if stage != previous_stage:
                    audit.emit('recovery_gripper_stage', stage=stage, command_left_gripper=float(last_action[7]),
                               measured_left_gripper=float(state[7]))
                    log.info('Recovery gripper stage: %s', stage)
                    votes.reset(ros_now())  # Completion needs normal frames captured AFTER the release.
                    s = snapshot()
            if (active == 'recovery' and time.monotonic() - phase_start >= spec['minimum_recovery_seconds']
                    and recovery_cycle.stage == 'released'
                    and s['fresh'] and s['last']['skill'] == 'normal' and s['streak'] >= spec['normal_hits']):
                audit.emit('state', state='VERIFY_RECOVERY', evidence=s, measured_left_gripper=float(state[7]))
                log.info('Recovery actions stopping; post-stop scene verification timeout=%.0fs', spec['verify_timeout'])
                clients.reset('recovery')
                still()
                votes.reset(ros_now())  # Fresh post-stop scene verification, not old in-motion normal votes.
                verify_end = time.monotonic() + spec['verify_timeout']
                while True:
                    s = guard()
                    _, measured = observations()
                    if (s['fresh'] and s['last']['skill'] == 'normal' and s['streak'] >= spec['normal_hits']
                            and float(measured[7]) <= spec['left_release_threshold']):
                        break
                    if time.monotonic() > verify_end:
                        raise TimeoutError('Recovery post-stop normal/release verification failed')
                    stop.wait(.05)
                clients.reset('normal')
                active, release_since, last_action = 'normal', None, None
                recovery_cycle = None
                votes.reset(ros_now())
                audit.emit('state', state='NORMAL_RESUMED', recoveries=recoveries)
                continue
            t = time.monotonic()
            inference_serial = monitor_resume_serial
            action = clients.action(active, obs).detach().cpu().numpy().reshape(-1)
            # An inference call can be slow. Recheck trigger, images, stop and pause BEFORE submitting it.
            latest = guard()
            observations()
            if inference_serial != monitor_resume_serial:
                audit.emit('action_discarded', reason='Monitor paused/resumed during inference; cached action cleared')
                continue
            if paused.is_set() or stop.is_set():
                continue
            if active == 'normal' and latest['fresh'] and latest['last']['skill'] == 'recovery' and latest['streak'] >= spec['recovery_hits']:
                audit.emit('normal_action_discarded', reason='Recovery trigger arrived during inference')
                continue
            if action.shape != (16,) or not np.all(np.isfinite(action)):
                raise ValueError('Policy returned an invalid action')
            audit.emit('action_submission', active_skill=active, action=action.tolist())
            _, _, terminated, truncated, _ = env.step(action)  # Existing control + SafetySupervisor unchanged.
            last_action = action
            steps += 1
            normal_steps += active == 'normal'
            log.info('Step %d time: %.3fs skill=%s', steps, time.monotonic() - t, active)
            if terminated or truncated or (active == 'normal' and normal_steps >= cfg.inference.max_episode_steps):
                audit.emit('state', state='DEMO_STEP_LIMIT_OR_ENV_DONE', steps=steps, normal_steps=normal_steps)
                break
    except InterruptedError as exc:
        audit.emit('stopped', reason=str(exc))
    except BaseException as exc:
        audit.emit('fault', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        stop.set()
        reader_stop.set()
        if reader is not None:
            reader.join(timeout=2)
        # No more env.step calls; terminating only owned servers also destroys their cached chunks.
        if clients is not None:
            if service_lease is not None:
                for skill in ['normal', 'recovery']:
                    try:
                        clients.reset(skill)
                    except Exception as exc:
                        audit.emit('service_reset_failed', skill=skill, error=str(exc))
            clients.close()
        if service_lease is not None:
            service_lease.close()
        children.close()
        if arm is not None:
            arm.env.close()
        audit.emit('finished', robot_actions_no_longer_submitted=True)
        audit.file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--demo-config', type=Path, default=ROOT / 'configs/deploy/lingbot_skill_demo.yaml')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    spec = load_demo(args.demo_config)
    validate_deployment(args.config)
    if args.preflight or not args.execute:
        print(json.dumps(dict(mode='preflight_no_model_no_ros_no_motion', servers=spec['expected_servers'],
                              recovery_hits=spec['recovery_hits'], normal_hits=spec['normal_hits']), indent=2))
        return
    execute(args.config.resolve(), spec)


if __name__ == '__main__':
    main()
