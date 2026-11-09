"""Load both existing LingBot checkpoint servers once; no ROS or robot control."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import threading
import uuid
from kuavo_deploy.utils.skill_demo import ROOT, load_demo, root_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo-config', type=Path, default=ROOT / 'configs/deploy/lingbot_skill_demo.yaml')
    parser.add_argument('--gpu', help='Physical GPU IDs for both VLA servers; default: vla_gpu in config')
    args = parser.parse_args()
    spec = load_demo(args.demo_config)
    manifest = root_path(spec.get('skill_service_manifest') or 'outputs/skill_servers/current.json')
    manifest.parent.mkdir(parents=True, exist_ok=True)
    owner = manifest.with_suffix('.owner.lock').open('a')
    try:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('A checkpoint service is already running for this manifest')
    from kuavo_deploy.src.scripts.script_skill_demo import Audit, OwnedProcesses, SkillClients
    from kuavo_server.launch import build_serve_argv, load_config, wrap_with_env
    run = manifest.parent / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir()
    audit = Audit(run)
    children = OwnedProcesses(audit)
    stop = threading.Event()
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda *_: stop.set())
    token, clients, published = uuid.uuid4().hex, None, False
    try:
        cfg = load_config(root_path(spec['server_config']))
        gpu = args.gpu if args.gpu is not None else str(spec['vla_gpu'])
        print(f'Loading Normal + Recovery on GPU {gpu}. Logs: {run}', flush=True)
        for skill in ['normal', 'recovery']:
            profile = spec[skill + '_profile']
            cmd = build_serve_argv(profile, cfg, ['--api_token', token])
            cmd = wrap_with_env(cmd, cfg['adapters'][profile].get('env', {}))
            children.start(skill + '_server', cmd, dict(os.environ, CUDA_VISIBLE_DEVICES=gpu))
        clients = SkillClients(spec, token)
        import time
        deadline, ready = time.monotonic() + spec['startup_timeout'], set()
        while not stop.is_set() and len(ready) < 2:
            children.check()
            for skill in ['normal', 'recovery']:
                if skill not in ready and clients.ready(skill):
                    ready.add(skill)
                    audit.emit('server_ready', skill=skill)
                    print(f'{skill}: server_ready', flush=True)
            if time.monotonic() > deadline:
                raise TimeoutError('Checkpoint service startup timed out')
            stop.wait(.2)
        if stop.is_set():
            return
        # Only metadata requests above: never select_action or initialize a robot.
        temporary = manifest.with_suffix('.tmp')
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as f:
            json.dump(dict(token=token, expected_servers=spec['expected_servers'], logs=str(run), gpu=gpu), f)
        temporary.replace(manifest)
        published = True
        def watch():
            try:
                while not stop.wait(.5):
                    children.check()
            except Exception as exc:
                print(f'Checkpoint service failed: {exc}', flush=True)
                stop.set()
        threading.Thread(target=watch, daemon=True).start()
        print('Skills ready. Enter=保持加载；q / quit / exit=关闭。现在可运行 eval.py。', flush=True)
        # Daemon input thread permits SIGTERM/child failure to exit even while waiting for input.
        def console():
            while not stop.is_set():
                try:
                    command = input('Skills [Enter=保持 / q=关闭]> ').strip().lower()
                except EOFError:
                    return  # Keep loaded until signal.
                if command in ['q', 'quit', 'exit']:
                    stop.set()
                    return
                print('两个 checkpoint 保持加载。', flush=True)
        threading.Thread(target=console, daemon=True).start()
        stop.wait()
    finally:
        if published:
            manifest.unlink(missing_ok=True)
        if clients is not None:
            clients.close()
        children.close()
        audit.file.close()
        owner.close()
    print('两个 checkpoint 服务已关闭。', flush=True)


if __name__ == '__main__':
    main()
