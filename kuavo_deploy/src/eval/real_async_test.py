# Copyright (C) 2025-2026 LejuRobotics.

from __future__ import annotations

import datetime
import time
import traceback
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import numpy as np
import rospy
import torch
from std_msgs.msg import Bool
from tqdm import tqdm

from lerobot.utils.random_utils import set_seed
from lerobot_patches import custom_patches  # noqa: F401

from kuavo_deploy.config import KuavoConfig
from kuavo_deploy.kuavo_service.client import PolicyClient
from kuavo_deploy.src.eval.async_action_buffer import ActionTimelineBuffer
from kuavo_deploy.utils.logging_utils import setup_logger
from kuavo_deploy.utils.policy_loader import (
    inject_task_prompt,
    load_native_policy_bundle,
    resolve_eval_output_dir,
)

log_model = setup_logger("model")
log_robot = setup_logger("robot")

pause_flag = Event()
stop_flag = Event()


def pause_callback(msg):
    if msg.data:
        pause_flag.set()
    else:
        pause_flag.clear()


def stop_callback(msg):
    if msg.data:
        stop_flag.set()


pause_sub = rospy.Subscriber("/kuavo/pause_state", Bool, pause_callback, queue_size=10)
stop_sub = rospy.Subscriber("/kuavo/stop_state", Bool, stop_callback, queue_size=10)


def setup_policy(pretrained_path, policy_type, device, task_prompt: str):
    if device.type == "cpu":
        log_model.warning("Using CPU for inference, this may be slow.")
        time.sleep(3)

    if policy_type == "client":
        preprocessor, postprocessor = lambda obs: obs, lambda action: action
        return PolicyClient(task_prompt=task_prompt), preprocessor, postprocessor, None

    policy, preprocessor, postprocessor, pretrained_model_dir = load_native_policy_bundle(
        pretrained_path=pretrained_path,
        device=device,
        strict=True,
    )
    log_model.info(f"Model loaded from {pretrained_model_dir}")
    log_model.info(f"Model type: {policy.config.type}")
    log_model.info(f"Model n_obs_steps: {policy.config.n_obs_steps}")
    return policy, preprocessor, postprocessor, pretrained_model_dir


def _to_chunk_tensor(actions: Any) -> torch.Tensor:
    if isinstance(actions, torch.Tensor):
        tensor = actions
    elif isinstance(actions, np.ndarray):
        tensor = torch.from_numpy(actions)
    else:
        tensor = torch.as_tensor(actions)

    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected action chunk shape [T, D], got {tuple(tensor.shape)}")
    return tensor


def _postprocess_chunk(actions: Any, postprocessor) -> list[np.ndarray]:
    chunk = _to_chunk_tensor(actions)
    out: list[np.ndarray] = []
    for idx in range(chunk.shape[0]):
        action = chunk[idx : idx + 1]
        processed = postprocessor(action)
        processed = _to_chunk_tensor(processed)
        out.append(processed[0].detach().cpu().numpy())
    return out


def _select_action_chunk(policy, observation: dict[str, Any]) -> Any:
    if hasattr(policy, "select_action_chunk"):
        return policy.select_action_chunk(observation)
    if hasattr(policy, "predict_action_chunk"):
        return policy.predict_action_chunk(observation)
    return policy.select_action(observation)


def inference_worker(
    *,
    config: KuavoConfig,
    env,
    policy,
    preprocessor,
    postprocessor,
    buffer: ActionTimelineBuffer,
    timeline_lock: Lock,
    stop_event: Event,
) -> None:
    cfg = config.inference
    policy_type = cfg.policy_type
    task_prompt = getattr(cfg, "task_prompt", "robot manipulation")
    low_watermark = max(0, int(cfg.async_low_watermark))
    fallback_chunk_id = 0

    try:
        while not stop_event.is_set() and not rospy.is_shutdown():
            if stop_flag.is_set():
                stop_event.set()
                break
            if pause_flag.is_set():
                time.sleep(0.05)
                continue
            if buffer.qsize() > low_watermark:
                time.sleep(0.005)
                continue

            # Capture the observation and timeline position between control
            # steps. Recheck the watermark after acquiring the lock because the
            # consumer may have advanced while this worker was waiting.
            with timeline_lock:
                if buffer.qsize() > low_watermark:
                    continue
                trigger = buffer.snapshot_trigger()
                observation = env.get_obs()
            if policy_type != "client":
                observation = inject_task_prompt(observation, task_prompt)
            observation = preprocessor(observation)

            start = time.time()
            with torch.inference_mode():
                if hasattr(policy, "select_action_chunk_async"):
                    response = policy.select_action_chunk_async(
                        observation, trigger.as_request_context()
                    )
                else:
                    response = {
                        "actions": _select_action_chunk(policy, observation),
                        "chunk_id": fallback_chunk_id,
                        "chunk_start_global_step": trigger.trigger_global_step,
                        "previous_chunk_id": trigger.previous_chunk_id,
                        "rtc_previous_offset": trigger.executed_offset_at_trigger,
                    }
                    fallback_chunk_id += 1

            if not isinstance(response, dict) or "actions" not in response:
                raise ValueError("Async inference must return actions plus chunk metadata")
            actions_np = _postprocess_chunk(response["actions"], postprocessor)
            execution_horizon = int(response.get("execution_horizon", len(actions_np)))
            chunk_id = int(response["chunk_id"])
            chunk_start = int(response["chunk_start_global_step"])
            if chunk_start != trigger.trigger_global_step:
                raise ValueError(
                    "Async response changed the chunk time origin: "
                    f"trigger={trigger.trigger_global_step}, response_start={chunk_start}"
                )
            response_previous_chunk_id = response.get("previous_chunk_id")
            response_previous_offset = response.get("rtc_previous_offset")
            if response_previous_chunk_id != trigger.previous_chunk_id:
                raise ValueError(
                    "Async response changed chunk lineage: "
                    f"trigger_previous={trigger.previous_chunk_id}, "
                    f"response_previous={response_previous_chunk_id}"
                )
            if response_previous_offset != trigger.executed_offset_at_trigger:
                raise ValueError(
                    "RTC response used a different physical-time offset: "
                    f"trigger_offset={trigger.executed_offset_at_trigger}, "
                    f"rtc_offset={response_previous_offset}"
                )

            with timeline_lock:
                merge = buffer.replace_with_chunk(
                    actions_np,
                    chunk_id=chunk_id,
                    chunk_start_global_step=chunk_start,
                    execution_horizon=execution_horizon,
                )
            elapsed = time.time() - start
            log_model.info(
                f"Async chunk ready: chunk_id={chunk_id}, produced={len(actions_np)}, "
                f"inserted={merge.inserted}, trigger_step={trigger.trigger_global_step}, "
                f"ready_step={merge.ready_global_step}, stale={merge.stale}, "
                f"previous_chunk_id={trigger.previous_chunk_id}, "
                f"previous_offset={trigger.executed_offset_at_trigger}, "
                f"rtc_applied={response.get('rtc_applied')}, "
                f"first_new_offset={merge.first_chunk_offset}, "
                f"buffer={buffer.qsize()}, time={elapsed:.3f}s"
            )
            if merge.inserted == 0:
                log_model.error(
                    f"Async chunk {chunk_id} is fully stale; stopping instead of "
                    f"advancing server/client chunk lineage: stale={merge.stale}, "
                    f"produced={len(actions_np)}"
                )
                stop_event.set()
                break
    except Exception:
        log_model.error("Async inference worker failed:\n" + traceback.format_exc())
        stop_event.set()


def control_worker(
    *,
    env,
    buffer: ActionTimelineBuffer,
    timeline_lock: Lock,
    stop_event: Event,
    max_steps: int,
    action_timeout: float,
) -> int:
    step = 0
    try:
        with tqdm(total=max_steps, desc="Async episode", unit="step", leave=False) as pbar:
            while step < max_steps and not stop_event.is_set() and not rospy.is_shutdown():
                while pause_flag.is_set() and not stop_flag.is_set():
                    log_model.info("Paused. Waiting for resume signal...")
                    time.sleep(0.5)
                if stop_flag.is_set():
                    stop_event.set()
                    break

                if not buffer.wait_for_action(timeout=action_timeout):
                    log_model.error(
                        "No time-aligned action available before timeout; stopping async rollout."
                    )
                    stop_event.set()
                    break

                with timeline_lock:
                    entry = buffer.pop_next()
                    if entry is None:
                        continue
                    env.step(entry.action)
                    buffer.mark_step_executed()
                step += 1
                pbar.update(1)
    except Exception:
        log_robot.error("Async control worker failed:\n" + traceback.format_exc())
        stop_event.set()
    return step


def kuavo_eval_async(config: KuavoConfig, env) -> None:
    cfg = config.inference
    eval_episodes = cfg.eval_episodes
    policy_type = cfg.policy_type
    task_prompt = getattr(cfg, "task_prompt", "robot manipulation")

    pretrained_path = (
        Path(cfg.pretrained_path)
        if cfg.pretrained_path
        else Path(f"outputs/train/{cfg.task}/{cfg.method}/{cfg.timestamp}/epoch{cfg.epoch}")
    )
    output_directory = resolve_eval_output_dir(pretrained_path, Path("outputs/eval"))
    output_directory.mkdir(parents=True, exist_ok=True)

    set_seed(seed=cfg.seed)
    device = torch.device(cfg.device)
    policy, preprocessor, postprocessor, _ = setup_policy(
        pretrained_path,
        policy_type,
        device,
        task_prompt=task_prompt,
    )

    if cfg.async_control_hz and cfg.async_control_hz > 0:
        env.rate = rospy.Rate(cfg.async_control_hz)
        log_robot.info(f"Async control rate set to {cfg.async_control_hz} Hz")

    log_file_path = output_directory / "evaluation_async.log"
    with log_file_path.open("w") as log_file:
        log_file.write(f"Evaluation Timestamp: {datetime.datetime.now()}\n")
        log_file.write(f"Total Episodes: {eval_episodes}\n")
        log_file.write(f"Policy Type: {policy_type}\n")

    for episode in tqdm(range(eval_episodes), desc="Async evaluating", unit="episode"):
        if stop_flag.is_set() or rospy.is_shutdown():
            break

        policy.reset()
        env.reset(seed=episode + cfg.start_seed)

        buffer = ActionTimelineBuffer(maxlen=cfg.async_buffer_size)
        timeline_lock = Lock()
        stop_event = Event()
        infer_thread = Thread(
            target=inference_worker,
            kwargs={
                "config": config,
                "env": env,
                "policy": policy,
                "preprocessor": preprocessor,
                "postprocessor": postprocessor,
                "buffer": buffer,
                "timeline_lock": timeline_lock,
                "stop_event": stop_event,
            },
            daemon=True,
            name="kuavo-async-inference",
        )
        infer_thread.start()

        warmup_actions = max(1, int(cfg.async_warmup_actions))
        if not buffer.wait_for_size(warmup_actions, timeout=cfg.async_action_timeout):
            log_model.warning("Warmup action wait timed out; control loop will wait on the buffer.")

        steps = control_worker(
            env=env,
            buffer=buffer,
            timeline_lock=timeline_lock,
            stop_event=stop_event,
            max_steps=cfg.max_episode_steps,
            action_timeout=cfg.async_action_timeout,
        )
        stop_event.set()
        infer_thread.join(timeout=2.0)

        with log_file_path.open("a") as log_file:
            log_file.write(f"Episode {episode + 1}: steps={steps}\n")

        if stop_flag.is_set():
            break
