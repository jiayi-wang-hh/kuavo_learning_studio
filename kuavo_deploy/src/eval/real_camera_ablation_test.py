#!/usr/bin/env python3
"""Observe-only camera ablation test for a real Kuavo robot.

One live observation is frozen. The policy is repeatedly evaluated with the
original observation and with one camera ablated at a time. Predicted actions
are saved but never executed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import rospy
import torch
import torch.nn.functional as F

import kuavo_deploy.kuavo_env  # noqa: F401  # Register Kuavo-Real.
from kuavo_deploy.config import load_kuavo_config
from kuavo_deploy.kuavo_service.client import PolicyClient
from kuavo_deploy.src.eval.real_observation_contrast_test import (
    action_to_numpy,
    save_observation,
    save_plot,
    summarize_conditions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="真机三路相机消融：冻结同一观测，只预测动作，不执行。"
    )
    parser.add_argument("--config", type=Path, required=True, help="真机 deploy YAML")
    parser.add_argument("--samples-per-condition", type=int, default=30)
    parser.add_argument(
        "--mode",
        choices=("mean", "zero", "blur"),
        default="mean",
        help="mean=通道均值图；zero=全零；blur=强模糊",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=101,
        help="blur 模式的奇数卷积核尺寸",
    )
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help="只消融指定相机短名或完整 observation key；默认自动测试全部相机",
    )
    parser.add_argument("--prompt", default=None, help="覆盖 YAML 中的 task_prompt")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/real_camera_ablation"),
    )
    parser.add_argument("--yes", action="store_true", help="跳过 OBSERVE 启动确认")
    args = parser.parse_args()

    if args.samples_per_condition <= 0:
        parser.error("--samples-per-condition 必须大于 0")
    if args.interval < 0:
        parser.error("--interval 不能小于 0")
    if args.blur_kernel <= 0 or args.blur_kernel % 2 == 0:
        parser.error("--blur-kernel 必须是正奇数")
    return args


def unwrap_env(env: gym.Env) -> gym.Env:
    return env.unwrapped


def camera_keys(observation: dict[str, Any]) -> list[str]:
    return sorted(
        key
        for key in observation
        if key.startswith("observation.images.") and "depth" not in key.lower()
    )


def resolve_camera_selection(all_keys: list[str], requested: list[str] | None) -> list[str]:
    if not requested:
        return all_keys
    resolved: list[str] = []
    for value in requested:
        matches = [
            key
            for key in all_keys
            if value == key or value == key.removeprefix("observation.images.")
        ]
        if not matches:
            raise ValueError(f"未知相机 {value!r}；可选项: {all_keys}")
        if matches[0] not in resolved:
            resolved.append(matches[0])
    return resolved


def ablate_tensor(image: torch.Tensor, mode: str, blur_kernel: int) -> torch.Tensor:
    output = image.detach().clone()
    if mode == "zero":
        return torch.zeros_like(output)

    if output.ndim == 3:
        batched = output.unsqueeze(0)
        restore = lambda value: value.squeeze(0)
    elif output.ndim == 4:
        batched = output
        restore = lambda value: value
    else:
        raise ValueError(f"相机张量必须为 CHW 或 BCHW，实际为 {tuple(output.shape)}")

    if mode == "mean":
        return restore(batched.mean(dim=(-2, -1), keepdim=True).expand_as(batched).clone())

    if mode == "blur":
        original_dtype = batched.dtype
        blurred = F.avg_pool2d(
            batched.float(), kernel_size=blur_kernel, stride=1, padding=blur_kernel // 2
        )
        return restore(blurred.to(original_dtype))

    raise ValueError(f"未知消融模式: {mode}")


def ablate_image(value: Any, mode: str, blur_kernel: int) -> Any:
    if isinstance(value, torch.Tensor):
        return ablate_tensor(value, mode, blur_kernel)

    array = np.asarray(value)
    tensor = torch.from_numpy(array.copy())
    ablated = ablate_tensor(tensor, mode, blur_kernel).cpu().numpy()
    return ablated.astype(array.dtype, copy=False)


def build_conditions(
    observation: dict[str, Any], selected_cameras: list[str], mode: str, blur_kernel: int
) -> dict[str, dict[str, Any]]:
    conditions = {"baseline": dict(observation)}
    for key in selected_cameras:
        condition_name = f"ablate_{key.removeprefix('observation.images.')}"
        modified = dict(observation)
        modified[key] = ablate_image(observation[key], mode, blur_kernel)
        conditions[condition_name] = modified
    return conditions


def predict(policy: PolicyClient, observation: dict[str, Any]) -> np.ndarray:
    # This resets inference-server state only; it does not call a robot API.
    policy.reset()
    with torch.inference_mode():
        return action_to_numpy(policy.select_action_chunk(observation))


def effect_summary(condition_actions: dict[str, np.ndarray]) -> dict[str, Any]:
    baseline = condition_actions["baseline"]
    baseline_mean = baseline.mean(axis=0)
    baseline_noise = baseline.std(axis=0).mean(axis=0)
    result: dict[str, Any] = {
        "definition": "effect=|mean(ablation)-mean(baseline)|; noise=mean per-step sample std",
        "baseline_noise_per_dimension": baseline_noise.tolist(),
        "ablations": {},
    }

    for name, actions in condition_actions.items():
        if name == "baseline":
            continue
        ablated_mean = actions.mean(axis=0)
        effect = np.abs(ablated_mean - baseline_mean).mean(axis=0)
        ablated_noise = actions.std(axis=0).mean(axis=0)
        pooled_noise = (baseline_noise + ablated_noise) / 2
        signal_to_noise = effect / np.maximum(pooled_noise, 1e-9)
        entry: dict[str, Any] = {
            "effect_per_dimension": effect.tolist(),
            "pooled_noise_per_dimension": pooled_noise.tolist(),
            "effect_to_noise_per_dimension": signal_to_noise.tolist(),
            "mean_effect": float(effect.mean()),
            "mean_effect_to_noise": float(signal_to_noise.mean()),
        }
        if effect.shape[0] == 16:
            entry["groups"] = {
                "left_arm_0_6": {
                    "mean_effect": float(effect[:7].mean()),
                    "mean_effect_to_noise": float(signal_to_noise[:7].mean()),
                },
                "left_effector_7": {
                    "mean_effect": float(effect[7]),
                    "mean_effect_to_noise": float(signal_to_noise[7]),
                },
                "right_arm_8_14": {
                    "mean_effect": float(effect[8:15].mean()),
                    "mean_effect_to_noise": float(signal_to_noise[8:15].mean()),
                },
                "right_effector_15": {
                    "mean_effect": float(effect[15]),
                    "mean_effect_to_noise": float(signal_to_noise[15]),
                },
            }
        result["ablations"][name] = entry
    return result


def main() -> int:
    args = parse_args()
    config = load_kuavo_config(args.config)
    if config.env.env_name != "Kuavo-Real":
        raise ValueError(f"只允许真机配置，当前 env_name={config.env.env_name!r}")

    prompt = args.prompt if args.prompt is not None else config.inference.task_prompt
    if not str(prompt).strip():
        raise ValueError("任务 prompt 为空")

    print("\n=== 真机三路相机消融 / OBSERVE ONLY ===")
    print("只保存模型预测，绝不调用 env.reset()、env.step() 或执行动作。")
    print(f"config: {args.config}")
    print(f"prompt: {prompt}")
    print(f"ablation mode: {args.mode}")
    if not args.yes:
        confirmation = input("确认机器人安全静止后，输入 OBSERVE 继续: ").strip()
        if confirmation != "OBSERVE":
            print("未确认，退出。")
            return 2

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    rospy.init_node("kuavo_real_camera_ablation", anonymous=True, disable_signals=True)
    env = None
    try:
        print("正在初始化观测环境并等待 ROS buffer……")
        env = unwrap_env(
            gym.make(
                config.env.env_name,
                max_episode_steps=config.inference.max_episode_steps,
                config=config,
            )
        )
        policy = PolicyClient(host=args.host, port=args.port, task_prompt=str(prompt))
        if not policy.policy.ping():
            raise ConnectionError(f"无法连接推理服务 tcp://{args.host}:{args.port}")

        input("摆好目标并保持机器人、目标和环境不动，然后按 Enter 冻结一帧观测: ")
        observation = env.get_obs()
        all_cameras = camera_keys(observation)
        selected_cameras = resolve_camera_selection(all_cameras, args.cameras)
        if not selected_cameras:
            raise RuntimeError(f"没有发现 RGB 相机；观测 keys={list(observation)}")
        conditions = build_conditions(
            observation, selected_cameras, args.mode, args.blur_kernel
        )

        metadata = {
            "created_at": dt.datetime.now().isoformat(),
            "config": str(args.config.resolve()),
            "prompt": str(prompt),
            "samples_per_condition": args.samples_per_condition,
            "ablation_mode": args.mode,
            "blur_kernel": args.blur_kernel if args.mode == "blur" else None,
            "camera_keys": all_cameras,
            "ablated_camera_keys": selected_cameras,
            "condition_order": list(conditions),
            "server": {"host": args.host, "port": args.port},
            "safety": "frozen observation; inference only; no action execution",
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        for condition_name, condition_obs in conditions.items():
            save_observation(condition_obs, output_dir / condition_name / "observation", True)

        collected: dict[str, list[np.ndarray]] = {name: [] for name in conditions}
        timing: dict[str, list[float]] = {name: [] for name in conditions}
        condition_names = list(conditions)

        # Interleave conditions so server warmup or temporal drift cannot favor one group.
        for sample_index in range(args.samples_per_condition):
            print(f"\nround {sample_index + 1}/{args.samples_per_condition}")
            for condition_name in condition_names:
                started = time.perf_counter()
                actions = predict(policy, conditions[condition_name])
                elapsed = time.perf_counter() - started
                collected[condition_name].append(actions)
                timing[condition_name].append(elapsed)
                sample_dir = output_dir / condition_name / f"sample_{sample_index:03d}"
                sample_dir.mkdir(parents=True, exist_ok=False)
                np.save(sample_dir / "actions.npy", actions)
                print(
                    f"  {condition_name:<26} chunk={actions.shape} "
                    f"infer={elapsed:.3f}s（未执行）"
                )
                if args.interval:
                    time.sleep(args.interval)

        condition_actions: dict[str, np.ndarray] = {}
        for condition_name, chunks in collected.items():
            shapes = {chunk.shape for chunk in chunks}
            if len(shapes) != 1:
                raise ValueError(f"{condition_name} 的 chunk shape 不一致: {shapes}")
            stacked = np.stack(chunks, axis=0)
            condition_actions[condition_name] = stacked
            np.save(output_dir / condition_name / "actions_all.npy", stacked)

        summarize_conditions(condition_actions, output_dir)
        effects = effect_summary(condition_actions)
        effects["mean_inference_seconds"] = {
            name: float(np.mean(values)) for name, values in timing.items()
        }
        (output_dir / "ablation_effects.json").write_text(
            json.dumps(effects, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        save_plot(condition_actions, output_dir)

        print(f"\n完成，所有动作均未执行。结果: {output_dir.resolve()}")
        for name, entry in effects["ablations"].items():
            print(
                f"  {name}: mean_effect={entry['mean_effect']:.6f}, "
                f"effect/noise={entry['mean_effect_to_noise']:.3f}"
            )
            groups = entry.get("groups", {})
            if groups:
                left = groups["left_arm_0_6"]
                print(
                    f"    left_arm effect={left['mean_effect']:.6f}, "
                    f"effect/noise={left['mean_effect_to_noise']:.3f}"
                )
        return 0
    except KeyboardInterrupt:
        print("\n用户中断；没有动作被执行。")
        return 130
    finally:
        if env is not None:
            env.close()
        if rospy.core.is_initialized():
            rospy.signal_shutdown("camera ablation finished")


if __name__ == "__main__":
    sys.exit(main())
