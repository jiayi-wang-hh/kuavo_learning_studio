#!/usr/bin/env python3
"""Observe-only contrast evaluation on a real Kuavo robot.

This script reads live robot observations and asks the inference server for
action chunks.  It deliberately never calls ``env.reset()``, ``env.step()``,
or a robot-control API, so predicted actions are recorded but not executed.

Typical use:

    python kuavo_deploy/src/eval/real_observation_contrast_test.py \
      --config configs/deploy/deploy_lingbot_package.yaml \
      --conditions face_up face_down \
      --samples-per-condition 10
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import rospy
import torch

# Importing the package registers Kuavo-Real with Gymnasium.
import kuavo_deploy.kuavo_env  # noqa: F401
from kuavo_deploy.config import load_kuavo_config
from kuavo_deploy.kuavo_service.client import PolicyClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "真机只观测对照测试：采集相机/关节状态并请求动作预测，"
            "但绝不向机器人执行预测动作。"
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="任意真机 deploy YAML")
    parser.add_argument(
        "--conditions",
        nargs="+",
        required=True,
        help="要比较的条件名称，例如 face_up face_down",
    )
    parser.add_argument("--samples-per-condition", type=int, default=10)
    parser.add_argument(
        "--live-each-sample",
        action="store_true",
        help="每次推理重新采集观测；默认每个条件冻结一帧后重复推理",
    )
    parser.add_argument("--interval", type=float, default=0.2, help="重复推理间隔（秒）")
    parser.add_argument("--prompt", default=None, help="覆盖 deploy YAML 中的 task_prompt")
    parser.add_argument("--host", default="localhost", help="kuavo_server 地址")
    parser.add_argument("--port", type=int, default=5555, help="kuavo_server 端口")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/real_observation_contrast"),
    )
    parser.add_argument("--no-images", action="store_true", help="不保存观测图像")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过启动时的 observe-only 人工确认",
    )
    args = parser.parse_args()

    if args.samples_per_condition <= 0:
        parser.error("--samples-per-condition 必须大于 0")
    if args.interval < 0:
        parser.error("--interval 不能小于 0")
    if len(set(args.conditions)) != len(args.conditions):
        parser.error("--conditions 中不能有重复名称")
    return args


def safe_name(value: str) -> str:
    name = re.sub(r"[^0-9A-Za-z_.-]+", "_", value.strip()).strip("._")
    return name or "condition"


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def action_to_numpy(value: Any) -> np.ndarray:
    actions = to_numpy(value)
    while actions.ndim > 2 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2:
        raise ValueError(f"动作必须为 [T, D]，实际得到 {actions.shape}")
    return actions.astype(np.float32, copy=False)


def image_to_hwc_uint8(value: Any) -> np.ndarray | None:
    image = to_numpy(value)
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        return None
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        if image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(image, 0, 255).astype(np.uint8)


def save_observation(
    observation: dict[str, Any], output_dir: Path, save_images: bool
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {}

    for key, value in observation.items():
        if key == "prompt":
            manifest[key] = str(value)
            continue
        try:
            array = to_numpy(value)
        except Exception:
            manifest[key] = {"python_type": type(value).__name__}
            continue

        manifest[key] = {"shape": list(array.shape), "dtype": str(array.dtype)}
        key_name = safe_name(key)
        if "image" in key.lower() and save_images:
            image = image_to_hwc_uint8(array)
            if image is not None:
                try:
                    import imageio.v3 as iio

                    iio.imwrite(output_dir / f"{key_name}.png", image)
                    continue
                except Exception as exc:
                    manifest[key]["png_error"] = str(exc)
        np.save(output_dir / f"{key_name}.npy", array)

    (output_dir / "observation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def predict_fresh_chunk(policy: PolicyClient, observation: dict[str, Any]) -> np.ndarray:
    # Reset only inference-server state. This does not call any robot API.
    policy.reset()
    with torch.inference_mode():
        return action_to_numpy(policy.select_action_chunk(observation))


def summarize_conditions(
    condition_actions: dict[str, np.ndarray], output_dir: Path
) -> dict[str, Any]:
    summary: dict[str, Any] = {"conditions": {}, "comparisons": {}}

    for name, actions in condition_actions.items():
        mean_chunk = actions.mean(axis=0)
        std_chunk = actions.std(axis=0)
        np.save(output_dir / f"{safe_name(name)}_action_mean.npy", mean_chunk)
        np.save(output_dir / f"{safe_name(name)}_action_std.npy", std_chunk)
        summary["conditions"][name] = {
            "samples": int(actions.shape[0]),
            "chunk_shape": list(actions.shape[1:]),
            "within_condition_mean_std": float(std_chunk.mean()),
            "per_dimension_mean": mean_chunk.mean(axis=0).tolist(),
            "per_dimension_std": actions.std(axis=(0, 1)).tolist(),
        }

    names = list(condition_actions)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left = condition_actions[left_name].mean(axis=0)
            right = condition_actions[right_name].mean(axis=0)
            if left.shape != right.shape:
                summary["comparisons"][f"{left_name}__vs__{right_name}"] = {
                    "error": f"chunk shape 不一致: {left.shape} vs {right.shape}"
                }
                continue
            difference = np.abs(left - right)
            summary["comparisons"][f"{left_name}__vs__{right_name}"] = {
                "mean_absolute_difference": float(difference.mean()),
                "root_mean_square_difference": float(np.sqrt(np.mean((left - right) ** 2))),
                "per_dimension_mean_absolute_difference": difference.mean(axis=0).tolist(),
                "per_dimension_max_absolute_difference": difference.max(axis=0).tolist(),
            }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def save_plot(condition_actions: dict[str, np.ndarray], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib 不可用，跳过曲线图: {exc}")
        return

    shapes = {actions.shape[1:] for actions in condition_actions.values()}
    if len(shapes) != 1:
        print("[WARN] 不同条件的 chunk shape 不一致，跳过曲线图")
        return

    horizon, action_dim = next(iter(shapes))
    fig, axes = plt.subplots(
        action_dim, 1, figsize=(12, max(3, 2.4 * action_dim)), squeeze=False
    )
    x_axis = np.arange(horizon)
    for dim in range(action_dim):
        axis = axes[dim, 0]
        for name, actions in condition_actions.items():
            mean = actions[:, :, dim].mean(axis=0)
            std = actions[:, :, dim].std(axis=0)
            axis.plot(x_axis, mean, label=name)
            axis.fill_between(x_axis, mean - std, mean + std, alpha=0.18)
        axis.set_ylabel(f"action {dim}")
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("predicted step")
    axes[0, 0].legend()
    fig.suptitle("Real-observation contrast (predictions are NOT executed)")
    fig.tight_layout()
    fig.savefig(output_dir / "action_contrast.png", dpi=150)
    plt.close(fig)


def unwrap_env(env: gym.Env) -> gym.Env:
    return env.unwrapped


def main() -> int:
    args = parse_args()
    config = load_kuavo_config(args.config)
    if config.env.env_name != "Kuavo-Real":
        raise ValueError(
            f"该脚本只允许真机只观测模式，配置解析出的 env_name={config.env.env_name!r}"
        )

    prompt = args.prompt if args.prompt is not None else config.inference.task_prompt
    if not str(prompt).strip():
        raise ValueError("任务 prompt 为空；请在 YAML 中设置 task_prompt 或传入 --prompt")

    print("\n=== 真机无动作对照测试 / OBSERVE ONLY ===")
    print("安全约束：脚本不会调用 env.reset()、env.step() 或机器人控制接口。")
    print("预测动作只保存到磁盘，不会发送给机器人。")
    print(f"config: {args.config}")
    print(f"prompt: {prompt}")
    print(f"conditions: {args.conditions}\n")
    if not args.yes:
        confirmation = input("确认机器人处于安全状态后，输入 OBSERVE 继续: ").strip()
        if confirmation != "OBSERVE":
            print("未确认，退出。")
            return 2

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created_at": dt.datetime.now().isoformat(),
        "config": str(args.config.resolve()),
        "prompt": str(prompt),
        "conditions": args.conditions,
        "samples_per_condition": args.samples_per_condition,
        "frozen_observation_per_condition": not args.live_each_sample,
        "server": {"host": args.host, "port": args.port},
        "safety": "observe_only; no env.reset; no env.step; no action execution",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rospy.init_node("kuavo_real_observation_contrast", anonymous=True, disable_signals=True)
    env = None
    try:
        print("正在初始化只读观测环境并等待 ROS 观测缓存……")
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

        condition_actions: dict[str, np.ndarray] = {}
        for condition in args.conditions:
            print(f"\n请布置条件 [{condition}]，保持机器人静止。")
            input("布置完成后按 Enter 采集观测并开始只推理: ")
            condition_dir = output_dir / safe_name(condition)
            condition_dir.mkdir(parents=True, exist_ok=False)

            frozen_observation = None
            actions_for_condition: list[np.ndarray] = []
            for sample_index in range(args.samples_per_condition):
                if frozen_observation is None or args.live_each_sample:
                    observation = env.get_obs()
                    if frozen_observation is None and not args.live_each_sample:
                        frozen_observation = observation
                else:
                    observation = frozen_observation

                sample_dir = condition_dir / f"sample_{sample_index:03d}"
                sample_dir.mkdir(parents=True, exist_ok=False)
                if sample_index == 0 or args.live_each_sample:
                    save_observation(observation, sample_dir, not args.no_images)

                started = time.perf_counter()
                actions = predict_fresh_chunk(policy, observation)
                elapsed = time.perf_counter() - started
                np.save(sample_dir / "actions.npy", actions)
                (sample_dir / "inference.json").write_text(
                    json.dumps(
                        {
                            "condition": condition,
                            "sample_index": sample_index,
                            "elapsed_seconds": elapsed,
                            "action_shape": list(actions.shape),
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                actions_for_condition.append(actions)
                print(
                    f"  [{sample_index + 1}/{args.samples_per_condition}] "
                    f"chunk={actions.shape}, infer={elapsed:.3f}s（未执行）"
                )
                if args.interval:
                    time.sleep(args.interval)

            shapes = {actions.shape for actions in actions_for_condition}
            if len(shapes) != 1:
                raise ValueError(f"条件 {condition!r} 内 action chunk shape 不一致: {shapes}")
            stacked = np.stack(actions_for_condition, axis=0)
            np.save(condition_dir / "actions_all.npy", stacked)
            condition_actions[condition] = stacked

        summary = summarize_conditions(condition_actions, output_dir)
        save_plot(condition_actions, output_dir)
        print(f"\n完成。所有预测均未执行。结果目录: {output_dir.resolve()}")
        for name, result in summary["comparisons"].items():
            if "mean_absolute_difference" in result:
                print(
                    f"  {name}: mean_abs_diff={result['mean_absolute_difference']:.6f}, "
                    f"rms_diff={result['root_mean_square_difference']:.6f}"
                )
        return 0
    except KeyboardInterrupt:
        print("\n用户中断。没有预测动作被执行。")
        return 130
    finally:
        if env is not None:
            env.close()
        if rospy.core.is_initialized():
            rospy.signal_shutdown("observe-only contrast test finished")


if __name__ == "__main__":
    sys.exit(main())
