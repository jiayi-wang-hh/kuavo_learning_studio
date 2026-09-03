#!/usr/bin/env python3
"""Compare Qwen2.5-VL, RoboBrain2.0 and RoboBrain2.5 scene reasoning."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor


# ============================================================
# Prompt
# ============================================================

PROMPT = """You are a robot manipulation reasoning assistant.

The robot's task instruction: pick the red apple from the cluttered scene.

Observe the image and answer the following questions:

1. What objects are visible?
2. What is the target object?
3. What is the robot currently doing?
4. What stage of the task has been completed?
5. Is there any failure or abnormal situation?
6. What should the robot do next?

Answer concisely.

Use exactly these section labels:
Objects:
Target:
Current action:
Task progress:
Failure:
Next step:
"""


# ============================================================
# Model configurations
# ============================================================

MODE_INFO = {
    "qwen": {
        "name": "Qwen2.5-VL-3B-Instruct baseline",
        "default_path": (
            "/media/data/jiayi/hf_model/Qwen2.5-VL-3B-Instruct"
        ),
        "default_log": (
            "/media/data/jiayi/outputs/vlm_test/logs/"
            "qwen25_vl_3b_baseline.log"
        ),
    },

    "robobrain": {
        "name": "RoboBrain2.0-7B",
        "default_path": (
            "/media/data/jiayi/hf_model/RoboBrain2.0-7B"
        ),
        "default_log": (
            "/media/data/jiayi/outputs/vlm_test/logs/"
            "robobrain2_7b.log"
        ),
    },

    "robobrain_2.5": {
        "name": "RoboBrain2.5-8B-NV",
        "default_path": (
            "/media/data/jiayi/hf_model/RoboBrain2.5-8B-NV"
        ),
        "default_log": (
            "/media/data/jiayi/outputs/vlm_test/logs/"
            "robobrain2_5_8b_nv.log"
        ),
    },
}


# ============================================================
# Arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--mode",
        choices=tuple(MODE_INFO),
        required=True,
    )

    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="Override default checkpoint path selected by --mode",
    )

    parser.add_argument(
        "--image_dir",
        type=Path,
        default=Path(
            "/media/data/jiayi/dataset/apple_scene_frames"
        ),
        help="Directory containing PNG images recursively",
    )

    parser.add_argument(
        "--output_log",
        type=Path,
        default=None,
        help="Override mode-specific output log path",
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )

    parser.add_argument(
        "--attn_implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=384,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 uses deterministic greedy decoding",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally test only the first N images",
    )

    parser.add_argument(
        "--robobrain25_repo",
        type=Path,
        default=Path(
            "/media/data/jiayi/RoboBrain2.5"
        ),
        help=(
            "Local clone of the official RoboBrain2.5 repository. "
            "Needed for UnifiedInference."
        ),
    )

    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================

def frame_time(path: Path) -> int:
    """Extract time from names such as frame_1s.png."""

    match = re.search(
        r"frame_(\d+)s",
        path.stem,
    )

    if match is None:
        raise ValueError(
            f"Cannot parse frame time from: {path}"
        )

    return int(match.group(1))


def resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path]:

    mode_info = MODE_INFO[args.mode]

    model_path = (
        args.model_path
        or Path(mode_info["default_path"])
    )

    output_log = (
        args.output_log
        or Path(mode_info["default_log"])
    )

    return model_path, output_log


# ============================================================
# Model loading
# ============================================================

def load_model_and_processor(
    model_path: Path,
    args: argparse.Namespace,
):

    model_name = MODE_INFO[args.mode]["name"]

    # ========================================================
    # RoboBrain 2.5
    #
    # IMPORTANT:
    # Use the official UnifiedInference implementation.
    # Do NOT treat RoboBrain2.5 as Qwen2.5-VL.
    # ========================================================

    if args.mode == "robobrain_2.5":

        repo_path = args.robobrain25_repo.resolve()

        if not repo_path.is_dir():
            raise NotADirectoryError(
                f"RoboBrain2.5 repo not found: {repo_path}"
            )

        repo_string = str(repo_path)

        if repo_string not in sys.path:
            sys.path.insert(
                0,
                repo_string,
            )

        try:
            from inference import UnifiedInference
        except ImportError as exc:
            raise ImportError(
                "Could not import UnifiedInference from "
                f"RoboBrain2.5 repo: {repo_path}"
            ) from exc

        print(
            "\nLoading RoboBrain2.5 using official "
            "UnifiedInference..."
        )

        model = UnifiedInference(
            str(model_path)
        )

        provenance = [
            f"Mode: {args.mode}",
            f"Model type: {model_name}",
            f"Full checkpoint: {model_path.resolve()}",
            f"Official repo: {repo_path}",
            "Inference backend: UnifiedInference",
            "Task mode: general",
            "do_sample: False",
            "LingBot weights loaded: no",
        ]

        print("\n".join(provenance))

        # UnifiedInference owns its processor internally.
        return model, None, provenance

    # ========================================================
    # Standard Transformers modes
    # ========================================================

    dtype = getattr(
        torch,
        args.dtype,
    )

    # ========================================================
    # Qwen2.5-VL
    # ========================================================

    if args.mode == "qwen":
        from transformers import (Qwen2_5_VLForConditionalGeneration)

        print(
            "\nLoading Qwen2.5-VL baseline..."
        )

        processor = AutoProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            use_fast=False,
        )

        model = (
            Qwen2_5_VLForConditionalGeneration
            .from_pretrained(
                str(model_path),
                torch_dtype=dtype,
                attn_implementation=(
                    args.attn_implementation
                ),
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        )

        model.to(
            args.device
        ).eval()

        provenance = [
            f"Mode: {args.mode}",
            f"Model type: {model_name}",
            f"Full checkpoint: {model_path.resolve()}",
            f"Processor source: {model_path.resolve()}",
            f"dtype: {args.dtype}",
            f"attention: {args.attn_implementation}",
            (
                "Inference backend: "
                "Qwen2_5_VLForConditionalGeneration"
            ),
            "LingBot weights loaded: no",
        ]

        print("\n".join(provenance))

        return model, processor, provenance

    # ========================================================
    # RoboBrain 2.0
    # ========================================================

    if args.mode == "robobrain":
        from transformers import (AutoModelForMultimodalLM)

        print(
            "\nLoading RoboBrain2.0..."
        )

        processor = AutoProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            use_fast=False,
        )

        model = AutoModelForMultimodalLM.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            attn_implementation=(
                args.attn_implementation
            ),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        model.to(
            args.device
        ).eval()

        provenance = [
            f"Mode: {args.mode}",
            f"Model type: {model_name}",
            f"Full checkpoint: {model_path.resolve()}",
            f"Processor source: {model_path.resolve()}",
            f"dtype: {args.dtype}",
            f"attention: {args.attn_implementation}",
            (
                "Inference backend: "
                "AutoModelForMultimodalLM"
            ),
            "LingBot weights loaded: no",
        ]

        print("\n".join(provenance))

        return model, processor, provenance

    raise ValueError(
        f"Unsupported mode: {args.mode}"
    )


# ============================================================
# Standard Transformers inference
# Qwen2.5-VL / RoboBrain2.0
# ============================================================

def generate_standard_transformers(
    model,
    processor,
    image_path: Path,
    args: argparse.Namespace,
) -> str:

    image = Image.open(
        image_path
    ).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": PROMPT,
                },
            ],
        }
    ]

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[chat_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        name: tensor.to(args.device)
        for name, tensor in inputs.items()
    }

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }

    if args.temperature > 0:
        generate_kwargs["temperature"] = (
            args.temperature
        )

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            **generate_kwargs,
        )

    input_length = inputs[
        "input_ids"
    ].shape[1]

    new_ids = output_ids[
        :,
        input_length:
    ]

    reasoning = processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return reasoning


# ============================================================
# Unified reasoning interface
# ============================================================

def generate_reasoning(
    model,
    processor,
    image_path: Path,
    args: argparse.Namespace,
) -> str:

    # ========================================================
    # RoboBrain 2.5
    #
    # Official UnifiedInference.
    # For benchmark reproducibility:
    # do_sample=False
    # ========================================================

    if args.mode == "robobrain_2.5":

        result = model.inference(
            PROMPT,
            str(image_path),
            task="general",
            do_sample=False,
        )

        # Official UnifiedInference returns:
        #
        # {
        #     "answer": "..."
        # }

        if isinstance(result, dict):

            if "answer" not in result:
                raise RuntimeError(
                    "RoboBrain2.5 inference returned a dict "
                    "without an 'answer' field:\n"
                    f"{result}"
                )

            return str(
                result["answer"]
            ).strip()

        # Defensive fallback in case the upstream API changes.
        return str(result).strip()

    # ========================================================
    # Qwen2.5-VL / RoboBrain2.0
    # ========================================================

    return generate_standard_transformers(
        model,
        processor,
        image_path,
        args,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    model_path, output_log = resolve_paths(
        args
    )

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not model_path.is_dir():
        raise NotADirectoryError(
            f"Model directory not found: {model_path}"
        )

    if not args.image_dir.is_dir():
        raise NotADirectoryError(
            f"Image directory not found: {args.image_dir}"
        )

    # --------------------------------------------------------
    # Check CUDA
    # --------------------------------------------------------

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA is unavailable; run on the GPU server "
            "or pass --device cpu"
        )

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_paths = sorted(
        args.image_dir.rglob("*.png"),
        key=lambda path: (
            path.parent.as_posix(),
            frame_time(path),
        ),
    )

    if args.limit is not None:

        if args.limit < 1:
            raise ValueError(
                "--limit must be at least 1"
            )

        image_paths = image_paths[
            :args.limit
        ]

    if not image_paths:
        raise RuntimeError(
            f"No PNG images found under: "
            f"{args.image_dir}"
        )

    # --------------------------------------------------------
    # Load model ONCE
    # --------------------------------------------------------

    model, processor, provenance = (
        load_model_and_processor(
            model_path,
            args,
        )
    )

    # --------------------------------------------------------
    # Prepare log
    # --------------------------------------------------------

    output_log.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Clear previous log before this run.
    output_log.write_text(
        "",
        encoding="utf-8",
    )

    model_name = MODE_INFO[
        args.mode
    ]["name"]

    print(
        f"\nFound {len(image_paths)} images."
    )

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print(
            f"\n[{index}/{len(image_paths)}] "
            f"Testing: {image_path}"
        )

        reasoning = generate_reasoning(
            model,
            processor,
            image_path,
            args,
        )

        print(
            f"\n=== {model_name} Reasoning ===\n"
        )

        print(reasoning)

        # ----------------------------------------------------
        # Log result
        # ----------------------------------------------------

        log_text = "\n".join(
            [
                "=" * 80,
                f"Image: {image_path.resolve()}",
                "",
                f"=== {model_name} Provenance ===",
                *provenance,
                "",
                f"=== {model_name} Reasoning ===",
                "",
                reasoning,
                "",
            ]
        )

        with output_log.open(
            "a",
            encoding="utf-8",
        ) as handle:

            handle.write(
                log_text
            )

            handle.write(
                "\n"
            )

    print(
        f"\nAll results saved to: "
        f"{output_log.resolve()}"
    )


if __name__ == "__main__":
    main()