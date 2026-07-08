import json
from pathlib import Path
from shutil import copytree

from huggingface_hub import hf_hub_download
from transformers.dynamic_module_utils import HF_MODULES_CACHE, TRANSFORMERS_DYNAMIC_MODULE_NAME


def _disable_flash_attention_config(config_path: Path) -> None:
    if not config_path.exists():
        return

    with open(config_path) as f:
        config = json.load(f)

    changed = False
    for block in (config, config.get("vision_config", {}), config.get("text_config", {})):
        if block.get("_attn_implementation") == "flash_attention_2":
            block["_attn_implementation"] = "sdpa"
            changed = True

    if changed:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
            f.write("\n")


def ensure_eagle_cache_ready(vendor_dir: Path, cache_dir: Path, assets_repo: str) -> None:
    """Populate the Eagle processor directory in cache and ensure tokenizer assets exist.

    - Copies the vendored Eagle files into cache_dir (overwriting when needed).
    - Downloads vocab.json and merges.txt into the same cache_dir if missing.
    """
    cache_dir = Path(cache_dir)
    vendor_dir = Path(vendor_dir)

    try:
        # Populate/refresh cache with vendor files to ensure a complete processor directory
        print(f"[GROOT] Copying vendor Eagle files to cache: {vendor_dir} -> {cache_dir}")
        copytree(vendor_dir, cache_dir, dirs_exist_ok=True)
        _disable_flash_attention_config(cache_dir / "config.json")

        module_cache_dir = Path(HF_MODULES_CACHE) / TRANSFORMERS_DYNAMIC_MODULE_NAME / cache_dir.name
        if module_cache_dir.exists():
            print(f"[GROOT] Refreshing Eagle dynamic module cache: {module_cache_dir}")
            copytree(vendor_dir, module_cache_dir, dirs_exist_ok=True)
            _disable_flash_attention_config(module_cache_dir / "config.json")
    except Exception as exc:  # nosec: B110
        print(f"[GROOT] Warning: Failed to copy vendor Eagle files to cache: {exc}")

    required_assets = [
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "chat_template.json",
        "special_tokens_map.json",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
    ]

    print(f"[GROOT] Assets repo: {assets_repo} \n Cache dir: {cache_dir}")

    for fname in required_assets:
        dst = cache_dir / fname
        if not dst.exists():
            print(f"[GROOT] Fetching {fname}")
            hf_hub_download(
                repo_id=assets_repo,
                filename=fname,
                repo_type="model",
                local_dir=str(cache_dir),
            )
            if fname == "config.json":
                _disable_flash_attention_config(dst)
