"""Portable resume contracts and completion markers for fully saved checkpoints."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil

from .config import Config
from .dataset import load_split, pixel_digest

class CheckpointError(ValueError):
    """Checkpoint cannot satisfy the requested operation."""

def guard_restored_state(state, optimizer, scheduler, expected):
    """Check the actual objects after Trainer has loaded their state dictionaries."""
    if state.global_step != expected["global_step"] or not optimizer.state:
        raise RuntimeError("Resume did not restore global step/optimizer; refusing to restart")
    if scheduler.last_epoch != state.global_step:
        raise RuntimeError("Resume scheduler step does not match Trainer step")
    state.best_model_checkpoint = expected["best_model_checkpoint"]

def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)

def read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CheckpointError(f"Cannot read checkpoint metadata: {path.name}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"Checkpoint metadata must be an object: {path.name}")
    return value

def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()

def versions() -> dict:
    result = {"python": platform.python_version()}
    for package in ("torch", "transformers", "trl", "unsloth", "unsloth_zoo", "peft", "bitsandbytes", "accelerate", "Pillow", "jiwer"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result

def manifest(config: Config, fingerprints: dict[str, str] | None = None) -> dict:
    # Ordered image pixels and raw labels are invariant to relocating Kaggle paths.
    if fingerprints is None:
        fingerprints = {}
        for split in ("train", "validation"):
            digest = hashlib.sha256()
            for sample in load_split(config.data_dir, split):
                payload = [pixel_digest(sample.image)[0], sample.text, sample.source_id, sample.proper_names]
                digest.update(json.dumps(payload, ensure_ascii=False).encode())
                digest.update(b"\n")
            fingerprints[split] = digest.hexdigest()
    else:
        fingerprints = {split: fingerprints[split] for split in ("train", "validation")}
    return {"schema": 2, "model_id": config.model_id, "model": config.section("model"),
            "lora": config.section("lora"), "training": config.section("training"),
            "data": config.section("data"), "prompt": config.section("inference")["prompt"],
            "evaluation": config.section("evaluation"), "inference": config.section("inference"),
            "fingerprints": fingerprints, "versions": versions()}

def write_manifest(folder: Path, config: Config, contract: dict | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "htr_manifest.json", contract or manifest(config))

def validate_adapter(path: Path, config: Config) -> dict:
    from .models import require_supported
    require_supported(config)
    try:
        adapter = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"Invalid adapter_config.json in {path}") from exc
    if not isinstance(adapter, dict):
        raise CheckpointError("adapter_config.json must be an object")
    if adapter.get("base_model_name_or_path") != config.model_id:
        raise CheckpointError("Adapter requires exactly the configured base model ID/quantization; cross-size or alternative bases are forbidden")
    lora = config.section("lora")
    if adapter.get("r") != lora["rank"] or adapter.get("lora_alpha") != lora["alpha"] or adapter.get("lora_dropout", 0) != lora["dropout"]:
        raise CheckpointError("Adapter LoRA rank/alpha/dropout differs from config")
    if adapter.get("bias", "none") != "none" or adapter.get("peft_type", "LORA") != "LORA":
        raise CheckpointError("Only bias-free LoRA adapters are supported")
    if lora["target_modules"] and set(adapter.get("target_modules", [])) != set(lora["target_modules"]):
        raise CheckpointError("Adapter target_modules differ from configured targets")
    weights = [path / "adapter_model.safetensors"]
    if not all(p.is_file() and p.stat().st_size > 0 for p in weights):
        raise CheckpointError("Missing or empty adapter_model.safetensors")
    return adapter

def seal_checkpoint(path: Path) -> None:
    required = ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth", "htr_manifest.json")
    missing = [name for name in required if not (path / name).is_file() or not (path / name).stat().st_size]
    if missing:
        raise CheckpointError(f"Incomplete checkpoint: {', '.join(missing)}")
    # Marker is written last and hashes all saved files, including scaler and processor.
    hashes = {p.name: file_digest(p) for p in sorted(path.iterdir()) if p.is_file() and p.name != "COMPLETE.json" and not p.name.endswith(".tmp")}
    atomic_json(path / "COMPLETE.json", {"schema": 1, "sha256": hashes})

def validate_resume(path: Path, config: Config, current: dict | None = None) -> dict:
    validate_adapter(path, config)
    try:
        if not (path / "COMPLETE.json").is_file():
            raise CheckpointError("Exact resume impossible: checkpoint has no COMPLETE.json; use explicit warm start for an adapter-only source")
        marker = read_object(path / "COMPLETE.json")
        if marker.get("schema") != 1:
            raise CheckpointError("Unsupported checkpoint completion schema")
        hashes = marker["sha256"]
        required = {"adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth", "htr_manifest.json"}
        if not isinstance(hashes, dict) or not required <= set(hashes):
            raise CheckpointError("Incomplete checkpoint inventory")
        for name, digest in hashes.items():
            if Path(name).name != name or file_digest(path / name) != digest:
                raise CheckpointError(f"Checkpoint integrity failed: {name}")
        saved = read_object(path / "htr_manifest.json")
        state = read_object(path / "trainer_state.json")
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise CheckpointError("Exact resume impossible: incomplete/unsealed checkpoint. Explicit warm start is available for adapter-only checkpoints") from exc
    current = current or manifest(config)
    runtime = saved.get("runtime", {})
    if not isinstance(runtime, dict):
        raise CheckpointError("Invalid checkpoint runtime contract")
    if runtime.get("bf16") is False and "scaler.pt" not in hashes:
        raise CheckpointError("FP16 resume requires saved AMP scaler.pt")
    for key, value in current.items():
        if saved.get(key) != value:
            raise CheckpointError(f"Exact resume incompatible: {key} differs")
    step = state.get("global_step")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise CheckpointError("Invalid trainer global_step")
    if path.name.startswith("checkpoint-") and path.name != f"checkpoint-{step}":
        raise CheckpointError("Checkpoint folder and Trainer global_step disagree")
    best = state.get("best_model_checkpoint")
    if best is not None and not isinstance(best, str):
        raise CheckpointError("Invalid best_model_checkpoint path")
    best_path = path.parent / Path(best.replace("\\", "/")).name if best else None
    if best_path and not (best_path / "adapter_model.safetensors").is_file():
        raise CheckpointError(f"Best checkpoint missing after relocation: {best_path}; copy it alongside the resume checkpoint")
    if best_path and best_path.resolve() != path.resolve():
        try:
            best_marker = read_object(best_path / "COMPLETE.json")
            for name in ("adapter_model.safetensors", "adapter_config.json"):
                if file_digest(best_path / name) != best_marker["sha256"][name]:
                    raise CheckpointError("Best checkpoint adapter integrity failed")
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise CheckpointError("Best checkpoint is incomplete/unsealed") from exc
    return {"global_step": step, "best_model_checkpoint": str(best_path) if best_path else None, "contract": saved}


def latest_valid_checkpoint(root: Path, config: Config, current: dict) -> tuple[Path, dict]:
    """Choose the most recently sealed compatible checkpoint, skipping interrupted saves."""
    candidates = [p for run in root.iterdir() if run.is_dir() for p in run.iterdir()
                  if p.is_dir() and not p.is_symlink() and re.fullmatch(r"checkpoint-[1-9][0-9]*", p.name)
                  and (p / "COMPLETE.json").is_file()] if root.is_dir() else []
    candidates.sort(key=lambda p: ((p / "COMPLETE.json").stat().st_mtime_ns, p.name), reverse=True)
    for path in candidates:
        try:
            return path, validate_resume(path, config, current)
        except (CheckpointError, OSError):
            continue
    raise CheckpointError(f"No valid compatible checkpoint under {root}; refusing to restart at step 0")


def prune_completed_checkpoints(run_dir: Path, keep: int, best: str | None = None) -> list[Path]:
    """Prune only sealed older checkpoints after the new checkpoint has been sealed."""
    if keep < 1:
        raise ValueError("keep must be positive")
    if not run_dir.is_dir():
        return []
    completed = [p for p in run_dir.iterdir() if p.is_dir() and not p.is_symlink()
                 and re.fullmatch(r"checkpoint-[1-9][0-9]*", p.name) and (p / "COMPLETE.json").is_file()]
    completed.sort(key=lambda p: int(p.name.split("-")[1]), reverse=True)
    protected = {p.resolve() for p in completed[:keep]}
    if best:
        protected.add(Path(best).resolve())
    removed = []
    for path in completed:
        if path.resolve() not in protected and path.resolve().parent == run_dir.resolve():
            shutil.rmtree(path)
            removed.append(path)
    return removed
