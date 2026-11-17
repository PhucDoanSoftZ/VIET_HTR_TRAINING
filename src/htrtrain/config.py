"""Strict YAML configuration with relative paths anchored at the repository root."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any
import yaml

class ConfigError(ValueError):
    pass

SUPPORTED = {
    "qwen3vl_4b": "4B",
    "qwen3vl_8b": "8B",
}

DEFAULTS = {
    "model": {"revision": "main", "gradient_checkpointing": "unsloth"},
    "paths": {"resume_from": None, "warm_start_from_adapter": None, "adapter_dir": None},
    "data": {"source": "5CD-AI/VietHTR-Line", "resize_policy": "error", "min_short_edge": 16},
    "lora": {"dropout": 0.0, "target_modules": None},
    "training": {"weight_decay": 0.01, "max_grad_norm": 1.0, "warmup_ratio": 0.0,
                 "dataloader_workers": 0, "pin_memory": True, "best_metric": "eval_loss", "validation_ocr_limit": 0},
    "inference": {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0},
    "evaluation": {"normalization": "nfc_whitespace", "without_diacritics": False},
    "logging": {"debug": False},
}
FIELDS = {
    "model": {"key", "id"}, "paths": {"data_dir", "checkpoint_dir", "output_dir"},
    "data": {"max_pixels"}, "hardware": {"min_free_vram_gb", "min_free_ram_gb", "min_free_disk_gb"},
    "lora": {"rank", "alpha", "vision_layers", "language_layers", "attention_modules", "mlp_modules"},
    "training": {"batch_size", "gradient_accumulation_steps", "epochs", "learning_rate", "warmup_steps", "optimizer", "scheduler", "save_steps", "eval_steps", "logging_steps", "save_total_limit", "max_seq_length", "seed", "early_stopping_patience"},
    "inference": {"prompt", "max_new_tokens"}, "evaluation": set(), "logging": set(),
}
for _section, _defaults in DEFAULTS.items():
    FIELDS[_section] |= set(_defaults)

@dataclass(frozen=True)
class Config:
    path: Path
    raw: dict[str, Any]
    model_key: str
    model_id: str
    data_dir: Path
    checkpoint_dir: Path
    output_dir: Path

    def section(self, key: str) -> dict[str, Any]:
        return self.raw[key]

    def optional_path(self, key: str) -> Path | None:
        value = self.raw["paths"].get(key)
        if value is None:
            return None
        base = self.path.parent.parent if self.path.parent.name == "configs" else self.path.parent
        return _root_path(value, base)

def _root_path(value: str, base: Path) -> Path:
    p = Path(value).expanduser()
    return (p if p.is_absolute() else base / p).resolve()

def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigError(f"Config not found: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Config must be a YAML mapping")
    if set(raw) - set(FIELDS):
        raise ConfigError("Unknown configuration section")
    for section, fields in FIELDS.items():
        supplied = raw.get(section, {})
        if not isinstance(supplied, dict) or set(supplied) - fields:
            raise ConfigError(f"Unknown field or invalid mapping in {section}")
        raw[section] = {**DEFAULTS.get(section, {}), **supplied}
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"Override must be key=value: {item}")
        key, value = item.split("=", 1)
        parts = key.split(".")
        cursor = raw
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                raise ConfigError(f"Unknown override: {key}")
            cursor = cursor[part]
        if parts[-1] not in cursor:
            raise ConfigError(f"Unknown override: {key}")
        try:
            cursor[parts[-1]] = yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid override: {key}") from exc
    for section, fields in FIELDS.items():
        missing = fields - set(raw[section])
        if missing:
            raise ConfigError(f"Missing {section} fields: {', '.join(sorted(missing))}")
    for name in ("model", "data", "paths", "training", "lora", "inference", "hardware"):
        if not isinstance(raw.get(name), dict):
            raise ConfigError(f"Missing mapping: {name}")
    model = raw["model"]
    key = model.get("key")
    model_id = model.get("id")
    if not isinstance(key, str) or not isinstance(model_id, str) or not model_id:
        raise ConfigError("model.key and model.id are required strings")
    import re
    if key in SUPPORTED:
        size = SUPPORTED[key]
        if not re.fullmatch(rf"(?:Qwen|unsloth)/Qwen3-VL-{size}-Instruct(?:-(?:unsloth-)?bnb-4bit)?", model_id):
            raise ConfigError(f"model.id does not match {key} Instruct architecture/size")
    training, lora, inf, hw = (raw[x] for x in ("training", "lora", "inference", "hardware"))
    for name in ("batch_size", "gradient_accumulation_steps", "epochs", "save_steps", "eval_steps", "logging_steps", "max_seq_length"):
        value = training.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"training.{name} must be a positive integer")
    if not isinstance(training.get("learning_rate"), (int, float)) or training["learning_rate"] <= 0:
        raise ConfigError("training.learning_rate must be positive")
    for name in ("seed", "warmup_steps", "save_total_limit", "early_stopping_patience"):
        value = training.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < (0 if name != "save_total_limit" else 1):
            raise ConfigError(f"training.{name} has an invalid value")
    if training["save_steps"] % training["eval_steps"] != 0:
        raise ConfigError("training.save_steps must be a multiple of training.eval_steps for best adapter selection")
    if training.get("optimizer") not in ("adamw_8bit", "adamw_torch") or training.get("scheduler") not in ("linear", "cosine"):
        raise ConfigError("Unsupported optimizer/scheduler")
    for name in ("rank", "alpha"):
        if not isinstance(lora.get(name), int) or isinstance(lora[name], bool) or lora[name] <= 0:
            raise ConfigError(f"lora.{name} must be a positive integer")
    for name in ("vision_layers", "language_layers", "attention_modules", "mlp_modules"):
        if not isinstance(lora.get(name), bool):
            raise ConfigError(f"lora.{name} must be boolean")
    if not isinstance(inf.get("prompt"), str) or not inf["prompt"].strip():
        raise ConfigError("inference.prompt is required")
    if not isinstance(inf.get("max_new_tokens"), int) or isinstance(inf["max_new_tokens"], bool) or inf["max_new_tokens"] <= 0:
        raise ConfigError("inference.max_new_tokens must be positive")
    pixels = raw["data"].get("max_pixels")
    if not isinstance(pixels, int) or isinstance(pixels, bool) or pixels <= 0:
        raise ConfigError("data.max_pixels must be a positive integer")
    for name in ("min_free_vram_gb", "min_free_ram_gb", "min_free_disk_gb"):
        if not isinstance(hw.get(name), (int, float)) or isinstance(hw[name], bool) or not math.isfinite(hw[name]) or hw[name] <= 0:
            raise ConfigError(f"hardware.{name} must be positive")
    paths = raw["paths"]
    for name in ("data_dir", "checkpoint_dir", "output_dir"):
        if not isinstance(paths.get(name), str) or not paths[name]:
            raise ConfigError(f"paths.{name} is required")
    for name in ("resume_from", "warm_start_from_adapter", "adapter_dir"):
        if paths.get(name) is not None and (not isinstance(paths[name], str) or not paths[name]):
            raise ConfigError(f"paths.{name} must be a path or null")
    if paths["resume_from"] and paths["warm_start_from_adapter"]:
        raise ConfigError("Choose resume_from or warm_start_from_adapter")
    for section, names in {"training": ("weight_decay", "warmup_ratio", "max_grad_norm"), "lora": ("dropout",), "inference": ("repetition_penalty",)}.items():
        for name in names:
            value = raw[section][name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ConfigError(f"Invalid {section}.{name}")
    for section, names in {"training": ("dataloader_workers", "validation_ocr_limit"), "data": ("min_short_edge",), "inference": ("num_beams",)}.items():
        for name in names:
            value = raw[section][name]
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if section == "training" else 1):
                raise ConfigError(f"Invalid {section}.{name}")
    for section, names in {"training": ("pin_memory",), "inference": ("do_sample",), "evaluation": ("without_diacritics",), "logging": ("debug",)}.items():
        for name in names:
            if not isinstance(raw[section][name], bool):
                raise ConfigError(f"{section}.{name} must be boolean")
    if not math.isfinite(training["learning_rate"]) or isinstance(training["learning_rate"], bool):
        raise ConfigError("Invalid learning_rate")
    if training["warmup_ratio"] > 1 or (training["warmup_steps"] and training["warmup_ratio"]):
        raise ConfigError("Choose warmup_steps or warmup_ratio in [0,1]")
    if not training["max_grad_norm"] or lora["dropout"] > 0.5:
        raise ConfigError("Invalid gradient clipping or dropout")
    if not (lora["vision_layers"] or lora["language_layers"]) or not (lora["attention_modules"] or lora["mlp_modules"]):
        raise ConfigError("LoRA selects no trainable layer/module group")
    targets = lora["target_modules"]
    if targets is not None and (not isinstance(targets, list) or not targets or not all(isinstance(x, str) and x for x in targets)):
        raise ConfigError("target_modules must be null or a nonempty list")
    if raw["data"]["resize_policy"] not in ("error", "downscale"):
        raise ConfigError("resize_policy must be error or downscale")
    if raw["evaluation"]["normalization"] not in ("none", "nfc_whitespace") or inf["do_sample"]:
        raise ConfigError("Unsupported normalization or nondeterministic decoding")
    if training["best_metric"] not in ("eval_loss", "eval_cer", "eval_wer") or (training["best_metric"] != "eval_loss" and not training["validation_ocr_limit"]):
        raise ConfigError("best_metric requires enabled validation OCR")
    if not isinstance(model["revision"], str) or not model["revision"] or not (isinstance(model["gradient_checkpointing"], bool) or model["gradient_checkpointing"] == "unsloth"):
        raise ConfigError("Invalid model revision/gradient_checkpointing")
    if not isinstance(raw["data"]["source"], str) or not raw["data"]["source"] or not inf["repetition_penalty"]:
        raise ConfigError("Invalid dataset source or repetition_penalty")
    if raw["logging"]["debug"] and training["dataloader_workers"]:
        raise ConfigError("Debug batch tracing requires dataloader_workers=0")
    # Configs live in <repo>/configs; paths stay portable across Windows and Linux.
    base = source.parent.parent if source.parent.name == "configs" else source.parent
    resolved = [_root_path(paths[name], base) for name in ("data_dir", "checkpoint_dir", "output_dir")]
    if len(set(resolved)) != 3 or any(p.is_relative_to(resolved[0]) for p in resolved[1:]):
        raise ConfigError("Checkpoint/output directories must be distinct and outside dataset root")
    return Config(source, raw, key, model_id,
                  _root_path(paths["data_dir"], base),
                  _root_path(paths["checkpoint_dir"], base),
                  _root_path(paths["output_dir"], base))
