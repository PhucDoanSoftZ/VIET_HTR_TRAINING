"""Read-only hardware preflight, with no model download."""
from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import tempfile
import psutil

from .config import Config

def doctor(config: Config) -> dict:
    report = {"ok": True, "errors": [], "warnings": [], "gpu": [], "packages": {}}
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        report["errors"].append("Distributed Qwen3-VL training is not verified in this repository; DDP replicates the full 32B model per GPU and does not pool VRAM. Launch one process, or use the 8B config on a suitable GPU")
    for package in ("torch", "transformers", "trl", "unsloth", "unsloth_zoo", "peft", "accelerate", "safetensors", "bitsandbytes"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    try:
        import torch
        report["torch_cuda"] = torch.version.cuda
        report["bf16"] = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        if torch.cuda.is_available():
            report["selected_gpu"] = torch.cuda.current_device()
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                report["gpu"].append({"index": i, "name": torch.cuda.get_device_name(i), "free_gb": round(free / 2**30, 2), "total_gb": round(total / 2**30, 2)})
            if torch.cuda.device_count() > 1:
                report["warnings"].append(f"Using CUDA device {report['selected_gpu']} only; other visible GPUs are not pooled. Set CUDA_VISIBLE_DEVICES to choose a different device")
    except ImportError:
        report["torch_cuda"] = None
        report["bf16"] = False
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.free,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
        report["nvidia_smi"] = result.stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.SubprocessError):
        report["nvidia_smi"] = []
    report["ram_available_gb"] = round(psutil.virtual_memory().available / 2**30, 2)
    if report["ram_available_gb"] < config.section("hardware")["min_free_ram_gb"]:
        report["errors"].append(f"Available RAM below configured minimum {config.section('hardware')['min_free_ram_gb']} GiB")
    for label, path in (("data", config.data_dir), ("checkpoint", config.checkpoint_dir), ("output", config.output_dir)):
        if label == "data" and not path.is_dir():
            report["errors"].append(f"Dataset directory missing: {path}")
            continue
        target = path
        while not target.exists() and target != target.parent:
            target = target.parent
        report[f"{label}_disk_free_gb"] = round(shutil.disk_usage(target).free / 2**30, 2)
        if report[f"{label}_disk_free_gb"] < config.section("hardware")["min_free_disk_gb"]:
            report["errors"].append(f"Free disk at {label} path below configured minimum {config.section('hardware')['min_free_disk_gb']} GiB")
        if label != "data":
            try:
                path.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=path):
                    pass
            except OSError as exc:
                report["errors"].append(f"{label} path not writable: {path}: {exc}")
    if not report["gpu"]:
        report["errors"].append("No CUDA GPU visible to PyTorch")
    elif report["gpu"][report["selected_gpu"]]["free_gb"] < config.section("hardware")["min_free_vram_gb"]:
        advice = "Select a larger GPU or use configs/qwen3vl_8b.yaml" if config.model_key == "qwen3vl_32b" else "Select a GPU with more free memory"
        report["errors"].append(f"Selected GPU {report['selected_gpu']} has {report['gpu'][report['selected_gpu']]['free_gb']} GiB free; minimum is {config.section('hardware')['min_free_vram_gb']} GiB. {advice}")
    if not report["nvidia_smi"]:
        report["warnings"].append("nvidia-smi unavailable")
    for package in ("torch", "transformers", "trl", "unsloth", "unsloth_zoo", "peft", "accelerate", "safetensors", "bitsandbytes"):
        if report["packages"][package] is None:
            report["errors"].append(f"Missing GPU dependency: {package}")
    if report["packages"]["transformers"] not in (None, "4.57.6"):
        report["errors"].append("Install documented GPU requirements: transformers==4.57.6")
    if report["packages"]["trl"] not in (None, "0.22.2"):
        report["errors"].append("Install documented GPU requirements: trl==0.22.2")
    report["ok"] = not report["errors"]
    return report
