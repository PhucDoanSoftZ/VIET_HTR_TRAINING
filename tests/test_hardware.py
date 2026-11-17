from pathlib import Path
import subprocess

import torch

from htrtrain.config import load_config
from htrtrain.hardware import doctor


ROOT = Path(__file__).resolve().parents[1]


def test_doctor_checks_selected_gpu_not_pooled_vram(tmp_path, monkeypatch):
    import htrtrain.hardware as hardware

    data = tmp_path / "data"
    data.mkdir()
    cfg = load_config(ROOT / "configs/qwen3vl_32b.yaml", [
        f"paths.data_dir={data.as_posix()}",
        f"paths.checkpoint_dir={(tmp_path / 'checkpoints').as_posix()}",
        f"paths.output_dir={(tmp_path / 'outputs').as_posix()}",
        "hardware.min_free_ram_gb=1", "hardware.min_free_disk_gb=1",
    ])
    gib = 2**30
    monkeypatch.setattr(hardware.importlib.metadata, "version", lambda name: {"transformers": "4.57.6", "trl": "0.22.2"}.get(name, "test"))
    monkeypatch.setattr(hardware.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="GPU A\nGPU B"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: ((20 if index == 0 else 80) * gib, 80 * gib))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"GPU {index}")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    report = doctor(cfg)
    assert report["selected_gpu"] == 0
    assert not report["ok"]
    assert any("Selected GPU 0" in error and "8b.yaml" in error for error in report["errors"])
    assert report["gpu"][1]["free_gb"] == 80

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert doctor(cfg)["ok"]
