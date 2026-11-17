import json
from pathlib import Path
import pytest
from htrtrain.checkpoints import CheckpointError, latest_valid_checkpoint, manifest, prune_completed_checkpoints, seal_checkpoint, validate_adapter, validate_resume
from htrtrain.config import load_config
from htrtrain.dataset import check_data

ROOT = Path(__file__).resolve().parents[1]


def config_for(root):
    return load_config(ROOT / "configs/qwen3vl_4b.yaml", [f"paths.data_dir={root.as_posix()}"])


def adapter(folder, base="unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"):
    folder.mkdir()
    (folder / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base, "r": 16, "lora_alpha": 16}), encoding="utf-8")
    (folder / "adapter_model.safetensors").write_bytes(b"fixture-not-a-real-model")


def checkpoint(folder, cfg):
    adapter(folder)
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (folder / name).write_bytes(b"fixture")
    (folder / "trainer_state.json").write_text('{"global_step":7500}', encoding="utf-8")
    (folder / "htr_manifest.json").write_text(json.dumps(manifest(cfg)), encoding="utf-8")
    seal_checkpoint(folder)


def test_adapter_4b_cannot_attach_to_8b(tiny_data, tmp_path):
    folder = tmp_path / "checkpoint-7500"
    adapter(folder)
    validate_adapter(folder, config_for(tiny_data))
    with pytest.raises(CheckpointError):
        validate_adapter(folder, load_config(ROOT / "configs/qwen3vl_8b.yaml"))


def test_adapter_only_cannot_resume(tiny_data, tmp_path):
    folder = tmp_path / "checkpoint-7500"
    adapter(folder)
    with pytest.raises(CheckpointError, match="Exact resume impossible"):
        validate_resume(folder, config_for(tiny_data))


def test_manifest_and_state_required(tiny_data, tmp_path):
    cfg = config_for(tiny_data)
    folder = tmp_path / "checkpoint-7500"
    checkpoint(folder, cfg)
    assert validate_resume(folder, cfg)["global_step"] == 7500
    changed = load_config(cfg.path, [f"paths.data_dir={tiny_data.as_posix()}", "training.seed=1"])
    with pytest.raises(CheckpointError, match="training differs"):
        validate_resume(folder, changed)


@pytest.mark.parametrize("filename", ["optimizer.pt", "adapter_model.safetensors", "rng_state.pth", "trainer_state.json"])
def test_corrupt_checkpoint_rejected(tiny_data, tmp_path, filename):
    cfg = config_for(tiny_data)
    folder = tmp_path / "checkpoint-7500"
    checkpoint(folder, cfg)
    (folder / filename).write_bytes(b"tampered")
    with pytest.raises(CheckpointError, match="integrity"):
        validate_resume(folder, cfg)


def test_image_change_invalidates_resume(tiny_data):
    from PIL import Image
    cfg = config_for(tiny_data)
    before = manifest(cfg)
    Image.new("RGB", (24, 16), "black").save(tiny_data / "images/train.png")
    assert manifest(cfg)["fingerprints"] != before["fingerprints"]


def test_path_relocation_keeps_fingerprint(tiny_data):
    cfg = config_for(tiny_data)
    before = manifest(cfg)
    row = {"image_path": "images/train.png", "text": "Tiếng Việt"}
    (tiny_data / "train.jsonl").write_text(json.dumps(row), encoding="utf-8")
    assert manifest(cfg)["fingerprints"] == before["fingerprints"]


def test_best_checkpoint_relocation(tiny_data, tmp_path):
    cfg = config_for(tiny_data)
    folder = tmp_path / "checkpoint-7500"
    checkpoint(folder, cfg)
    (folder / "trainer_state.json").write_text(json.dumps({"global_step":7500, "best_model_checkpoint":"/kaggle/working/checkpoint-7500"}))
    seal_checkpoint(folder)
    assert validate_resume(folder, cfg)["best_model_checkpoint"] == str(folder)


def test_fp16_requires_scaler(tiny_data, tmp_path):
    cfg = config_for(tiny_data)
    folder = tmp_path / "checkpoint-7500"
    checkpoint(folder, cfg)
    contract = manifest(cfg)
    contract["runtime"] = {"bf16": False}
    (folder / "htr_manifest.json").write_text(json.dumps(contract))
    seal_checkpoint(folder)
    with pytest.raises(CheckpointError, match="scaler"):
        validate_resume(folder, cfg)


def test_invalid_marker_schema_rejected(tiny_data, tmp_path):
    cfg = config_for(tiny_data)
    folder = tmp_path / "checkpoint-7500"
    checkpoint(folder, cfg)
    (folder / "COMPLETE.json").write_text("[]")
    with pytest.raises(CheckpointError, match="must be an object"):
        validate_resume(folder, cfg)


def test_audit_fingerprints_reuse_decoded_pixels(tiny_data):
    cfg = config_for(tiny_data)
    report = check_data(tiny_data, ("train", "validation"))
    assert manifest(cfg, report["fingerprints"])["fingerprints"] == manifest(cfg)["fingerprints"]


def test_latest_resume_skips_corrupt_newer_checkpoint(tiny_data, tmp_path):
    cfg = config_for(tiny_data)
    root = tmp_path / "qwen3vl_4b"
    run = root / "run"
    run.mkdir(parents=True)
    good = run / "checkpoint-7500"
    checkpoint(good, cfg)
    bad = run / "checkpoint-8000"
    checkpoint(bad, cfg)
    (bad / "trainer_state.json").write_text('{"global_step":8000}', encoding="utf-8")
    seal_checkpoint(bad)
    (bad / "optimizer.pt").write_bytes(b"corrupt")
    chosen, info = latest_valid_checkpoint(root, cfg, manifest(cfg))
    assert chosen == good and info["global_step"] == 7500
    (good / "optimizer.pt").write_bytes(b"corrupt")
    with pytest.raises(CheckpointError, match="refusing to restart"):
        latest_valid_checkpoint(root, cfg, manifest(cfg))


def test_prune_only_sealed_and_keep_best(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for step in (100, 200, 300, 400):
        folder = run / f"checkpoint-{step}"
        folder.mkdir()
        if step != 200:
            (folder / "COMPLETE.json").write_text("{}")
    removed = prune_completed_checkpoints(run, 1, str(run / "checkpoint-100"))
    assert [p.name for p in removed] == ["checkpoint-300"]
    assert (run / "checkpoint-100").exists() # Best remains usable.
    assert (run / "checkpoint-200").exists() # Interrupted save is never pruned.
    assert (run / "checkpoint-400").exists() # Latest remains usable.
