import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from PIL import Image
from htrtrain.config import ConfigError, load_config
from htrtrain.dataset import DatasetError, check_data, prepare_image, resolve_image
from htrtrain.models import check_trainable
from htrtrain.observability import RunLog
from htrtrain.train import resolve_start, training_kwargs

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("override", ["training.learning_rate=.nan", "training.learning_rate=true", "lora.rank=true", "inference.max_new_tokens=true", "training.warmup_ratio=2", "lora.dropout=1", "data.resize_policy=unknown", "training.best_metric=eval_cer", "logging.debug=1", "hardware.min_free_vram_gb=.inf", "training.optimizer=[]"])
def test_bad_config_rejected_before_model(override):
    with pytest.raises(ConfigError):
        load_config(ROOT / "configs/qwen3vl_4b.yaml", [override])


def test_duplicate_pixels_and_provenance(tiny_data):
    Image.open(tiny_data / "images/train.png").save(tiny_data / "images/test.png")
    report = check_data(tiny_data)
    assert not report["ok"] and report["test_overlap_images"] == 1


def test_train_validation_leakage(tiny_data):
    Image.open(tiny_data / "images/train.png").save(tiny_data / "images/validation.png")
    assert check_data(tiny_data, ("train", "validation"))["overlap_records"] == 1


def test_provenance_leakage_even_if_pixels_differ(tiny_data):
    for split in ("train", "test"):
        path = tiny_data / f"{split}.jsonl"
        row = json.loads(path.read_text(encoding="utf-8"))
        row["source_id"] = "same-origin"
        path.write_text(json.dumps(row), encoding="utf-8")
    assert not check_data(tiny_data)["ok"]


def test_unsafe_legacy_and_ambiguous_paths(tiny_data):
    with pytest.raises(DatasetError):
        resolve_image(tiny_data, "/kaggle/../images/train.png")
    with pytest.raises(DatasetError):
        resolve_image(tiny_data, "/arbitrary/file.png")


def test_resize_requires_opt_in(tiny_data):
    with pytest.raises(DatasetError, match="exceeds"):
        prepare_image(tiny_data / "images/train.png", {"max_pixels": 100})


def test_resume_flags_do_not_silently_override_each_other():
    cfg = load_config(ROOT / "configs/qwen3vl_4b.yaml")
    with pytest.raises(ValueError):
        resolve_start(cfg, Path("a"), Path("b"))


def test_training_safety_settings():
    cfg = load_config(ROOT / "configs/qwen3vl_8b.yaml")
    args = training_kwargs(cfg, {"bf16": True}, Path("out"))
    assert args["remove_unused_columns"] is False and args["save_only_model"] is False
    assert args["bf16"] is True and args["fp16"] is False
    assert args["logging_nan_inf_filter"] is False
    assert training_kwargs(cfg, {"bf16": False}, Path("out"), True)["max_steps"] == 1


def test_non_lora_trainable_rejected():
    class Model:
        def named_parameters(self):
            return [("base.weight", torch.nn.Parameter(torch.ones(10)))]
    with pytest.raises(RuntimeError, match="non-LoRA"):
        check_trainable(Model(), load_config(ROOT / "configs/qwen3vl_4b.yaml"))


def test_structured_logs_no_duplicate_handlers(tmp_path):
    log = RunLog(tmp_path / "run", "qwen3vl_4b")
    count = len(log.console.handlers)
    log.emit("metrics", component="test", step=1, loss=0.5)
    log.close()
    second = RunLog(tmp_path / "run2", "qwen3vl_4b")
    assert len(second.console.handlers) == count
    second.close()
    row = json.loads((tmp_path / "run/events.jsonl").read_text())
    assert {"timestamp", "level", "component", "run_id", "event", "model"} <= row.keys()
    assert (tmp_path / "run/metrics.csv").is_file()


def test_module_cli_returns_failure_code():
    result = subprocess.run([sys.executable, "-m", "htrtrain", "train", "--config", "not-found.yaml", "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 2 and "ERROR" in result.stderr
