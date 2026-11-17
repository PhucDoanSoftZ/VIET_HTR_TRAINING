from pathlib import Path
import pytest
from htrtrain.config import load_config, ConfigError
from htrtrain.models import require_supported, UnsupportedModelError


ROOT = Path(__file__).resolve().parents[1]


def test_config_paths_and_override():
    cfg = load_config(ROOT / "configs/qwen3vl_8b.yaml", ["training.batch_size=2", "paths.data_dir=data/viet_htr", "paths.resume_from=checkpoints/checkpoint-7500"])
    assert cfg.section("training")["batch_size"] == 2
    assert cfg.data_dir == ROOT / "data/viet_htr"
    assert cfg.optional_path("resume_from") == ROOT / "checkpoints/checkpoint-7500"


def test_invalid_config_rejected():
    with pytest.raises(ConfigError):
        load_config(ROOT / "configs/qwen3vl_8b.yaml", ["training.batch_size=0"])
    with pytest.raises(ConfigError):
        load_config(ROOT / "configs/qwen3vl_8b.yaml", ["training.no_such_key=1"])


def test_unverified_models_rejected():
    for name in ("gemma4",):
        with pytest.raises(UnsupportedModelError):
            require_supported(load_config(ROOT / f"configs/{name}.yaml"))


def test_32b_config_and_model_identity():
    cfg = load_config(ROOT / "configs/qwen3vl_32b.yaml")
    require_supported(cfg)
    assert cfg.model_id == "unsloth/Qwen3-VL-32B-Instruct-unsloth-bnb-4bit"
    assert cfg.section("lora")["vision_layers"] is False
    assert cfg.section("training")["learning_rate"] == 2e-5
    assert cfg.section("training")["best_metric"] == "eval_loss"
    assert load_config(cfg.path, ["training.learning_rate=0.00005", "training.best_metric=eval_cer"]).section("training")["best_metric"] == "eval_cer"
    with pytest.raises(ConfigError, match="architecture/size"):
        load_config(cfg.path, ["model.id=unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"])
