from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from htrtrain.config import load_config
from htrtrain.models import check_trainable, load_base

ROOT = Path(__file__).resolve().parents[1]


class TinyParameters:
    def __init__(self, vision=True, language=True):
        self.vision, self.language = vision, language

    def named_parameters(self):
        base = torch.nn.Parameter(torch.ones(10000), requires_grad=False)
        yield "base.weight", base
        if self.vision:
            yield "model.visual.attn.lora_B.weight", torch.nn.Parameter(torch.ones(2))
        if self.language:
            yield "model.language.layers.q_proj.lora_B.weight", torch.nn.Parameter(torch.ones(2))


def test_trainable_selection_and_freeze_policy():
    cfg = load_config(ROOT / "configs/qwen3vl_4b.yaml")
    report = check_trainable(TinyParameters(), cfg)
    assert report["trainable_parameters"] == 4
    with pytest.raises(RuntimeError, match="freeze policy"):
        check_trainable(TinyParameters(vision=False), cfg)
    frozen = load_config(cfg.path, ["lora.vision_layers=false"])
    assert check_trainable(TinyParameters(vision=False), frozen)["trainable_parameters"] == 2


def test_loader_passes_revision_quantization_and_lora_settings(monkeypatch):
    import sys
    calls = {}
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_vl"), is_loaded_in_4bit=True,
                            hf_device_map={"": 0}, named_parameters=TinyParameters().named_parameters)
    processor = SimpleNamespace(image_processor=SimpleNamespace(size={}), chat_template="Qwen template", tokenizer=SimpleNamespace(pad_token_id=0))

    class Backend:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls["model_id"], calls["loader"] = name, kwargs
            return model, processor

        @staticmethod
        def get_peft_model(model, **kwargs):
            calls["lora"] = kwargs
            return model

        @staticmethod
        def for_training(model, **kwargs):
            return model

    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastVisionModel=Backend))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(BitsAndBytesConfig=lambda **kwargs: kwargs))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    cfg = load_config(ROOT / "configs/qwen3vl_4b.yaml", ["model.revision=fixed-sha", "lora.dropout=0.05"])
    loaded, _ = load_base(cfg, True)
    assert loaded is model
    assert calls["loader"]["revision"] == "fixed-sha"
    assert calls["loader"]["load_in_4bit"] is True
    assert calls["lora"]["lora_dropout"] == 0.05
    assert calls["loader"]["device_map"] == {"": 0}
    assert calls["lora"]["use_gradient_checkpointing"] == cfg.section("model")["gradient_checkpointing"]


@pytest.mark.parametrize("bad_shape", [False, True])
def test_adapter_tensor_topology_is_checked(monkeypatch, bad_shape):
    import sys
    import htrtrain.models as backend
    import htrtrain.checkpoints as checkpoints
    model = TinyParameters()
    expected = {"lora_B.weight": torch.zeros(2, 3)}
    saved = {"lora_B.weight": torch.ones(2, 4 if bad_shape else 3)}
    loaded = []
    monkeypatch.setattr(checkpoints, "validate_adapter", lambda *args: None)
    monkeypatch.setattr(backend, "load_base", lambda *args, **kwargs: (model, "processor"))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        get_peft_model_state_dict=lambda model: expected,
        set_peft_model_state_dict=lambda model, state, **kwargs: loaded.append(state)))
    monkeypatch.setitem(sys.modules, "safetensors.torch", SimpleNamespace(load_file=lambda *args, **kwargs: saved))
    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastVisionModel=SimpleNamespace(for_inference=lambda model: model)))
    cfg = load_config(ROOT / "configs/qwen3vl_4b.yaml")
    if bad_shape:
        with pytest.raises(RuntimeError, match="topology"):
            backend.load_adapter(cfg, Path("adapter"))
        assert not loaded
    else:
        assert backend.load_adapter(cfg, Path("adapter"))[0] is model
        assert len(loaded) == 1
