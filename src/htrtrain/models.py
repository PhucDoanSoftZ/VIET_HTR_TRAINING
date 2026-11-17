"""Qwen3-VL 4B/8B/32B Unsloth loading with explicit GPU placement and PEFT checks."""
from pathlib import Path
from importlib import import_module
from .config import Config, SUPPORTED

class UnsupportedModelError(RuntimeError):
    """No validated implementation exists for this architecture/size."""


def _gpu_dependency(name: str):
    """Load optional GPU packages only when model execution is requested."""
    try:
        return import_module(name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Cannot load GPU dependency {name!r}: missing {exc.name!r}. "
            "On the Linux GPU server, run bash scripts/setup_server.sh; "
            "CPU development does not require these packages."
        ) from exc

def require_supported(config: Config):
    if config.model_key not in SUPPORTED:
        raise UnsupportedModelError(f"{config.model_key}: backend pending verification; use Qwen3-VL 4B/8B/32B")

def configure_processor(processor, config):
    if getattr(processor, "image_processor", None) is None or not getattr(processor, "chat_template", None):
        raise RuntimeError("Processor lacks image_processor/chat_template")
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        raise RuntimeError("Processor tokenizer has no pad token")
    # Qwen dynamic resolution still snaps to patches. The same limits apply to every path.
    processor.image_processor.max_pixels = config.section("data")["max_pixels"]
    if isinstance(getattr(processor.image_processor, "size", None), dict):
        processor.image_processor.size["longest_edge"] = config.section("data")["max_pixels"]
    return processor

def load_base(config: Config, for_training=False):
    require_supported(config)
    FastVisionModel = _gpu_dependency("unsloth").FastVisionModel  # Before Transformers/TRL.
    import torch
    from transformers import BitsAndBytesConfig
    kwargs = {"load_in_4bit": True, "revision": config.section("model")["revision"],
              "use_gradient_checkpointing": config.section("model")["gradient_checkpointing"],
              "device_map": {"": torch.cuda.current_device()}, "trust_remote_code": False,
              "attn_implementation": "sdpa"}
    if "bnb-4bit" not in config.model_id:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    model, processor = FastVisionModel.from_pretrained(config.model_id, **kwargs)
    if getattr(model.config, "model_type", None) != "qwen3_vl" or not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("Loaded model is not a quantized Qwen3-VL")
    placement = getattr(model, "hf_device_map", {})
    if any(str(device) in {"cpu", "disk"} for device in placement.values()):
        raise RuntimeError("CPU/disk offload is forbidden; select a larger GPU or smaller model")
    selected = torch.cuda.current_device()
    for device in placement.values():
        device_name = str(device)
        if (isinstance(device, int) and device != selected) or (device_name.startswith("cuda:") and device_name != f"cuda:{selected}"):
            raise RuntimeError("Model weights span multiple GPUs unexpectedly; this Trainer path requires one complete model replica")
    processor = configure_processor(processor, config)
    if for_training:
        lora = config.section("lora")
        kwargs = dict(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            bias="none", random_state=config.section("training")["seed"],
            use_gradient_checkpointing=config.section("model")["gradient_checkpointing"],
            finetune_vision_layers=lora["vision_layers"], finetune_language_layers=lora["language_layers"],
            finetune_attention_modules=lora["attention_modules"], finetune_mlp_modules=lora["mlp_modules"])
        if lora["target_modules"]:
            kwargs["target_modules"] = lora["target_modules"]
        model = FastVisionModel.get_peft_model(model, **kwargs)
        FastVisionModel.for_training(model, use_gradient_checkpointing=config.section("model")["gradient_checkpointing"])
        check_trainable(model, config)
    else:
        FastVisionModel.for_inference(model)
    return model, processor

def load_adapter(config: Config, adapter_path: Path, for_training=False):
    from .checkpoints import validate_adapter
    validate_adapter(adapter_path, config)
    # Build through Unsloth's training preparation so k-bit input gradients and
    # optimized LoRA modules are configured identically for new/resumed runs.
    model, processor = load_base(config, for_training=True)
    peft = _gpu_dependency("peft")
    from safetensors.torch import load_file
    FastVisionModel = _gpu_dependency("unsloth").FastVisionModel
    weights = load_file(str(adapter_path / "adapter_model.safetensors"), device="cpu")
    expected = peft.get_peft_model_state_dict(model)
    if set(weights) != set(expected) or any(weights[key].shape != expected[key].shape for key in expected):
        raise RuntimeError("Adapter tensor keys/shapes differ from configured Unsloth LoRA topology")
    peft.set_peft_model_state_dict(model, weights, ignore_mismatched_sizes=False)
    if for_training:
        FastVisionModel.for_training(model, use_gradient_checkpointing=config.section("model")["gradient_checkpointing"])
        check_trainable(model, config)
    else:
        FastVisionModel.for_inference(model)
    return model, processor

def check_trainable(model, config):
    total, trainable = 0, 0
    vision, language = False, False
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        if not parameter.requires_grad:
            continue
        trainable += parameter.numel()
        if "lora_" not in name:
            raise RuntimeError(f"Unexpected non-LoRA trainable parameter: {name}")
        is_vision = ".visual." in name or ".vision" in name
        if not config.section("lora")["attention_modules"] and any(x in name for x in ("q_proj", "k_proj", "v_proj", "o_proj", ".attn.")):
            raise RuntimeError("Attention LoRA targets violate configured freeze policy")
        if not config.section("lora")["mlp_modules"] and ".mlp." in name:
            raise RuntimeError("MLP LoRA targets violate configured freeze policy")
        vision |= is_vision
        language |= not is_vision
    lora = config.section("lora")
    if not trainable or trainable / max(total, 1) > 0.2:
        raise RuntimeError("Unexpected trainable parameter count; inspect LoRA targets")
    if vision != lora["vision_layers"] or language != lora["language_layers"]:
        raise RuntimeError("Adapter vision/language targets differ from configured freeze policy")
    return {"trainable_parameters": trainable, "total_parameters": total, "trainable_ratio": trainable / total}
