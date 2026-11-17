"""Shared image/prompt/decoding path for validation, benchmark and page OCR."""
from pathlib import Path
from .config import Config
from .dataset import messages, prepare_image

def predict(model, processor, image_path: Path, config: Config, details=False):
    import torch
    image = prepare_image(image_path, config.section("data"))
    prompt = processor.apply_chat_template(messages(config.section("inference")["prompt"]), tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[image], text=[prompt], add_special_tokens=False, return_tensors="pt", truncation=False)
    inputs.pop("token_type_ids", None)
    if inputs["input_ids"].shape[1] > config.section("training")["max_seq_length"]:
        raise ValueError("OCR prompt/image exceeds token budget; increase max_seq_length or explicitly downscale")
    device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    settings = config.section("inference")
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=settings["max_new_tokens"], do_sample=settings["do_sample"],
            num_beams=settings["num_beams"], repetition_penalty=settings["repetition_penalty"], use_cache=True)
    new_ids = ids[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    result = {"text": text, "generated_tokens": int(new_ids.shape[1]), "hit_token_limit": new_ids.shape[1] >= settings["max_new_tokens"]}
    return result if details else text
