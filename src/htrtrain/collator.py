"""Qwen processor batching with exact prompt-prefix masking and no truncation."""
from .dataset import messages

class OCRCollator:
    def __init__(self, processor, prompt, max_length):
        self.processor, self.prompt, self.max_length = processor, prompt, max_length

    def __call__(self, records):
        import torch
        texts = [self.processor.apply_chat_template(messages(self.prompt, r["text"]), tokenize=False, add_generation_prompt=False) for r in records]
        prefix = self.processor.apply_chat_template(messages(self.prompt), tokenize=False, add_generation_prompt=True)
        images = [r["image"] for r in records]
        kwargs = dict(images=images, add_special_tokens=False, return_tensors="pt", padding=True, truncation=False)
        batch = self.processor(text=texts, **kwargs)
        prompts = self.processor(text=[prefix] * len(records), **kwargs)
        if not {"pixel_values", "image_grid_thw", "attention_mask", "input_ids"} <= set(batch):
            raise ValueError("Qwen3-VL processor missing vision tensors or masks")
        ids, mask = batch["input_ids"], batch["attention_mask"]
        image_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        grid = batch["image_grid_thw"]
        merge = self.processor.image_processor.merge_size
        if grid.shape[0] != len(records) or batch["pixel_values"].numel() == 0:
            raise ValueError("Expected exactly one nonempty image per OCR record")
        expected_image_tokens = grid.prod(dim=1) // (merge * merge)
        if not torch.equal((ids == image_id).sum(dim=1).to(expected_image_tokens.device), expected_image_tokens):
            raise ValueError("Image tokens do not match Qwen image_grid_thw; processor mismatch")
        if ids.shape[1] > self.max_length:
            raise ValueError(f"Sequence length {ids.shape[1]} exceeds max_seq_length={self.max_length}; truncation is forbidden")
        labels = torch.full_like(ids, -100)
        for i, record in enumerate(records):
            full = ids[i][mask[i].bool()]
            prompt_ids = prompts["input_ids"][i][prompts["attention_mask"][i].bool()]
            n = len(prompt_ids)
            if n >= len(full) or not torch.equal(full[:n], prompt_ids):
                raise ValueError(f"{record['record_id']}: processor/template prompt-prefix mismatch or empty target")
            positions = mask[i].nonzero(as_tuple=True)[0]
            labels[i, positions[n:]] = full[n:]
        batch["labels"] = labels
        batch.pop("token_type_ids", None)
        return batch
