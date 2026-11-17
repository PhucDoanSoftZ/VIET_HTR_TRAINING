"""CPU contract tests; this processor fixture does not certify the GPU stack."""
import pytest
import torch
from types import SimpleNamespace
from PIL import Image
from htrtrain.collator import OCRCollator


class ProcessorFixture:
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: ord("%"))
    image_processor = SimpleNamespace(merge_size=1)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "USER%PROMPT:ASSISTANT:" + (messages[1]["content"][0]["text"] + "~" if len(messages) == 2 else "")

    def __call__(self, text, images, **kwargs):
        assert kwargs["truncation"] is False
        tokens = [[ord(c) for c in t] for t in text]
        width = max(map(len, tokens))
        return {"input_ids": torch.tensor([t + [0] * (width - len(t)) for t in tokens]),
                "attention_mask": torch.tensor([[1] * len(t) + [0] * (width - len(t)) for t in tokens]),
                "pixel_values": torch.zeros(len(images), 3, 16, 16), "image_grid_thw": torch.ones(len(images), 3, dtype=torch.long)}


def rows():
    return [{"image": Image.new("RGB", (16, 16)), "text": text, "record_id": f"train:{i}"} for i, text in enumerate(("Tiếng Việt", "abc"))]


def test_exact_prompt_padding_and_target_mask():
    batch = OCRCollator(ProcessorFixture(), "OCR", 100)(rows())
    prefix_length = len("USER%PROMPT:ASSISTANT:")
    assert torch.all(batch["labels"][:, :prefix_length] == -100)
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)
    for i, row in enumerate(rows()):
        assert batch["labels"][i][batch["labels"][i] != -100].tolist() == [ord(c) for c in row["text"] + "~"]
    assert "pixel_values" in batch and "image_grid_thw" in batch


def test_overlength_never_truncates():
    with pytest.raises(ValueError, match="truncation is forbidden"):
        OCRCollator(ProcessorFixture(), "OCR", 2)(rows())


def test_prefix_mismatch_fails():
    class Mismatch(ProcessorFixture):
        def apply_chat_template(self, messages, **kwargs):
            text = super().apply_chat_template(messages, **kwargs)
            return "WRONG" + text if len(messages) == 1 else text
    with pytest.raises(ValueError, match="mismatch"):
        OCRCollator(Mismatch(), "OCR", 100)(rows())


def test_image_grid_mismatch_fails():
    class Mismatch(ProcessorFixture):
        def __call__(self, *args, **kwargs):
            batch = super().__call__(*args, **kwargs)
            batch["image_grid_thw"] *= 2
            return batch
    with pytest.raises(ValueError, match="Image tokens"):
        OCRCollator(Mismatch(), "OCR", 100)(rows())


def test_real_cpu_backward_only_updates_adapter():
    batch = OCRCollator(ProcessorFixture(), "OCR", 100)(rows())
    torch.manual_seed(1)
    embedding = torch.nn.Embedding(10000, 8)
    embedding.weight.requires_grad_(False)
    lora = torch.nn.Linear(8, 10000, bias=False)
    optimizer = torch.optim.AdamW(lora.parameters(), lr=0.01)
    logits = lora(embedding(batch["input_ids"]))
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 10000), batch["labels"][:, 1:].reshape(-1), ignore_index=-100)
    loss.backward()
    before = lora.weight.detach().clone()
    optimizer.step()
    assert embedding.weight.grad is None
    assert torch.isfinite(loss)
    assert not torch.equal(before, lora.weight)
