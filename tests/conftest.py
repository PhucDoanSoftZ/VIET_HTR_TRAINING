import json
from pathlib import Path
import pytest
from PIL import Image


@pytest.fixture
def tiny_data(tmp_path: Path) -> Path:
    root = tmp_path / "data" / "viet_htr"
    (root / "images").mkdir(parents=True)
    for i, split in enumerate(("train", "validation", "test")):
        Image.new("RGB", (24, 16), (200 + i, 200, 200)).save(root / "images" / f"{split}.png")
        path = f"/kaggle/input/viet-htr/data_clean/images/{split}.png" if i == 0 else f"images/{split}.png"
        (root / f"{split}.jsonl").write_text(json.dumps({"image_path": path, "text": "Tiếng Việt"}, ensure_ascii=False) + "\n", encoding="utf-8")
    return root
