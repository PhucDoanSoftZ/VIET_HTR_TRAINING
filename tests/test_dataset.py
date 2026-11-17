import json
import pytest
from htrtrain.dataset import DatasetError, VisionDataset, check_data, load_split, resolve_image


def test_kaggle_and_relative_paths(tiny_data):
    samples = load_split(tiny_data, "train")
    assert samples[0].image == tiny_data / "images/train.png"
    assert samples[0].text == "Tiếng Việt"
    assert check_data(tiny_data)["splits"]["train"]["count"] == 1
    assert check_data(tiny_data)["test_overlap_images"] == 0


def test_missing_and_corrupt(tiny_data):
    (tiny_data / "images/validation.png").write_bytes(b"not an image")
    (tiny_data / "images/test.png").unlink()
    report = check_data(tiny_data)
    assert report["splits"]["validation"]["issues"]["corrupt_image"] == 1
    assert report["splits"]["test"]["issues"]["missing_image"] == 1


def test_empty_label_and_traversal(tiny_data):
    (tiny_data / "train.jsonl").write_text(json.dumps({"image_path": "images/train.png", "text": " "}), encoding="utf-8")
    with pytest.raises(DatasetError):
        load_split(tiny_data, "train")
    with pytest.raises(DatasetError):
        resolve_image(tiny_data, "images/../secret.png")


def test_vision_dataset(tiny_data):
    ds = VisionDataset(load_split(tiny_data, "train"), "OCR", 400)
    row = ds[0]
    assert row["text"] == "Tiếng Việt"
    assert row["image"].width == 24


def test_existing_viet_htr_clean_layout_and_relative_path(tmp_path):
    from PIL import Image
    root = tmp_path / "viet_htr"
    clean = root / "VietHTR_DataClean"
    image_path = clean / "images" / "train" / "000000.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (24, 16), "white").save(image_path)
    (clean / "train.jsonl").write_text(json.dumps({
        "image": "/kaggle/working/data_clean/images/train/wrong.png",
        "image_relpath": "images/train/000000.png",
        "text": "Giấy nhượng đất",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    sample = load_split(root, "train")[0]
    assert sample.image == image_path
    assert sample.text == "Giấy nhượng đất"
