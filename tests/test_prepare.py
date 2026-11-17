import json

import pytest

from htrtrain.cli import main
from htrtrain.dataset import DatasetError, check_data, load_split
from htrtrain.prepare import prepare_data


def test_prepare_preserves_labels_and_image_bytes(tiny_data, tmp_path):
    original = "  Đặng Thị Bích, 1968.  "
    train_file = tiny_data / "train.jsonl"
    row = json.loads(train_file.read_text(encoding="utf-8"))
    row["text"] = original
    row["source_id"] = "document-1"
    row["proper_names"] = [original.split(",")[0].strip()]
    train_file.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    source_bytes = (tiny_data / "images/train.png").read_bytes()
    source_json = train_file.read_bytes()
    destination = tmp_path / "prepared"

    result = prepare_data(tiny_data, destination)

    assert result["counts"] == {"train": 1, "validation": 1, "test": 1}
    prepared = load_split(destination, "train")[0]
    assert prepared.text == original
    assert prepared.source_id == "document-1"
    assert prepared.proper_names == tuple(row["proper_names"])
    assert prepared.image.read_bytes() == source_bytes
    assert train_file.read_bytes() == source_json
    assert check_data(destination)["ok"]


def test_prepare_rejects_bad_data_without_publishing(tiny_data, tmp_path):
    (tiny_data / "images/test.png").write_bytes(b"broken")
    destination = tmp_path / "prepared"
    with pytest.raises(DatasetError, match="data-check failed"):
        prepare_data(tiny_data, destination)
    assert not destination.exists()


def test_prepare_will_not_overwrite_or_nest(tiny_data, tmp_path):
    destination = tmp_path / "prepared"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(DatasetError, match="already exists"):
        prepare_data(tiny_data, destination)
    assert sentinel.read_text() == "keep"
    with pytest.raises(DatasetError, match="separate directories"):
        prepare_data(tiny_data, tiny_data / "nested")


def test_prepare_cli_is_explicit(tiny_data, tmp_path, capsys):
    destination = tmp_path / "prepared"
    assert main(["data-prepare", "--source", str(tiny_data), "--dest", str(destination)]) == 0
    assert destination.is_dir()
    assert '"label_policy": "verbatim"' in capsys.readouterr().out
