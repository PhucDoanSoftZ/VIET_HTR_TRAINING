from pathlib import Path
import pytest
from htrtrain.cli import main, parser


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("size", ("8b", "32b"))
def test_cpu_dry_run_uses_fixture_only(tiny_data, capsys, size):
    code = main(["train", "--config", str(ROOT / f"configs/qwen3vl_{size}.yaml"),
                 "--set", f"paths.data_dir={tiny_data.as_posix()}", "--dry-run"])
    assert code == 0
    assert '"dry_run": true' in capsys.readouterr().out


def test_data_check_fixture(tiny_data, capsys):
    assert main(["data-check", "--data-dir", str(tiny_data)]) == 0
    assert '"test_overlap_images": 0' in capsys.readouterr().out


def test_eval_defaults_to_validation():
    assert parser().parse_args(["eval", "--config", "configs/qwen3vl_32b.yaml"]).split == "validation"


def test_dry_run_does_not_read_beyond_two_records(tiny_data):
    for split in ("train", "validation"):
        path = tiny_data / f"{split}.jsonl"
        original = path.read_text(encoding="utf-8")
        import json
        from PIL import Image
        Image.new("RGB", (24, 16), (10, 20, 30) if split == "train" else (40, 50, 60)).save(tiny_data / "images" / f"{split}2.png")
        extra = json.dumps({"image_path": f"images/{split}2.png", "text": "fixture"})
        path.write_text(original + extra + "\nINVALID RECORD BEYOND LIMIT\n", encoding="utf-8")
    assert main(["train", "--config", str(ROOT / "configs/qwen3vl_8b.yaml"), "--set", f"paths.data_dir={tiny_data.as_posix()}", "--dry-run"]) == 0
