"""Explicit, offline conversion to the canonical clean-data layout.

This command never changes source labels or image bytes. It is not called by train.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .dataset import DatasetError, check_data, iter_split


SPLITS = ("train", "validation", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_data(source: Path, destination: Path) -> dict:
    """Audit source, copy original images, and write portable JSONL in a new directory."""
    source, destination = source.expanduser().resolve(), destination.expanduser().resolve()
    if not source.is_dir():
        raise DatasetError(f"Source directory missing: {source}")
    if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise DatasetError("Source and destination must be separate directories")
    if destination.exists():
        raise DatasetError(f"Destination already exists: {destination}. Choose a new path to preserve existing data")
    report = check_data(source, SPLITS)
    if not report["ok"]:
        raise DatasetError("Source data-check failed; fix missing/corrupt/duplicate records before preparing: "
                           + json.dumps(report["splits"], ensure_ascii=False))
    destination.parent.mkdir(parents=True, exist_ok=True)
    # TemporaryDirectory removes only artifacts created by this invocation if it fails.
    with tempfile.TemporaryDirectory(prefix=".htr-prepare-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "prepared"
        (stage / "images").mkdir(parents=True)
        counts: dict[str, int] = {}
        for split in SPLITS:
            count = 0
            with (stage / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as writer:
                for sample in iter_split(source, split):
                    count += 1
                    # Split and record number prevent filename collisions. Original bytes are kept.
                    suffix = sample.image.suffix.lower()
                    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
                        suffix = ".img"
                    name = f"{split}-{count:07d}{suffix}"
                    shutil.copyfile(sample.image, stage / "images" / name)
                    row = {"image_path": f"images/{name}", "text": sample.text}
                    if sample.source_id is not None:
                        row["source_id"] = sample.source_id
                    if sample.proper_names:
                        row["proper_names"] = list(sample.proper_names)
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] = count
        staged_report = check_data(stage, SPLITS)
        if not staged_report["ok"] or any(staged_report["splits"][s]["count"] != counts[s] for s in SPLITS):
            raise DatasetError("Prepared data failed integrity check; destination was not published")
        manifest = {
            "format": "htrtrain-clean-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "counts": counts,
            "label_policy": "verbatim",
            "image_policy": "original_bytes",
            "splits_sha256": {s: _sha256(stage / f"{s}.jsonl") for s in SPLITS},
        }
        (stage / "prepare_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        # Same-volume rename publishes only a complete, audited dataset.
        stage.rename(destination)
    return {"destination": str(destination), **manifest}
