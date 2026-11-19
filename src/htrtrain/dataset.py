"""Local JSONL validation, split integrity and shared image preprocessing."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from PIL import Image, ImageOps, UnidentifiedImageError

class DatasetError(ValueError):
    """Invalid dataset; no records are silently dropped."""

@dataclass(frozen=True)
class Sample:
    image: Path
    text: str
    split: str
    line: int
    source_id: str | None = None
    proper_names: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    gt_verification: str | None = None
    quality_status: str | None = None

def jsonl_path(root: Path, split: str) -> Path:
    if split not in {"train", "validation", "test"}:
        raise DatasetError(f"Invalid split: {split}")
    candidates = [root / f"{split}.jsonl", root / "data_clean" / f"{split}.jsonl",
                  root / "VietHTR_DataClean" / f"{split}.jsonl"]
    existing = [p for p in candidates if p.is_file()]
    if len(existing) != 1:
        raise DatasetError(f"Expected exactly one {split}.jsonl under {root}, data_clean/ or VietHTR_DataClean/")
    return existing[0]

def resolve_image(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError("Empty image path")
    parts = PurePosixPath(value.replace("\\", "/")).parts
    if ".." in parts or "\x00" in value:
        raise DatasetError("Unsafe image path")
    if "images" in parts:
        suffix = parts[parts.index("images") + 1:]
        candidates = [root / "images" / Path(*suffix), root / "data_clean/images" / Path(*suffix),
                      root / "VietHTR_DataClean/images" / Path(*suffix)]
    else:
        suffix = parts[1:] if parts and parts[0] == "data_clean" else parts
        if not suffix or PurePosixPath(*suffix).is_absolute() or ":" in suffix[0]:
            raise DatasetError("Absolute legacy image paths must contain images/")
        candidates = [root / Path(*suffix), root / "data_clean" / Path(*suffix),
                      root / "VietHTR_DataClean" / Path(*suffix), root / "images" / Path(*suffix)]
    if not suffix:
        raise DatasetError("Image path has no filename")
    resolved = [p.resolve() for p in candidates]
    if any(not p.is_relative_to(root.resolve()) for p in resolved):
        raise DatasetError("Image path escapes dataset root")
    matches = list(dict.fromkeys(p for p in resolved if p.is_file()))
    if len(matches) > 1:
        raise DatasetError("Ambiguous image path; remove duplicate directory layouts")
    return matches[0] if matches else resolved[0]

def iter_split(root: Path, split: str):
    with jsonl_path(root, split).open(encoding="utf-8") as stream:
        for line, raw in enumerate(stream, 1):
            if not raw.strip():
                raise DatasetError(f"{split}:{line}: blank JSONL record")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{split}:{line}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise DatasetError(f"{split}:{line}: expected object")
            image = next((row[k] for k in ("image_relpath", "image_path", "image", "path", "file_name") if k in row), None)
            text = next((row[k] for k in ("text", "transcription", "label", "ground_truth", "gt") if k in row), None)
            if not isinstance(text, str) or not text.strip():
                raise DatasetError(f"{split}:{line}: empty or non-string label")
            if any(marker in text for marker in ("<|im_start|>", "<|im_end|>", "<|image_pad|>")):
                raise DatasetError(f"{split}:{line}: label contains reserved chat token")
            origin = row.get("source_id", row.get("original_id"))
            if origin is not None and not isinstance(origin, str):
                raise DatasetError(f"{split}:{line}: source_id must be string")
            names = row.get("proper_names", [])
            if not isinstance(names, list) or not all(isinstance(name, str) and name.strip() and name in text for name in names):
                raise DatasetError(f"{split}:{line}: proper_names must be a list of nonempty substrings of text")
            sources = row.get('source_ids', [])
            if not isinstance(sources, list) or not all(isinstance(x, str) and x for x in sources):
                raise DatasetError(f"{split}:{line}: source_ids must be a list of nonempty strings")
            status = row.get('gt_verification')
            if status is not None and not isinstance(status, str):
                raise DatasetError(f"{split}:{line}: gt_verification must be a string")
            quality = row.get('quality_status')
            if quality is not None and not isinstance(quality, str):
                raise DatasetError(f"{split}:{line}: quality_status must be a string")
            yield Sample(resolve_image(root, image), text, split, line, origin, tuple(names), tuple(sources), status, quality)

def load_split(root: Path, split: str, limit: int | None = None) -> list[Sample]:
    from itertools import islice
    if limit is not None and limit <= 0:
        raise DatasetError("limit must be positive")
    records = iter_split(root, split)
    result = list(islice(records, limit)) if limit is not None else list(records)
    if not result:
        raise DatasetError(f"{split}: no samples")
    return result

def pixel_digest(path: Path) -> tuple[str, tuple[int, int], str]:
    with Image.open(path) as source:
        source.load()  # verify() alone can miss decoding errors.
        mode = source.mode
        image = ImageOps.exif_transpose(source).convert("RGB")
    digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    return digest, image.size, mode

def check_data(root: Path, splits=("train", "validation", "test"), limit=None, settings=None) -> dict:
    report = {"root": str(root), "splits": {}, "overlap_records": 0, "test_overlap_images": 0,
              "limited": limit is not None, "fingerprints": {},
              "provenance_check": "source_id/original_id when present; transformed derivatives require provenance"}
    seen: dict[tuple[str, str], str] = {}
    for split in splits:
        samples = load_split(root, split, limit)
        issues, modes = Counter(), Counter()
        dimensions = []
        provenance = 0
        split_fingerprint = hashlib.sha256()
        for sample in samples:
            if split == 'train' and sample.gt_verification and sample.gt_verification.startswith('provisional'):
                issues['provisional_train_gt'] += 1
            if split == 'train' and sample.quality_status in ('reject', 'rejected'):
                issues['rejected_train_crop'] += 1
            if not sample.image.is_file():
                issues["missing_image"] += 1
                continue
            try:
                digest, size, mode = pixel_digest(sample.image)
            except (OSError, ValueError, UnidentifiedImageError):
                issues["corrupt_image"] += 1
                continue
            payload = [digest, sample.text, sample.source_id, sample.proper_names]
            if sample.source_ids:
                payload.append(sample.source_ids)
            split_fingerprint.update(json.dumps(payload, ensure_ascii=False).encode())
            split_fingerprint.update(b"\n")
            keys = [("pixels", digest), ("path", str(sample.image))]
            sources = set(sample.source_ids)
            if sample.source_id:
                sources.add(sample.source_id)
            if sources:
                keys.extend(("source", source) for source in sources)
                provenance += 1
            duplicate_splits = {seen[k] for k in keys if k in seen and (k[0] != 'source' or seen[k] != split)}
            if duplicate_splits:
                issues["duplicate_or_leakage"] += 1
                if duplicate_splits - {split}:
                    report["overlap_records"] += 1
                    if split == "test" or "test" in duplicate_splits:
                        report["test_overlap_images"] += 1
            for key in keys:
                seen.setdefault(key, split)
            modes[mode] += 1
            dimensions.append(size)
            if settings:
                if min(size) < settings["min_short_edge"]:
                    issues["short_edge_too_small"] += 1
                if size[0] * size[1] > settings["max_pixels"]:
                    scale = (settings["max_pixels"] / (size[0] * size[1])) ** 0.5
                    if settings["resize_policy"] == "error":
                        issues["oversized_image"] += 1
                    elif int(min(size) * scale) < settings["min_short_edge"]:
                        issues["downscale_loses_detail"] += 1
        report["splits"][split] = {"count": len(samples), "issues": dict(issues), "modes": dict(modes),
            "provenance_records": provenance, "min_width": min((s[0] for s in dimensions), default=0),
            "min_height": min((s[1] for s in dimensions), default=0),
            "max_aspect_ratio": max((max(s) / min(s) for s in dimensions), default=0)}
        report["fingerprints"][split] = split_fingerprint.hexdigest()
    report["ok"] = not any(s["issues"] for s in report["splits"].values())
    return report

def prepare_image(path: Path, settings: dict) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if min(image.size) < settings.get("min_short_edge", 1):
        raise DatasetError("Image short edge below configured minimum")
    if image.width * image.height > settings["max_pixels"]:
        if settings.get("resize_policy", "error") == "error":
            raise DatasetError("Image exceeds max_pixels; choose downscale explicitly or increase pixel budget")
        scale = (settings["max_pixels"] / (image.width * image.height)) ** 0.5
        image.thumbnail((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
        if min(image.size) < settings.get("min_short_edge", 1):
            raise DatasetError("Downscale would destroy short-edge detail; increase max_pixels")
    return image

def messages(prompt: str, answer: str | None = None) -> list[dict]:
    result = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if answer is not None:
        result.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return result

class VisionDataset:
    def __init__(self, samples: list[Sample], prompt: str, settings: dict | int):
        self.samples, self.prompt = samples, prompt
        self.settings = settings if isinstance(settings, dict) else {"max_pixels": settings, "resize_policy": "downscale"}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return {"image": prepare_image(sample.image, self.settings), "text": sample.text,
                "record_id": f"{sample.split}:{sample.line}"}
