"""Independent validation/test OCR metrics and privacy-conscious prediction artifacts."""
from __future__ import annotations

import csv
import gc
from pathlib import Path
import unicodedata

import jiwer
from .checkpoints import atomic_json, validate_adapter, versions
from .config import Config
from .dataset import load_split
from .inference import predict
from .models import load_adapter, load_base, require_supported
from .observability import new_run_id

def normalize(text: str, policy="nfc_whitespace", without_diacritics=False) -> str:
    if policy == "nfc_whitespace":
        text = " ".join(unicodedata.normalize("NFC", text).split())
    elif policy != "none":
        raise ValueError("Unknown normalization policy")
    if without_diacritics:
        text = "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)).replace("đ", "d").replace("Đ", "D")
    return text

def score(refs: list[str], predictions: list[str], policy="nfc_whitespace", without_diacritics=False) -> dict:
    if len(refs) != len(predictions) or not refs:
        raise ValueError("Nonempty equal-sized references and predictions required")
    refs = [normalize(x, policy, without_diacritics) for x in refs]
    predictions = [normalize(x, policy, without_diacritics) for x in predictions]
    return {"cer": jiwer.cer(refs, predictions), "wer": jiwer.wer(refs, predictions)}

def error_counts(ref, prediction, policy="nfc_whitespace"):
    alignment = jiwer.process_characters(normalize(ref, policy), normalize(prediction, policy))
    return {"insertions": alignment.insertions, "deletions": alignment.deletions, "substitutions": alignment.substitutions}

def benchmark(config: Config, split: str, limit: int | None, adapter: Path | None, compare_base: bool) -> dict:
    require_supported(config)
    if split not in {"test", "validation"}:
        raise ValueError("Use validation for model selection and test for final independent benchmark")
    if compare_base and adapter is None:
        raise ValueError("--compare-base requires an adapter")
    if adapter:
        validate_adapter(adapter, config)
    from .hardware import doctor
    report = doctor(config)
    if not report["ok"]:
        raise RuntimeError("GPU preflight failed: " + "; ".join(report["errors"]))
    samples = load_split(config.data_dir, split, limit)
    output = config.output_dir / config.model_key / new_run_id() / "metrics"
    output.mkdir(parents=True)
    records = [{"record_id": f"{s.split}:{s.line}", "image": str(s.image), "ground_truth": s.text} for s in samples]
    variants = ["base"] if adapter is None else (["base", "adapter"] if compare_base else ["adapter"])
    metrics, commits = {}, {}
    evaluation = config.section("evaluation")
    for variant in variants:
        model, processor = load_base(config) if variant == "base" else load_adapter(config, adapter)
        try:
            predictions = []
            commits[variant] = getattr(model.config, "_commit_hash", None)
            for sample, record in zip(samples, records):
                result = predict(model, processor, sample.image, config, details=True)
                record[variant] = result["text"]
                record[f"{variant}_hit_token_limit"] = result["hit_token_limit"]
                for key, value in error_counts(sample.text, result["text"], evaluation["normalization"]).items():
                    record[f"{variant}_{key}"] = value
                predictions.append(result["text"])
            refs = [s.text for s in samples]
            metrics[variant] = score(refs, predictions, evaluation["normalization"])
            if evaluation["without_diacritics"]:
                metrics[variant]["without_diacritics"] = score(refs, predictions, evaluation["normalization"], True)
        finally:
            del model, processor
            gc.collect()
            import torch
            torch.cuda.empty_cache()
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {"split": split, "samples": len(samples), "metrics": metrics, "model_id": config.model_id,
               "commits": commits, "inference": config.section("inference"), "preprocessing": config.section("data"),
               "normalization": evaluation, "versions": versions(), "output": str(output),
               "adapter": str(adapter) if adapter else None,
               "error_analysis": "Character insertions/deletions/substitutions do not establish hallucination intent; entity/modality analysis requires annotations"}
    atomic_json(output / "metrics.json", summary)
    return summary
