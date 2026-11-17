"""Supervised vision-to-text training using Trainer and exact OCR target labels."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import signal
import time

from .checkpoints import atomic_json, latest_valid_checkpoint, manifest, validate_adapter, validate_resume, write_manifest
from .collator import OCRCollator
from .config import Config
from .dataset import VisionDataset, check_data, load_split
from .hardware import doctor
from .models import _gpu_dependency, check_trainable, load_adapter, load_base, require_supported
from .observability import RunLog, callbacks, new_run_id, utc_now

def resolve_start(config, resume, warm_start):
    if resume and warm_start:
        raise ValueError("Choose --resume or --warm-start-from-adapter")
    if not resume and not warm_start:
        resume = Path("latest") if config.section("paths").get("resume_from") == "latest" else config.optional_path("resume_from")
        warm_start = config.optional_path("warm_start_from_adapter")
    if resume and warm_start:
        raise ValueError("Choose exact resume or adapter warm start")
    return resume, warm_start

def training_kwargs(config, preflight, run_dir, smoke=False):
    t = config.section("training")
    return dict(output_dir=str(run_dir), num_train_epochs=t["epochs"],
        per_device_train_batch_size=1 if smoke else t["batch_size"],
        per_device_eval_batch_size=t["batch_size"],
        gradient_accumulation_steps=1 if smoke else t["gradient_accumulation_steps"],
        max_steps=1 if smoke else -1, learning_rate=t["learning_rate"],
        warmup_steps=0 if smoke else t["warmup_steps"], warmup_ratio=0 if smoke else t["warmup_ratio"],
        optim=t["optimizer"], lr_scheduler_type=t["scheduler"], weight_decay=t["weight_decay"], max_grad_norm=t["max_grad_norm"],
        save_strategy="steps", eval_strategy="steps", save_steps=1 if smoke else t["save_steps"],
        eval_steps=1 if smoke else t["eval_steps"], logging_steps=1 if smoke else t["logging_steps"],
        save_total_limit=None, save_only_model=False, save_safetensors=True,
        load_best_model_at_end=not smoke, metric_for_best_model=t["best_metric"], greater_is_better=False,
        seed=t["seed"], data_seed=t["seed"], report_to="none", bf16=preflight["bf16"], fp16=not preflight["bf16"],
        gradient_checkpointing=bool(config.section("model")["gradient_checkpointing"]),
        remove_unused_columns=False, dataloader_num_workers=t["dataloader_workers"],
        dataloader_pin_memory=t["pin_memory"], logging_nan_inf_filter=False,
        prediction_loss_only=True, label_names=["labels"], push_to_hub=False)

def train(config: Config, resume: Path | None = None, warm_start: Path | None = None,
          dry_run=False, gpu_smoke=False) -> dict:
    require_supported(config)
    resume, warm_start = resolve_start(config, resume, warm_start)
    if dry_run and gpu_smoke:
        raise ValueError("Choose --dry-run or --gpu-smoke")
    if gpu_smoke and resume:
        raise ValueError("GPU smoke uses an isolated new run; clear resume_from first")
    if dry_run:
        if resume:
            # This is checkpoint verification and intentionally explicit; ordinary dry-run remains bounded.
            raise ValueError("Use exact resume validation on the server; CPU dry-run must not read the full dataset")
        if warm_start:
            validate_adapter(warm_start, config)
        report = check_data(config.data_dir, ("train", "validation"), limit=2, settings=config.section("data"))
        if not report["ok"]:
            raise ValueError("Dataset sample check failed: " + json.dumps(report))
        return {"dry_run": True, "data": report}
    preflight = doctor(config)
    if not preflight["ok"]:
        raise RuntimeError("GPU preflight failed: " + "; ".join(preflight["errors"]))
    # Test records are audited for leakage only, never passed to the model or selection metric.
    report = check_data(config.data_dir, ("train", "validation") if gpu_smoke else ("train", "validation", "test"), limit=2 if gpu_smoke else None, settings=config.section("data"))
    if not report["ok"]:
        raise ValueError("Dataset audit failed: " + json.dumps(report))
    if warm_start:
        validate_adapter(warm_start, config)
    # Avoid imports that patch Transformers until all inexpensive checks have passed.
    FastVisionModel = _gpu_dependency("unsloth").FastVisionModel
    # Distribution metadata alone does not catch broken transitive imports.
    # Resolve these before loading any model weights.
    for dependency in ("peft", "accelerate", "safetensors.torch", "bitsandbytes"):
        _gpu_dependency(dependency)
    import torch
    from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback, set_seed
    set_seed(config.section("training")["seed"])
    run_id = new_run_id()
    run_dir = config.checkpoint_dir / config.model_key / run_id
    output = config.output_dir / config.model_key / run_id
    log = RunLog(output, config.model_key, config.section("logging")["debug"])
    start = time.perf_counter()
    metadata = {"run_id": run_id, "started_at": utc_now(), "source_checkpoint": str(resume or warm_start) if resume or warm_start else None,
                "mode": "gpu_smoke" if gpu_smoke else "training", "hardware": preflight, "config": config.raw}
    atomic_json(output / "run_manifest.json", metadata)
    previous_signals = {}
    stage = "checkpoint_validation"
    try:
        if gpu_smoke:
            # Bounded smoke fingerprints never scan the real dataset.
            contract = {"schema": 2, "mode": "gpu_smoke", "config": config.raw}
        else:
            contract = manifest(config, report["fingerprints"])
        if resume == Path("latest"):
            resume, resume_info = latest_valid_checkpoint(config.checkpoint_dir / config.model_key, config, contract)
        else:
            resume_info = validate_resume(resume, config, contract) if resume else None
        metadata["source_checkpoint"] = str(resume or warm_start) if resume or warm_start else None
        atomic_json(output / "run_manifest.json", metadata)
        stage = "model_load"
        model, processor = load_adapter(config, resume or warm_start, True) if resume or warm_start else load_base(config, True)
        runtime = {"model_commit": getattr(model.config, "_commit_hash", None),
                   "chat_template_sha256": hashlib.sha256(processor.chat_template.encode()).hexdigest(),
                   "image_processor": processor.image_processor.to_dict(),
                   "trainable_shapes": {name: list(p.shape) for name, p in model.named_parameters() if p.requires_grad},
                   "cuda": torch.version.cuda, "bf16": preflight["bf16"]}
        runtime = json.loads(json.dumps(runtime))
        if resume_info and resume_info["contract"].get("runtime") != runtime:
            raise RuntimeError("Exact resume processor/model revision/parameter topology/CUDA mismatch")
        contract["runtime"] = runtime
        metadata["contract"] = contract
        atomic_json(output / "run_manifest.json", metadata)
        log.emit("model_loaded", component="model", **check_trainable(model, config))
        samples = load_split(config.data_dir, "train", limit=2 if gpu_smoke else None)
        validation = load_split(config.data_dir, "validation", limit=2 if gpu_smoke else None)
        dataset = VisionDataset(samples, config.section("inference")["prompt"], config.section("data"))
        validation_dataset = VisionDataset(validation, config.section("inference")["prompt"], config.section("data"))
        base_collator = OCRCollator(processor, config.section("inference")["prompt"], config.section("training")["max_seq_length"])
        trainer = None

        def traced_collator(records):
            batch = base_collator(records)
            if log.debug:
                log.emit("batch_trace", component="collator", records=[r["record_id"] for r in records],
                         step=trainer.state.global_step if trainer else 0,
                         shapes={k: list(v.shape) for k, v in batch.items()}, supervised_tokens=int((batch["labels"] != -100).sum()))
            return batch

        # Probe the actual model processor, not a text-only surrogate.
        traced_collator([dataset[0]])

        class OCRTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                inputs["use_cache"] = False
                result = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
                loss = result[0] if return_outputs else result
                if not torch.isfinite(loss.detach()):
                    raise RuntimeError(f"NaN/Inf loss at step {self.state.global_step}")
                return result

            def evaluation_loop(self, *args, **kwargs):
                output = super().evaluation_loop(*args, **kwargs)
                count = config.section("training")["validation_ocr_limit"]
                if count:
                    from .evaluate import score
                    from .inference import predict
                    was_training = self.model.training
                    FastVisionModel.for_inference(self.model)
                    try:
                        subset = validation[:count]
                        predictions = [predict(self.model, processor, s.image, config) for s in subset]
                        quality = score([s.text for s in subset], predictions, config.section("evaluation")["normalization"])
                        output.metrics.update({f"eval_{key}": value for key, value in quality.items()})
                        # Save per-record validation predictions as artifacts, never to event logs.
                        prediction_dir = log.folder / "validation"
                        prediction_dir.mkdir(exist_ok=True)
                        atomic_json(prediction_dir / f"step-{self.state.global_step}.json", {
                            "step": self.state.global_step, "metrics": quality,
                            "records": [{"record_id": f"validation:{s.line}", "ground_truth": s.text, "prediction": p} for s, p in zip(subset, predictions)]})
                    finally:
                        FastVisionModel.for_training(self.model, use_gradient_checkpointing=config.section("model")["gradient_checkpointing"])
                        self.model.train(was_training)
                return output

            def evaluate(self, *args, **kwargs):
                begun = time.perf_counter()
                metrics = super().evaluate(*args, **kwargs)
                log.emit("validation_timing", component="evaluation", step=self.state.global_step, duration_seconds=time.perf_counter() - begun)
                return metrics

            def _save_checkpoint(self, *args, **kwargs):
                begun = time.perf_counter()
                super()._save_checkpoint(*args, **kwargs)
                log.emit("checkpoint_written", component="checkpoint", step=self.state.global_step, duration_seconds=time.perf_counter() - begun)

        class Guard(TrainerCallback):
            def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
                grads = [p.grad for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
                if not grads or not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()) or not bool(torch.stack([torch.any(g != 0) for g in grads]).any()):
                    raise RuntimeError("LoRA gradients are missing, nonfinite or all zero")
                if gpu_smoke:
                    for p in model.parameters():
                        if p.requires_grad and p.grad is not None and bool(torch.any(p.grad != 0)):
                            self.probe_parameter, self.before = p, p.detach().clone()
                            break
                    log.emit("backward_verified", component="train", step=state.global_step)

            def on_step_end(self, args, state, control, **kwargs):
                if gpu_smoke and (not hasattr(self, "before") or torch.equal(self.before, self.probe_parameter.detach())):
                    raise RuntimeError("GPU smoke optimizer did not change the probed LoRA parameter")
                if stop_requested:
                    control.should_save = True
                    control.should_training_stop = True
                return control

        stop_requested = False

        def request_stop(signum, frame):
            nonlocal stop_requested
            stop_requested = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_signals[sig] = signal.signal(sig, request_stop)
        args = TrainingArguments(**training_kwargs(config, preflight, run_dir, gpu_smoke))
        handlers = [callbacks(log, config, contract, resume_info), Guard()]
        patience = config.section("training")["early_stopping_patience"]
        if patience and not gpu_smoke:
            handlers.append(EarlyStoppingCallback(early_stopping_patience=patience))
        trainer = OCRTrainer(model=model, processing_class=processor, data_collator=traced_collator,
            train_dataset=dataset, eval_dataset=validation_dataset, args=args, callbacks=handlers)
        stage = "training"
        result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        if resume_info and trainer.state.global_step <= resume_info["global_step"]:
            raise RuntimeError("Resume made no progress; check total epochs and source global step")
        if gpu_smoke:
            stage = "checkpoint_reload"
            checkpoint = run_dir / "checkpoint-1"
            validate_resume(checkpoint, config, contract)
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()
            reloaded, processor = load_adapter(config, checkpoint)
            with torch.no_grad():
                batch = base_collator([dataset[0]])
                device = reloaded.get_input_embeddings().weight.device
                reload_loss = reloaded(**{k: v.to(device) for k, v in batch.items()}).loss
            if not torch.isfinite(reload_loss):
                raise RuntimeError("Reload smoke produced nonfinite loss")
            summary = {"gpu_smoke": True, "optimizer_steps": 1, "save_reload_verified": True, "reload_loss": float(reload_loss)}
        else:
            best = trainer.state.best_model_checkpoint
            if best and not stop_requested:
                destination = output / "best_adapter"
                trainer.model.save_pretrained(destination, safe_serialization=True)
                processor.save_pretrained(destination)
                write_manifest(destination, config, contract)
            summary = {"global_step": result.global_step, "training_loss": result.training_loss,
                       "stopped_by_signal": stop_requested, "best_checkpoint": best,
                       "best_adapter": str(output / "best_adapter") if best and not stop_requested else None,
                       "checkpoints": str(run_dir)}
            atomic_json(output / "log_history.json", {"history": trainer.state.log_history})
        atomic_json(output / "summary.json", summary)
        metadata.update(ended_at=utc_now(), duration_seconds=time.perf_counter() - start, status="stopped" if stop_requested else "complete")
        atomic_json(output / "run_manifest.json", metadata)
        log.emit("run_complete", duration_seconds=time.perf_counter() - start)
        return {**summary, "output": str(output)}
    except (RuntimeError, ValueError, OSError) as exc:
        metadata.update(ended_at=utc_now(), status="failed", error_type=type(exc).__name__)
        atomic_json(output / "run_manifest.json", metadata)
        log.emit("run_failed", level="ERROR", component=stage, error_type=type(exc).__name__,
                 hint="Reduce batch/pixels after OOM; inspect checkpoint integrity, finite loss and processor contract")
        raise
    finally:
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
        log.close()
