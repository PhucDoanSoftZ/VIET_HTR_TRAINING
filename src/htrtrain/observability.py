"""Privacy-preserving structured events, console summaries and metric exports."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
import uuid

def utc_now():
    return datetime.now(timezone.utc).isoformat()

class RunLog:
    def __init__(self, folder: Path, model: str, debug: bool = False):
        self.run_id = folder.name
        self.folder, self.model, self.debug = folder, model, debug
        folder.mkdir(parents=True, exist_ok=True)
        self.events = (folder / "events.jsonl").open("a", encoding="utf-8", buffering=1)
        self.metrics = (folder / "metrics.jsonl").open("a", encoding="utf-8", buffering=1)
        self.console = logging.getLogger("htrtrain")
        if not self.console.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
            self.console.addHandler(handler)
        self.console.setLevel(logging.INFO)
        self.console.propagate = False

    def emit(self, event: str, component="pipeline", level="INFO", **fields):
        # Callers send only IDs, numeric metrics, shapes and controlled diagnostics.
        record = {"timestamp": utc_now(), "level": level, "component": component,
                  "run_id": self.run_id, "model": self.model, "event": event, **fields}
        self.events.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        if event == "metrics":
            self.metrics.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        if event != "batch_trace" or self.debug:
            self.console.log(getattr(logging, level), "%s run=%s step=%s", event, self.run_id, fields.get("step", "-"))

    def close(self):
        self.events.close()
        self.metrics.close()
        with (self.folder / "metrics.jsonl").open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        if rows:
            keys = sorted(set().union(*(row.keys() for row in rows)))
            with (self.folder / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)

def new_run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]

def gpu_memory(torch):
    return {"gpu_allocated_gb": torch.cuda.memory_allocated() / 2**30,
            "gpu_reserved_gb": torch.cuda.memory_reserved() / 2**30,
            "gpu_peak_gb": torch.cuda.max_memory_allocated() / 2**30}

def callbacks(log, config, contract, resume_info=None):
    from transformers import TrainerCallback
    import torch
    from .checkpoints import guard_restored_state, prune_completed_checkpoints, seal_checkpoint, write_manifest

    class Events(TrainerCallback):
        def __init__(self):
            self.step_start = time.perf_counter()

        def on_train_begin(self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs):
            if resume_info:
                guard_restored_state(state, optimizer, lr_scheduler, resume_info)
            log.emit("train_begin", step=state.global_step, effective_batch_size=args.train_batch_size * args.gradient_accumulation_steps)

        def on_step_begin(self, args, state, control, **kwargs):
            self.step_start = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            elapsed = time.perf_counter() - self.step_start
            log.emit("metrics", component="train", step=state.global_step, epoch=state.epoch,
                     step_seconds=elapsed, samples_per_second=args.train_batch_size * args.gradient_accumulation_steps / max(elapsed, 1e-9),
                     estimated_eta_seconds=max(0, state.max_steps - state.global_step) * elapsed, **gpu_memory(torch))

        def on_log(self, args, state, control, logs=None, **kwargs):
            import math
            logs = logs or {}
            for key in ("loss", "eval_loss", "grad_norm"):
                if key in logs and not math.isfinite(float(logs[key])):
                    raise RuntimeError(f"Nonfinite {key} at step {state.global_step}")
            log.emit("metrics", component="trainer", step=state.global_step, epoch=state.epoch, **{k: v for k, v in logs.items() if k != "epoch"})

        def on_save(self, args, state, control, **kwargs):
            start = time.perf_counter()
            path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            write_manifest(path, config, contract)
            seal_checkpoint(path)
            removed = prune_completed_checkpoints(Path(args.output_dir), config.section("training")["save_total_limit"], state.best_model_checkpoint)
            log.emit("checkpoint_complete", component="checkpoint", step=state.global_step, sealing_seconds=time.perf_counter() - start)
            for old in removed:
                log.emit("checkpoint_pruned", component="checkpoint", step=state.global_step, path=str(old))

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            log.emit("evaluation_complete", component="evaluation", step=state.global_step, **(metrics or {}))

    return Events()
