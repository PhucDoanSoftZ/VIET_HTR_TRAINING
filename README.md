# viet-htr-training

Vietnamese handwriting OCR with local JSONL data, Qwen3-VL 4B/8B QLoRA, strict resume, validation and independent test benchmarks. The implementation is ready for a GPU smoke test; the Qwen/Unsloth GPU stack and OCR quality have **not** been experimentally verified here. 32B and Gemma 4 remain explicitly unsupported.

## Develop with fixtures

```powershell
python -m pip install -e ".[dev]"
# Collator/backward CPU tests also require an existing CPU PyTorch installation.
python -m pytest tests -q
python -m htrtrain --help
```

Tests generate tiny images/JSONL in temporary directories. They do not scan `data/viet_htr/`, download model weights or start Qwen training. Processor/backend fixtures test contracts, not GPU compatibility. A small real CPU network tests the masked loss/backward/optimizer path.

## Paths and data

`data/viet_htr/`, `checkpoints/`, `outputs/` are prepared and ignored by Git. Copy clean data directly into the first directory or keep its `data_clean/` or `VietHTR_DataClean/` subdirectory. Each layout accepts `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `images/`. Do not keep multiple layouts at once.

If your dataset is **already clean**, copy it to `data/viet_htr/` and run `data-check`; no separate conversion is required. The current `data/viet_htr/VietHTR_DataClean/` layout is supported directly, including its `image_relpath` field. Training reads the clean directory from `paths.data_dir` in YAML. It does not invoke `data-prepare`.

For a separate source that needs a portable standard layout, explicitly run:

```bash
python -m htrtrain data-prepare --source /path/to/source --dest data/viet_htr/standard
python -m htrtrain data-check --data-dir data/viet_htr/standard
python -m htrtrain train --config configs/qwen3vl_8b.yaml --set paths.data_dir=data/viet_htr/standard --dry-run
```

`data-prepare` requires source train/validation/test JSONL and images. It validates schema, decodes images, detects duplicates/leakage, preserves every Ground Truth string and original image byte, writes canonical relative image paths, then publishes a new directory only after verifying it. It never overwrites an existing destination. No resize, image enhancement, augmentation or text normalization is applied to the stored dataset. The model still decodes images and applies its processor at training time, which is required for Qwen3-VL.

Each JSONL object needs `image_path` (aliases: `image`, `path`, `file_name`) and `text` (aliases: `transcription`, `label`, `ground_truth`, `gt`). Optional `source_id` or `original_id` identifies augmented derivatives of the same source image. Paths containing legacy Kaggle `images/` are mapped under the new dataset root. Traversal, symlink escapes, ambiguous files, blank records and empty labels are rejected. Original labels are never rewritten.

```bash
python -m htrtrain data-check --config configs/qwen3vl_8b.yaml
python -m htrtrain train --config configs/qwen3vl_8b.yaml --dry-run
```

The full checker detects duplicates within/across splits using decoded RGB pixels, paths and available source IDs. It reports modes, dimensions and aspect ratios. Pixel hashes cannot identify arbitrary transformed derivatives without source IDs. Image/label semantic alignment requires human inspection of actual data. `--limit 2` and CPU `--dry-run` read at most two records per requested split. Training audits all three splits for leakage, but only train/validation records reach the model; test never selects a checkpoint.

All local locations are configured under `paths:` in YAML. Relative paths resolve from the project root for files in `configs/`, otherwise from the config directory. `resume_from`, `warm_start_from_adapter`, and `adapter_dir` default to `null`. CLI path flags override YAML; `--set training.learning_rate=0.0001` overrides individual settings. Unknown fields, wrong types, nonfinite values and incompatible combinations fail before loading weights.

## Processing and training

One image preparation function and the model processor serve training, validation, test and page OCR. EXIF orientation is applied and images become RGB. `resize_policy: error` rejects oversized images; opt into `downscale` explicitly. Minimum short-edge checks prevent severe manual shrinkage. The Qwen processor still snaps/resizes to its native patch constraints: inspect actual line/page legibility during GPU smoke before choosing the pixel budget. Page images often need a larger `max_pixels`, token budget and `max_new_tokens`; defaults target line OCR, with no automatic page segmentation.

The collator uses the Qwen chat template, verifies that expanded image tokens match `image_grid_thw`, and compares the fully processed prompt prefix against the full sequence. Prompt, image and padding labels are `-100`; only answer/termination tokens contribute to loss. Overlength sequences fail; there is no token truncation or sequence packing. The standard Transformers `Trainer` consumes these labels directly, avoiding extra TRL dataset/tokenization transforms.

The loader uses one explicitly visible GPU, 4-bit loading, LoRA-only trainable parameters and configurable vision/language/attention/MLP selections. Original model IDs are quantized with NF4/double quantization; prequantized repositories retain their published quantization recipe. The runtime records model revision, processor template/settings, parameter topology, versions and CUDA precision. Exact adapter base IDs must match, including quantization variant.

The initial 4B/8B settings are experiments, not established optima: LR `2e-4`, batch `1`, accumulation `4`, rank/alpha `16`, dropout `0`, two epochs, linear schedule, warmup `50`, weight decay `0.01`, gradient clip `1.0`. Compare LR `1e-4`, frozen vision and cosine scheduling one change at a time using validation CER. BF16/FP16 follows actual GPU support. The loader requests SDPA and adds no explicit compile, CPU offload or distributed launch. Unsloth's own optimized kernels/compilation still require the GPU compatibility check. Dataloader workers default to `0`; increase only after measuring local image decode throughput. Debug tracing requires `0` workers.

`training.validation_ocr_limit: 0` selects best checkpoint by validation loss. For OCR-based selection, set a fixed validation subset size (or the entire validation count), `best_metric: eval_cer`, and optionally `early_stopping_patience`. CER/WER are added before Trainer best-model/early-stop callbacks. This generation pass adds cost at every evaluation. No augmentation is enabled or implemented: small rotation/contrast/scan noise should first be evaluated on real images, with train-only source grouping and an explicit seeded ablation. Bucketing/curriculum also await evidence from real data.

## SSH deployment

From the source machine, copy the repository without environments or artifacts:

```bash
rsync -av --exclude .venv --exclude data --exclude checkpoints --exclude outputs ./ USER@HOST:/path/viet-htr-training/
bash scripts/upload_data.sh /local/extracted/data USER@HOST /path/viet-htr-training
rsync -av /local/checkpoint-7500/ USER@HOST:/path/viet-htr-training/checkpoints/checkpoint-7500/
ssh USER@HOST
```

Alternatively copy the clean ZIP using `scp`, inspect it using `unzip -l`, then extract the trusted archive under `data/viet_htr/`. No script uploads data automatically. On Ubuntu install Python/venv, NVIDIA drivers, rsync and tmux through the server administrator. Select a supported CUDA PyTorch build for that server; `nvidia-smi` alone does not prove that the installed PyTorch has CUDA support.

```bash
cd /path/viet-htr-training
export CUDA_VISIBLE_DEVICES=GPU_INDEX_FROM_NVIDIA_SMI
bash scripts/setup_server.sh configs/qwen3vl_8b.yaml
source .venv/bin/activate
python -m htrtrain data-check --data-dir data/viet_htr
python -m htrtrain doctor --config configs/qwen3vl_8b.yaml
python -m htrtrain train --config configs/qwen3vl_8b.yaml --dry-run
python -m htrtrain train --config configs/qwen3vl_8b.yaml --gpu-smoke
```

Setup reuses `.venv`, resolves Unsloth and pinned notebook dependencies together, runs `pip check`, exports `outputs/environment-lock.txt`, then runs doctor. It never starts training. Exact Torch/Unsloth/Triton/driver compatibility still requires the server check; resolver failure must be fixed as a dependency problem rather than bypassed with `--no-deps`. Python 3.10–3.12 is the conservative server choice. Pin `model.revision` to the recorded commit and preserve the resolved environment lock before a long run.

Doctor checks PyTorch CUDA visibility, all GPU names/free/total VRAM, BF16, RAM, disk, write permissions, single-process/single-visible-GPU constraints and the documented Transformers/TRL versions. VRAM thresholds are configurable conservative gates, not OOM guarantees. GPU smoke reads at most two train/validation samples, runs **one** forward/backward/optimizer step, checks finite nonzero LoRA gradients and a changed parameter, seals a checkpoint, releases the original model, reloads it and checks finite forward loss. Smoke checkpoints have a distinct contract and cannot be resumed as full training.

Only after smoke passes, start a deliberately authorized full run:

```bash
tmux new -s htr
bash scripts/run_train.sh configs/qwen3vl_8b.yaml
# Detach: Ctrl-b d. Reconnect: tmux attach -t htr.
```

SIGINT (Ctrl-c) or SIGTERM requests save/stop at the next optimizer boundary. Do not use SIGKILL for a safe stop. A blocked kernel, OOM or power loss may prevent saving; resume the latest sealed checkpoint. New runs get unique folders under `checkpoints/<model>/<run-id>/` and `outputs/<model>/<run-id>/`, so earlier experiments are preserved.

## Checkpoints, logs and resume

Checkpoints contain adapter safetensors, Trainer/optimizer/scheduler/RNG state, optional AMP scaler, and `htr_manifest.json`. `COMPLETE.json` is written last with SHA256 of every saved file. Unsealed, missing, empty or modified files are rejected. The ordered dataset fingerprint uses decoded pixels and unchanged labels, so relocating Kaggle paths is allowed; changing images, GT, order, seed, batch, accumulation, LoRA, schedule, versions or processor/model runtime invalidates exact resume. This prevents silent optimizer reset; it does not guarantee bitwise GPU determinism.

Keep the best checkpoint alongside the resume checkpoint. Old absolute best-checkpoint paths are rebased in memory without editing source artifacts. Trainer must actually restore the expected global step, nonempty optimizer state and scheduler step. The previous Kaggle checkpoint lacks this repository's manifest/seal unless it was produced here, so its exact state cannot be certified retrospectively. Use explicit warm start if intended, matching its original base ID and LoRA config; do not invent a manifest to force acceptance.

```bash
python -m htrtrain train --config configs/qwen3vl_4b.yaml --resume checkpoints/qwen3vl_4b/RUN/checkpoint-7500
python -m htrtrain train --config configs/qwen3vl_4b.yaml --warm-start-from-adapter checkpoints/checkpoint-7500
rsync -av --exclude '*.tmp' checkpoints/ /mnt/persistent/viet-htr/checkpoints/
rsync -av --exclude '*.tmp' outputs/ /mnt/persistent/viet-htr/outputs/
```

Backup to persistent storage, ideally after a save/stop. A concurrent copy may be incomplete; validate the marker/checksums before resume and copy again if needed. Use only trusted local checkpoints. No automatic transfer, secret handling or Git push occurs.

`events.jsonl` contains timestamp, level, component, event, run ID, model ID, timing and step/epoch when applicable. `metrics.jsonl` and `metrics.csv` include Trainer loss/LR/gradient norm, step duration, effective samples/sec, estimated ETA and CUDA allocated/reserved/peak memory. Evaluation/checkpoint/sealing times are recorded separately. GPU utilization is not estimated from VRAM; no utilization claim is made. Debug batch traces include record IDs, tensor shapes and supervised token counts only. Predictions/GT are saved in explicit validation/benchmark artifacts, never in event logs. Console progress is concise. Run manifests record start/end/status, source checkpoint and resolved config/environment. Disk errors, nonfinite loss/gradients and resume mismatches propagate to a nonzero CLI exit code.

## Fair OCR evaluation

```bash
python -m htrtrain eval --config configs/qwen3vl_8b.yaml --split validation --limit 200 --adapter outputs/qwen3vl_8b/RUN/best_adapter --compare-base
python -m htrtrain eval --config configs/qwen3vl_8b.yaml --split test --limit 200 --adapter outputs/qwen3vl_8b/RUN/best_adapter --compare-base
python -m htrtrain ocr --config configs/qwen3vl_8b.yaml --image /data/page.png --adapter outputs/qwen3vl_8b/RUN/best_adapter
```

Base and adapter share configured base revision, prompt, image processing and deterministic decoding. Models load sequentially and GPU memory is released between variants. Original GT/predictions, character insertion/deletion/substitution counts and token-limit flags go to CSV; CER/WER and settings go to JSON. The default normalization is Unicode NFC plus whitespace collapse, preserving case, punctuation and accents; `none` leaves text unchanged before jiwer's standard word tokenization. Optional accent-free scores are reported separately. No spelling correction or OCR postprocessing is applied before scoring.

Entity-level errors, printed/handwritten/mixed modality comparisons and distinguishing visual mistakes from inferred additions require appropriate annotated evaluation data. Insertion/deletion counts flag additions/omissions; they do not establish hallucination intent. Never select hyperparameters using test results or assume more epochs improve CER.

See [audit findings](docs/AUDIT.md) for severity counts and remaining verification boundaries. APIs were checked against the [official Qwen3-VL notebook](https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_VL_%288B%29-Vision.ipynb), [Qwen3-VL processor documentation](https://huggingface.co/docs/transformers/v4.57.1/model_doc/qwen3_vl), [Trainer source at 4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/trainer.py), and [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization). Documentation verification does not replace a GPU experiment.
