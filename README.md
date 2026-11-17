# viet-htr-training

Vietnamese handwriting OCR with local JSONL data and Qwen3-VL 32B QLoRA as the primary experiment; 8B is the fallback. The 4B config remains available for compatibility. Source, config and CPU contracts are ready for a GPU smoke test; 32B GPU loading, training stability, VRAM fit and OCR quality have **not** been experimentally verified here. Gemma 4 remains unsupported.

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

Each JSONL object needs `image_path` (aliases: `image`, `path`, `file_name`) and `text` (aliases: `transcription`, `label`, `ground_truth`, `gt`). Optional `source_id` or `original_id` identifies augmented derivatives of the same source image. Optional `proper_names` is a list of annotated name substrings for evaluation. Paths containing legacy Kaggle `images/` are mapped under the new dataset root. Traversal, symlink escapes, ambiguous files, blank records and empty labels are rejected. Original labels are never rewritten.

```bash
python -m htrtrain data-check --config configs/qwen3vl_8b.yaml
python -m htrtrain train --config configs/qwen3vl_8b.yaml --dry-run
```

The full checker detects duplicates within/across splits using decoded RGB pixels, paths and available source IDs. It reports modes, dimensions and aspect ratios. Pixel hashes cannot identify arbitrary transformed derivatives without source IDs. Image/label semantic alignment requires human inspection of actual data. `--limit 2` and CPU `--dry-run` read at most two records per requested split. Training audits all three splits for leakage, but only train/validation records reach the model; test never selects a checkpoint.

All local locations are configured under `paths:` in YAML. Relative paths resolve from the project root for files in `configs/`, otherwise from the config directory. `resume_from`, `warm_start_from_adapter`, and `adapter_dir` default to `null`. CLI path flags override YAML; `--set training.learning_rate=0.0001` overrides individual settings. Unknown fields, wrong types, nonfinite values and incompatible combinations fail before loading weights.

## Processing and training

One image preparation function and the model processor serve training, validation, test and page OCR. EXIF orientation is applied and images become RGB. `resize_policy: error` rejects oversized images; opt into `downscale` explicitly. Minimum short-edge checks prevent severe manual shrinkage. The Qwen processor still snaps/resizes to its native patch constraints: inspect actual line/page legibility during GPU smoke before choosing the pixel budget. Page images often need a larger `max_pixels`, token budget and `max_new_tokens`; defaults target line OCR, with no automatic page segmentation.

The collator uses the Qwen chat template, verifies that expanded image tokens match `image_grid_thw`, and compares the fully processed prompt prefix against the full sequence. Prompt, image and padding labels are `-100`; only answer/termination tokens contribute to loss. Overlength sequences fail; there is no token truncation or sequence packing. The standard Transformers `Trainer` consumes these labels directly, avoiding extra TRL dataset/tokenization transforms.

The loader places one complete 4-bit model replica on the selected CUDA device, even when other GPUs are visible. Doctor reports free/total VRAM for every GPU and gates on the **selected** GPU; it never pools memory. This repository does not launch distributed training: Unsloth documents DDP/FSDP generally, but its Qwen3-VL 32B path has not been validated here. Original model IDs are quantized with NF4/double quantization; prequantized repositories retain their published quantization recipe. LoRA-only trainable parameters and configurable vision/language/attention/MLP selections are checked after loading. The runtime records model revision, processor template/settings, parameter topology, versions and CUDA precision. Exact adapter base IDs must match, including quantization variant.

The initial 32B hypothesis uses LR `2e-5`, rank/alpha `8/16`, batch `1`, accumulation `8`, frozen vision layers, two epochs, linear schedule, warmup `50`, and save/eval every `250` optimizer steps. The 8B fallback uses LR `5e-5` with the same rank, batch and vision-freeze starting point. The prior 4B Kaggle result motivates these choices but does not establish an optimum for either model. Try `--set training.learning_rate=0.00005` for a controlled 32B comparison after a successful smoke test. BF16/FP16 follows actual GPU support. The loader requests SDPA and adds no explicit CPU offload or resolution reduction. Dataloader workers default to `0`; increase only after measuring local image decode throughput.

The 32B/8B YAML files use `validation_ocr_limit: 128` for a fixed validation OCR subset and initially select best checkpoints by **validation loss**. After generation evaluation works on the target GPU, set `training.best_metric: eval_cer`; CER/WER are added before Trainer best-model/early-stop callbacks. Generation adds cost at every evaluation. The `test` split stays out of model selection. No augmentation is enabled or implemented: small rotation/contrast/scan noise should first be evaluated on real images, with train-only source grouping and an explicit seeded ablation.

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
bash scripts/setup_server.sh configs/qwen3vl_32b.yaml
source .venv/bin/activate
python -m htrtrain data-check --data-dir data/viet_htr
python -m htrtrain doctor --config configs/qwen3vl_32b.yaml
python -m htrtrain train --config configs/qwen3vl_32b.yaml --dry-run
python -m htrtrain train --config configs/qwen3vl_32b.yaml --gpu-smoke
```

Setup reuses `.venv`, resolves Unsloth and pinned notebook dependencies together, runs `pip check`, exports `outputs/environment-lock.txt`, then runs doctor. It never starts training. Exact Torch/Unsloth/Triton/driver compatibility still requires the server check; resolver failure must be fixed as a dependency problem rather than bypassed with `--no-deps`. Python 3.10–3.12 is the conservative server choice. Pin `model.revision` to the recorded commit and preserve the resolved environment lock before a long run.

Doctor checks PyTorch CUDA visibility, every GPU's free/total VRAM, selected GPU free VRAM, BF16, RAM, disk, write permissions, single-process execution and the documented Transformers/TRL versions. The 32B YAML requires 60 GiB free on the selected device as a conservative gate, **not** an OOM guarantee. If unavailable, choose the 8B YAML; multiple smaller cards do not meet this single-replica gate. GPU smoke reads at most two train/validation samples, runs **one** forward/backward/optimizer step, checks finite nonzero LoRA gradients and a changed parameter, performs bounded validation generation/CER, seals a checkpoint, releases the original model, reloads it and checks finite forward loss. Smoke checkpoints have a distinct contract and cannot be resumed as full training.

Only after smoke passes, start a deliberately authorized full run:

```bash
tmux new -s htr
bash scripts/run_train.sh configs/qwen3vl_32b.yaml
# Detach: Ctrl-b d. Reconnect: tmux attach -t htr.
```

SIGINT (Ctrl-c) or SIGTERM requests save/stop at the next optimizer boundary. Do not use SIGKILL for a safe stop. A blocked kernel, OOM or power loss may prevent saving; resume the latest sealed checkpoint. New runs get unique folders under `checkpoints/<model>/<run-id>/` and `outputs/<model>/<run-id>/`, so earlier experiments are preserved.

## Checkpoints, logs and resume

Checkpoints contain adapter safetensors, Trainer/optimizer/scheduler/RNG state, optional AMP scaler, and `htr_manifest.json`. `COMPLETE.json` is written last with SHA256 of every saved file. Unsealed, missing, empty or modified files are rejected. The ordered dataset fingerprint uses decoded pixels, unchanged labels and optional proper-name annotations. Full training reuses hashes from its initial dataset audit instead of decoding train/validation images a second time. Changing images, GT, order, seed, batch, accumulation, LoRA, schedule, versions or processor/model runtime invalidates exact resume. Trainer's automatic rotation is disabled; the configured `save_total_limit` is applied **after** sealing, retaining the best and latest sealed checkpoints. This prevents silent optimizer reset; it does not guarantee bitwise GPU determinism.

Keep the best checkpoint alongside the resume checkpoint. Old absolute best-checkpoint paths are rebased in memory without editing source artifacts. Trainer must actually restore the expected global step, nonempty optimizer state and scheduler step. The previous Kaggle 4B checkpoint is outside this 32B/8B training plan. New training starts from the selected Instruct base model. `--resume latest` finds the most recently sealed compatible checkpoint under `checkpoints/<model>/`; if none exists, it fails instead of starting at step 0.

```bash
python -m htrtrain train --config configs/qwen3vl_32b.yaml --resume latest
python -m htrtrain train --config configs/qwen3vl_32b.yaml --resume checkpoints/qwen3vl_32b/RUN/checkpoint-7500
rsync -av --exclude '*.tmp' checkpoints/ /mnt/persistent/viet-htr/checkpoints/
rsync -av --exclude '*.tmp' outputs/ /mnt/persistent/viet-htr/outputs/
```

Backup to persistent storage, ideally after a save/stop. A concurrent copy may be incomplete; validate the marker/checksums before resume and copy again if needed. Use only trusted local checkpoints. No automatic transfer, secret handling or Git push occurs.

`events.jsonl` contains timestamp, level, component, event, run ID, model ID, timing and step/epoch when applicable. `metrics.jsonl` and `metrics.csv` include Trainer loss/LR/gradient norm, step duration, effective samples/sec, estimated ETA and CUDA allocated/reserved/peak memory. Evaluation/checkpoint/sealing times are recorded separately. GPU utilization is not estimated from VRAM; no utilization claim is made. Debug batch traces include record IDs, tensor shapes and supervised token counts only. Predictions/GT are saved in explicit validation/benchmark artifacts, never in event logs. Console progress is concise. Run manifests record start/end/status, source checkpoint and resolved config/environment. Disk errors, nonfinite loss/gradients and resume mismatches propagate to a nonzero CLI exit code.

## Fair OCR evaluation

```bash
python -m htrtrain eval --config configs/qwen3vl_32b.yaml --split validation --limit 200 --adapter outputs/qwen3vl_32b/RUN/best_adapter --compare-base
python -m htrtrain eval --config configs/qwen3vl_32b.yaml --split test --adapter outputs/qwen3vl_32b/RUN/best_adapter --compare-base
python -m htrtrain ocr --config configs/qwen3vl_32b.yaml --image /data/giay_nhuong_dat.png --adapter outputs/qwen3vl_32b/RUN/best_adapter
```

Base and adapter share configured base revision, prompt, image processing and deterministic decoding. Models load sequentially and GPU memory is released between variants. Original GT/predictions, character insertion/deletion/substitution counts, digit-sequence checks, optional annotated proper-name exact matches and token-limit flags go to CSV; CER/WER and aggregate diagnostics go to JSON. The 32B/8B configs also report accent-free CER/WER separately. Optional `proper_names: ["Tên riêng"]` annotations in JSONL enable proper-name reporting; the metric remains null without them. The default normalization is Unicode NFC plus whitespace collapse, preserving case, punctuation and accents. No spelling correction or OCR postprocessing is applied before scoring.

Printed/handwritten/mixed modality comparisons and distinguishing visual mistakes from inferred additions require annotated evaluation data. Insertion/deletion counts flag additions/omissions; they do not establish hallucination intent. Full-page land-transfer documents may exceed the line-OCR image/token budget; tune YAML on a separate validation set and inspect qualitative output. Never select hyperparameters using test results or assume more epochs improve CER.

See [audit findings](docs/AUDIT.md) for severity counts and remaining verification boundaries. Model identity and API choices were checked against the [Qwen 32B model card](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct), [Unsloth 32B 4-bit model card](https://huggingface.co/unsloth/Qwen3-VL-32B-Instruct-unsloth-bnb-4bit), [Unsloth Qwen3-VL guide](https://unsloth.ai/docs/models/qwen3-vl-how-to-run-and-fine-tune), [vision fine-tuning flags](https://unsloth.ai/docs/basics/vision-fine-tuning), [Qwen3-VL processor documentation](https://huggingface.co/docs/transformers/v4.57.1/model_doc/qwen3_vl), [Trainer checkpoint documentation](https://huggingface.co/docs/transformers/main_classes/trainer), and [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization). Documentation verification does not replace a GPU experiment.
