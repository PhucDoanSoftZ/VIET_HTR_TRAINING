# Publishing this code repository

This repository may be public as source code. VietHTR-Line is gated and its dataset card lists CC BY-NC-ND 4.0; do not redistribute images, labels, ZIP archives or derived examples here without checking the dataset's terms and any privacy constraints. The trained checkpoint and OCR outputs also stay out of this repository. The Qwen3-VL base model has its own license; referencing its model ID does not include its weights.

The repository has no source-code license file yet. Public visibility lets people view and fork the code; it does not grant a broad reuse license. The owner can add a code license separately after choosing its terms.

Before each push, inspect the exact staged paths with `git diff --cached --name-only` and check the tracked file list with `git ls-files`. Stage only source, configs, tests, docs and shell scripts. `.gitignore` excludes `data/`, `checkpoints/`, `outputs/`, environment files and common image/label formats, but it does not remove files that were already committed in a previous Git history.

Data workflow:

1. Existing clean data: set `paths.data_dir` in YAML to its directory or parent and run `python -m htrtrain data-check --config configs/qwen3vl_8b.yaml`.
2. Optional standardized copy: call `python -m htrtrain data-prepare --source /path/to/source --dest /path/to/new/clean` explicitly. It validates all splits and copies image bytes and text unchanged into a new portable layout.
3. Train reads only the clean directory selected by YAML. Its image decoding and Qwen processor work at runtime because the model needs image tensors; it does not call `data-prepare` or change Ground Truth.

Dataset source and access terms: https://huggingface.co/datasets/5CD-AI/VietHTR-Line
