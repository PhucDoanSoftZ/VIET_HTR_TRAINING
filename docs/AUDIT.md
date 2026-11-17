# Technical audit

Review of the initial repository: **2 Critical, 8 High, 5 Medium, 1 Low**. Locations below refer to the source before this audit; each issue has been addressed in code or the unsupported behavior removed. No model weights were downloaded and no GPU experiment was run.

| Severity | Initial location | Finding and correction |
|---|---|---|
| Critical | `train.py:66` | “Some labels masked” does not prove answer-only loss. Replaced with exact processor prompt-prefix masking, per-record target checks and CPU contract tests. |
| Critical | `train.py:63` | Sequence limit could truncate image/answer tokens. Truncation is now forbidden; image token counts must match image grids. |
| High | `dataset.py:103` | Only path-based test overlap was checked. Added decoded pixel hashes, within/across-split checks and optional original-source grouping. |
| High | `checkpoints.py:75` | No weight existence/integrity or completion marker. Added nonempty required files, complete inventory, checksums and marker written last. |
| High | `train.py:19` | Conflicting resume flags silently discarded warm start; actual restored progress was not checked. Conflicts now fail; restored optimizer/scheduler/global step are guarded. |
| High | `checkpoints.py:64` | Alternate base/quantization variants were accepted by model-size regex. Adapter base ID is now exact; loaded parameter topology/freeze policy is checked. |
| High | `__main__.py:4` | CLI errors returned exit code zero. The module entry point now propagates `main()` status. |
| High | `hardware.py:62` | VRAM of any GPU could pass while loader chose another GPU. Gate on the selected GPU and place the full model replica there; report all visible devices without pooling VRAM. |
| High | `train.py:77` | Smoke checked only forward; no LoRA gradient/update/save/reload evidence. Smoke now exercises exactly one optimizer step and isolated save/reload. |
| High | `scripts/upload_data.sh:12` | Remote path was interpolated into SSH shell syntax without validation. Host/path validation and protected rsync arguments added. |
| Medium | `config.py:63` | Unknown keys/nonfinite numbers/boolean numeric values and incompatible settings slipped through. Added strict fields/types/ranges/cross-field checks. |
| Medium | `dataset.py:145`, `inference.py:18` | Resize logic duplicated; automatic downscale could remove small strokes. Shared EXIF/RGB preparation with explicit downscale policy and detail checks. |
| Medium | `train.py:128` | Logs/manifests lacked run lifecycle, timing and resource/step diagnostics. Added JSONL/CSV events/metrics, safe batch traces and run manifests. |
| Medium | `train.py:118`, `evaluate.py:59` | Runs overwrote shared output paths; generation evaluation lacked exception cleanup. Added unique run IDs and GPU cleanup in finally blocks. |
| Medium | `evaluate.py:30`, `train.py:104` | Validation OCR metrics and error detail were unavailable. Added optional validation CER/WER before best/early-stop callbacks; character counts and token-limit flags. |
| Low | `scripts/setup_server.sh:10` | Separate installs could silently change dependencies; shell check previously only parsed the first script. Resolve together, export environment lock, parse every shell script individually. |

Pending runtime checks: exact server CUDA/PyTorch/Unsloth/PEFT compatibility, actual Qwen3-VL-32B processor and gradients, single-step checkpoint round trip, interruption/resume on GPU, representative resolution/VRAM, and measured validation/test CER/WER. CPU fixtures and backend mocks do not certify these. Pixel hashes cannot prove derivative grouping without provenance or semantic image/GT alignment. The 32B source/config path is implemented but not GPU validated. Distributed Qwen3-VL training, augmentation, curriculum, custom kernels and Gemma 4 remain unsupported pending evidence.

The dependency pins for Transformers/TRL/Datasets follow the official Unsloth notebook. Remaining CUDA-related versions are resolved on the actual server and exported, rather than claiming a GPU-tested lock from a Windows CPU environment. Setup fails on resolver/`pip check` errors. No quality or throughput superiority is claimed.
