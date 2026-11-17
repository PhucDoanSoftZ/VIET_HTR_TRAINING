#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi missing' >&2; exit 1; }
nvidia-smi
python_bin=${PYTHON_BIN:-python3}
[[ -d .venv ]] || "$python_bin" -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gpu.txt unsloth
python -m pip install -e .
python -m pip check
mkdir -p outputs checkpoints data/viet_htr
python -m pip freeze > outputs/environment-lock.txt
python -m htrtrain doctor --config "${1:-configs/qwen3vl_8b.yaml}"
