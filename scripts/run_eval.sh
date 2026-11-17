#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
config=${1:-configs/qwen3vl_8b.yaml}
if [[ $# -gt 0 ]]; then shift; fi
exec python -m htrtrain eval --config "$config" --split test "$@"
