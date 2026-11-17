#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
config=${1:-configs/qwen3vl_8b.yaml}
if [[ $# -gt 0 ]]; then shift; fi
python -m htrtrain doctor --config "$config"
exec python -m htrtrain train --config "$config" "$@"
