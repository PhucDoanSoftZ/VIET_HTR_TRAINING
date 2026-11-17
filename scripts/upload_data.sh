#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 LOCAL_DATA_DIR SSH_HOST REMOTE_REPO_DIR" >&2
  exit 2
fi
local_data=$1
host=$2
remote_repo=$3
[[ -d "$local_data" ]] || { echo "Dataset directory missing: $local_data" >&2; exit 2; }
[[ "$host" != -* && "$host" =~ ^[a-zA-Z0-9_.@-]+$ ]] || { echo 'Invalid SSH host' >&2; exit 2; }
[[ "$remote_repo" =~ ^/[a-zA-Z0-9_./-]+$ && "$remote_repo" != *..* ]] || { echo 'Use a safe absolute remote path without spaces or traversal' >&2; exit 2; }
ssh -- "$host" "mkdir -p -- '$remote_repo/data/viet_htr'"
rsync -av --protect-args --progress -- "$local_data/" "$host:$remote_repo/data/viet_htr/"
