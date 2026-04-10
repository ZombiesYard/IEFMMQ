#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LABEL_STUDIO_VENV="${LABEL_STUDIO_VENV:-$HOME/venvs/label-studio-wsl}"
LABEL_STUDIO_HOST="${LABEL_STUDIO_HOST:-0.0.0.0}"
LABEL_STUDIO_PORT="${LABEL_STUDIO_PORT:-8080}"
LABEL_STUDIO_DATA_ROOT="${LABEL_STUDIO_DATA_ROOT:-$REPO_ROOT/.label_studio}"

if [[ ! -f "$LABEL_STUDIO_VENV/bin/activate" ]]; then
  echo "Label Studio venv not found at: $LABEL_STUDIO_VENV" >&2
  echo "Create it first, for example:" >&2
  echo "  python3 -m venv ~/venvs/label-studio-wsl" >&2
  echo "  source ~/venvs/label-studio-wsl/bin/activate" >&2
  echo "  python -m pip install --upgrade pip wheel setuptools" >&2
  echo "  pip install label-studio" >&2
  exit 1
fi

mkdir -p \
  "$LABEL_STUDIO_DATA_ROOT/xdg_data" \
  "$LABEL_STUDIO_DATA_ROOT/xdg_config" \
  "$LABEL_STUDIO_DATA_ROOT/xdg_cache"

export XDG_DATA_HOME="$LABEL_STUDIO_DATA_ROOT/xdg_data"
export XDG_CONFIG_HOME="$LABEL_STUDIO_DATA_ROOT/xdg_config"
export XDG_CACHE_HOME="$LABEL_STUDIO_DATA_ROOT/xdg_cache"

source "$LABEL_STUDIO_VENV/bin/activate"
exec label-studio start --host "$LABEL_STUDIO_HOST" --port "$LABEL_STUDIO_PORT"
