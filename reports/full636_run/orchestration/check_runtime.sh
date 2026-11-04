#!/usr/bin/env bash
set -euo pipefail
source /mnt/c/Users/28678/Documents/Codex/2026-11-04/https-github-com-xtr1976536-tokenizer-pde/work/runtime_env.sh
if [[ "${1:-preflight}" == test ]]; then
  exec "$PDE_PYTHON" -m pytest -q --basetemp "$TASK_ROOT/work/pytest-linux" tests
fi
exec "$PDE_PYTHON" -u "$TASK_ROOT/work/preflight636.py"
