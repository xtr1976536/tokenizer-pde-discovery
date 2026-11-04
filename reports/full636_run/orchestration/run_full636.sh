#!/usr/bin/env bash
set -euo pipefail
source /mnt/c/Users/28678/Documents/Codex/2026-11-04/https-github-com-xtr1976536-tokenizer-pde/work/runtime_env.sh
exec "$PDE_PYTHON" -u -m scripts.pde_experiment run --scale 636 --device cuda --output "$TASK_ROOT/outputs/large636"
