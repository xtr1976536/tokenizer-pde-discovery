#!/bin/bash
set -euo pipefail
cd /data1/users/liugang/src/eq-bpe/project
export CPATH="$PWD/.build-deps/usr/include/python3.12:$PWD/.build-deps/usr/include" PYTHONPATH=.:src PYTHONUTF8=1 TMPDIR="$PWD/tmp" XDG_CACHE_HOME="$PWD/.cache" MPLCONFIGDIR="$PWD/.cache/matplotlib" TORCH_EXTENSIONS_DIR="$PWD/.cache/torch_extensions" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
if [[ "${1:---check}" != --check && "${1:---check}" != --start ]]; then echo 'Usage: bash run_130.sh [--check|--start]'; exit 2; fi
.venv/bin/python -c 'import json; r=json.load(open("validation/environment_130/READY.json")); assert r["status"]=="READY" and r["settings"]["epochs"]==130; print("130-epoch environment preflight: READY")'
avail=$(df -PB1 /data1 | awk 'NR==2 {print $4}')
if (( avail < 107374182400 )); then echo 'Less than 100 GiB free; refusing launch'; exit 2; fi
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
if [ "${1:---check}" = --check ]; then echo 'Check only: no formal training started'; exit 0; fi
exec 9>.formal-run.lock
flock -n 9 || { echo 'An existing formal run holds the lock'; exit 2; }
GPU_UUID=$(nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$1==1 && $3==0 && $4==0 {print $2}')
if [ -z "$GPU_UUID" ]; then echo 'GPU 1 occupied; refusing launch'; exit 2; fi
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
mkdir -p runs/epochs130/{212,636} logs
printf '%s\n' "$GPU_UUID" > runs/epochs130/gpu_uuid.txt
for scale in 212 636; do
 avail=$(df -PB1 /data1 | awk 'NR==2 {print $4}')
 size=$(du -sb . | cut -f1)
 if (( avail < 107374182400 || size > 85899345920 )); then echo 'Space guard: stop before next scale'; exit 2; fi
 nice -n 10 .venv/bin/python -u -m scripts.pde_experiment run --scale "$scale" --device cuda --epochs 130 --output "$PWD/runs/epochs130/$scale" > "logs/formal_${scale}.log" 2>&1
 done
