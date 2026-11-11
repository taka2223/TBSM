#!/usr/bin/env bash
set -euo pipefail
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo 'The alternating launcher is retired. Use experiments/run_imagenet_arm.py with --arm and --gpu.' >&2
exit 1
