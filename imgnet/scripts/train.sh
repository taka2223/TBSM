#!/bin/bash
set -euo pipefail
export DM_MODEL_ZOO="${DM_MODEL_ZOO:-/path/to/your/model_zoo}"
export TORCH_HOME="${TORCH_HOME:-$DM_MODEL_ZOO}"
export HF_HOME="${HF_HOME:-$DM_MODEL_ZOO}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$DM_MODEL_ZOO/hub}"
export TIMM_HOME="${TIMM_HOME:-$DM_MODEL_ZOO/hub/checkpoints}"
if [ "$#" -ne 1 ]; then
    echo "Usage: bash scripts/train.sh configs/xxx.yaml"
    exit 1
fi
config="$1"
if [ ! -f "$config" ]; then
    echo "Config not found: $config"
    exit 1
fi
nproc_per_node="$(python -c 'import sys, yaml; cfg=yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8")) or {}; value=cfg["runtime"]["nproc_per_node"]; assert isinstance(value, int) and value >= 1; print(value)' "$config")"
deterministic="$(python -c 'import sys, yaml; cfg=yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8")) or {}; value=cfg["runtime"]["deterministic"]; assert isinstance(value, bool); print(str(value).lower())' "$config")"
case "$nproc_per_node" in
    ''|*[!0-9]*)
        echo "runtime.nproc_per_node must be a positive integer, got: $nproc_per_node"
        exit 1
        ;;
esac
if [ "$nproc_per_node" -lt 1 ]; then
    echo "runtime.nproc_per_node must be >= 1, got: $nproc_per_node"
    exit 1
fi
if [ "$deterministic" = "true" ]; then
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi
if [ "$nproc_per_node" -eq 1 ]; then
    python main.py --config "$config"
elif [ -n "${MASTER_PORT:-}" ]; then
    torchrun --nproc_per_node="$nproc_per_node" --master_port="$MASTER_PORT" main.py --config "$config"
else
    torchrun --standalone --nproc_per_node="$nproc_per_node" main.py --config "$config"
fi
