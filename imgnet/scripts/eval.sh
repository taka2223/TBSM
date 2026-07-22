export DM_MODEL_ZOO="${DM_MODEL_ZOO:-/path/to/your/model_zoo}"
export TORCH_HOME="${TORCH_HOME:-$DM_MODEL_ZOO}"
export HF_HOME="${HF_HOME:-$DM_MODEL_ZOO}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$DM_MODEL_ZOO/hub}"
export TIMM_HOME="${TIMM_HOME:-$DM_MODEL_ZOO/hub/checkpoints}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800

set -euo pipefail

torchrun --nproc_per_node="${NPROC_PER_NODE:-8}" --master_port="${MASTER_PORT:-29501}" -m metrics.eval "$@"
