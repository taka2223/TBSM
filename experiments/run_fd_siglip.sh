#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="${FD_SIGLIP_CONFIG:-$repo_root/experiments/configs/imagenet_sit_l_siglip256/fd_siglip.yaml}"
python_bin="${TBSM_PYTHON:-python}"
output="${FD_SIGLIP_OUTPUT:-$repo_root/outputs/imagenet_sit_l_siglip256/fd_siglip_seed0}"
pid_file="$output/runner.pid"
runner_log="$output/runner.log"

command="${1:-start}"
mkdir -p "$output"

case "$command" in
  start)
    if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
      echo "FD-SigLIP is already running as PID $(<"$pid_file")"
      exit 1
    fi
    extra=()
    if [[ -n "${2:-}" ]]; then
      extra+=(--until "$2")
    fi
    nohup setsid env \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
      TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
      "$python_bin" -u "$repo_root/experiments/imagenet_fd_siglip.py" train \
      --config "$config" "${extra[@]}" </dev/null >>"$runner_log" 2>&1 &
    pid=$!
    printf '%s\n' "$pid" > "$pid_file"
    echo "FD-SigLIP started: PID=$pid log=$runner_log"
    ;;
  status)
    if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
      echo "running PID=$(<"$pid_file")"
    else
      echo "not running"
    fi
    tail -20 "$runner_log" 2>/dev/null || true
    ;;
  stop)
    if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
      kill -TERM -- "-$(<"$pid_file")"
      echo "sent SIGTERM to process group $(<"$pid_file")"
    else
      echo "not running"
    fi
    ;;
  *)
    echo "usage: $0 {start [until_step]|status|stop}" >&2
    exit 2
    ;;
esac
