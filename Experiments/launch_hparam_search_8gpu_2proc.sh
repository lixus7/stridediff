#!/usr/bin/env bash
set -euo pipefail
# Usage:
#   bash Experiments/launch_hparam_search_8gpu_2proc.sh [name] [config_file] [run_tag] [gpu_list] [proc_per_gpu] [extra_eval_flags]
# Example:
#   bash Experiments/launch_hparam_search_8gpu_2proc.sh etth ./Config/etth.yaml etth_m10_16w "0,1,2,3,4,5,6,7" 2
#   bash Experiments/launch_hparam_search_8gpu_2proc.sh energy ./Config/energy.yaml energy_m10_disc "0,5,6,7" 1 "--eval_corr --eval_disc --eval_pred"
NAME="${1:-etth}"
CONFIG_FILE="${2:-./Config/etth.yaml}"
RUN_TAG="${3:-${NAME}_m10_16w_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST_CSV="${4:-0,1,2,3,4,5,6,7}"
PROC_PER_GPU="${5:-2}"
EXTRA_EVAL_FLAGS="${6:-}"

IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST_CSV}"
NUM_GPUS="${#GPU_IDS[@]}"
NUM_WORKERS=$((NUM_GPUS * PROC_PER_GPU))

echo "NAME=${NAME}"
echo "CONFIG_FILE=${CONFIG_FILE}"
echo "RUN_TAG=${RUN_TAG}"
echo "GPU_LIST=${GPU_LIST_CSV}"
echo "PROC_PER_GPU=${PROC_PER_GPU}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "EXTRA_EVAL_FLAGS=${EXTRA_EVAL_FLAGS}"

for gpu_pos in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$gpu_pos]}"
  for local_rank in $(seq 0 $((PROC_PER_GPU - 1))); do
    worker_id=$((gpu_pos * PROC_PER_GPU + local_rank))
    log_file="hparam_search_${RUN_TAG}_gpu${gpu}_w${worker_id}.log"

    CUDA_VISIBLE_DEVICES="${gpu}" nohup python -u Experiments/hparam_search.py \
      --name "${NAME}" \
      --config_file "${CONFIG_FILE}" \
      --milestone 10 \
      --gpu 0 \
      --workdir . \
      --search_mode random \
      --max_trials 5000 \
      --eval_iterations 3 \
      --eval_repeats 1 \
      --num_workers "${NUM_WORKERS}" \
      --worker_id "${worker_id}" \
      --run_tag "${RUN_TAG}" \
      ${EXTRA_EVAL_FLAGS} \
      > "${log_file}" 2>&1 &

    echo "started: gpu=${gpu}, worker_id=${worker_id}, log=${log_file}"
  done
done

echo "All workers launched."
echo "Check logs: ls -1 hparam_search_${RUN_TAG}_gpu*_w*.log"
