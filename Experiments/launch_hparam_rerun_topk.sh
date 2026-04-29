#!/usr/bin/env bash
set -euo pipefail
# Re-run top-K hyperparameter combinations from a previous report.
#
# Usage:
#   bash Experiments/launch_hparam_rerun_topk.sh [name] [config_file] [run_tag] [gpu_list] [proc_per_gpu] [report_path] [top_k] [extra_eval_flags]
# Example:
#   bash Experiments/launch_hparam_rerun_topk.sh stock ./Config/stocks.yaml stock_m10_16w_rerun "0,1,2,3" 1 \
#       OUTPUT/fmri/hparam_search/fmri_m10_16w/report.md 50 "--eval_corr --eval_disc --eval_pred"

NAME="${1:-stock}"
CONFIG_FILE="${2:-./Config/stocks.yaml}"
RUN_TAG="${3:-${NAME}_m10_16w_rerun_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST_CSV="${4:-0,1,2,3}"
PROC_PER_GPU="${5:-1}"
REPORT_PATH="${6:-OUTPUT/fmri/hparam_search/fmri_m10_16w/report.md}"
TOP_K="${7:-50}"
EXTRA_EVAL_FLAGS="${8:---eval_corr --eval_disc --eval_pred}"

# Strip trailing commas from GPU list to avoid empty array elements.
GPU_LIST_CSV="${GPU_LIST_CSV%,}"

IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST_CSV}"
NUM_GPUS="${#GPU_IDS[@]}"
NUM_WORKERS=$((NUM_GPUS * PROC_PER_GPU))

echo "NAME=${NAME}"
echo "CONFIG_FILE=${CONFIG_FILE}"
echo "RUN_TAG=${RUN_TAG}"
echo "GPU_LIST=${GPU_LIST_CSV}"
echo "PROC_PER_GPU=${PROC_PER_GPU}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "REPORT_PATH=${REPORT_PATH}"
echo "TOP_K=${TOP_K}"
echo "EXTRA_EVAL_FLAGS=${EXTRA_EVAL_FLAGS}"
echo "MODE=report_topk (re-run from report)"

for gpu_pos in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$gpu_pos]}"
  for local_rank in $(seq 0 $((PROC_PER_GPU - 1))); do
    worker_id=$((gpu_pos * PROC_PER_GPU + local_rank))
    log_file="hparam_rerun_${RUN_TAG}_gpu${gpu}_w${worker_id}.log"

    CUDA_VISIBLE_DEVICES="${gpu}" nohup python -u Experiments/hparam_search.py \
      --name "${NAME}" \
      --config_file "${CONFIG_FILE}" \
      --milestone 10 \
      --gpu 0 \
      --workdir . \
      --search_mode report_topk \
      --report_path "${REPORT_PATH}" \
      --report_top_k "${TOP_K}" \
      --report_dedup \
      --eval_iterations 5 \
      --eval_repeats 2 \
      --num_workers "${NUM_WORKERS}" \
      --worker_id "${worker_id}" \
      --run_tag "${RUN_TAG}" \
      ${EXTRA_EVAL_FLAGS} \
      > "${log_file}" 2>&1 &

    echo "started: gpu=${gpu}, worker_id=${worker_id}, log=${log_file}"
  done
done

echo "All workers launched."
echo "Check logs: ls -1 hparam_rerun_${RUN_TAG}_gpu*_w*.log"
