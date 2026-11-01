#!/bin/bash
# Usage: run_queue.sh <gpu> <jobs file> <parallel>. Each job line is "<dataset> <train.py args...>".
# train.py skips a job whose result json already exists, so the queue can be restarted.
GPU=$1; JOBS=$2; PAR=${3:-3}
cd ~/seqrec/code
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mt
run() {
  log=~/seqrec/logs/$(echo "$*" | tr ' ' '_' | tr -d '-').log
  CUDA_VISIBLE_DEVICES=$GPU python train.py "$@" > "$log" 2>&1
}
export -f run; export GPU
grep -v '^#' "$JOBS" | xargs -P "$PAR" -L 1 bash -c 'run "$@"' _
