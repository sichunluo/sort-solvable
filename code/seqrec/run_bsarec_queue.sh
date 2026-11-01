#!/bin/bash
# Worker for jobs_bsarec.txt (or the job list given as $2): claims jobs with mkdir (shared by all workers), runs each
# on GPU $1. Usage: bash run_bsarec_queue.sh <gpu> [jobs file] ; run several workers to run jobs in parallel.
# Finished jobs are skipped.
GPU=$1; JOBS=${2:-$HOME/seqrec/code/jobs_bsarec.txt}
B=$HOME/seqrec/bsarec
mkdir -p $B/claims $B/output $B/logs
while read -r name args; do
  [ -z "$name" ] && continue
  [ -f "$B/output/${name}_test.npz" ] && continue
  mkdir "$B/claims/$name" 2>/dev/null || continue
  CUDA_VISIBLE_DEVICES=$GPU python $HOME/seqrec/code/run_bsarec.py $B/src $args --gpu_id $GPU \
    --data_dir $B/data/ --output_dir $B/output/ --train_name $name > $B/logs/$name.log 2>&1 \
    || { echo "FAILED $name" >> $B/logs/failed.txt; rmdir "$B/claims/$name"; }
done < "$JOBS"
