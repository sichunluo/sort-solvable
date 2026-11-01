#!/bin/bash
# Worker for jobs_esasrec.txt (one split per line): conv run, then the refit for the same number of steps.
# Claims jobs with mkdir so several workers can share the list. Usage: bash run_esasrec_queue.sh <gpu> [seed]
GPU=$1; SEED=${2:-32}
S=$HOME/seqrec
mkdir -p $S/esasrec_claims $S/esasrec_logs
while read -r ds; do
  [ -z "$ds" ] && continue
  [ -f "$S/results/$ds/esasrec_refit_s${SEED}_ranks.npz" ] && continue
  mkdir "$S/esasrec_claims/${ds}_s$SEED" 2>/dev/null || continue
  for mode in conv refit; do
    tag=$([ $mode = conv ] && echo esasrec || echo esasrec_refit)
    [ -f "$S/results/$ds/${tag}_s${SEED}_ranks.npz" ] && continue
    CUDA_VISIBLE_DEVICES=$GPU $S/esasrec_env/bin/python $S/code/run_esasrec.py $S/esasrec $S/data $S/results $ds $mode $SEED \
      > $S/esasrec_logs/${ds}_${mode}_s$SEED.log 2>&1 || { echo "FAILED $ds $mode $SEED" >> $S/esasrec_logs/failed.txt; break; }
  done
done < $S/code/jobs_esasrec.txt
