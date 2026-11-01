#!/bin/bash
# PCTM/MC on the leaky folds, then PCTM re-tuning on the temporal splits. Usage: run_experts_time.sh <gpu>
cd ~/seqrec/code
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mt
for b in beauty sports toys ml1m; do for k in 0 1; do CUDA_VISIBLE_DEVICES=$1 python run_pctm.py ${b}_tl$k | tail -1; done; done
for b in beauty sports toys ml1m; do CUDA_VISIBLE_DEVICES=$1 python tune_pctm.py ${b}_t | tail -1; done
echo EXPERTS_TIME_DONE
