#!/bin/bash
# PCTM re-tuning: full grid on the temporal splits, small grid around the frozen config on LOO. Usage: run_tune.sh <gpu>
cd ~/seqrec/code
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mt
export CUDA_VISIBLE_DEVICES=$1
for b in beauty sports toys ml1m; do python tune_pctm.py ${b}_t | tail -1; done
for b in beauty sports toys ml1m; do python tune_pctm.py $b --small | tail -1; done
echo TUNE_DONE
