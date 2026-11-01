#!/bin/bash
# Usage: run_ltr.sh <gpu>. LTR dump + fit for every dataset and SASRec seeds 1-3, waiting for missing checkpoints.
cd ~/seqrec/code
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mt
for seed in 1 2 3; do
  for ds in beauty sports toys ml1m; do
    [ -f ~/seqrec/results/$ds/ltr_s$seed.json ] && continue
    until [ -f ~/seqrec/results/$ds/sasrec_s$seed.json ]; do sleep 60; done
    [ -f ~/seqrec/results/$ds/ltr_test_s$seed.npz ] || CUDA_VISIBLE_DEVICES=$1 python ltr_dump.py $ds $seed
    ~/seqrec/ltr_env/bin/python -W ignore ltr_fit.py ~/seqrec/results $ds $seed | cut -c1-400
  done
done
