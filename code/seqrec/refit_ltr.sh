#!/bin/bash
# Waits for run_ltr.sh to finish dumping, then refits every LTR model with the current ltr_fit.py.
cd ~/seqrec/code
while pgrep -f run_ltr.sh > /dev/null; do sleep 60; done
for seed in 1 2 3; do
  for ds in beauty sports toys ml1m; do
    [ -f ~/seqrec/results/$ds/ltr_test_s$seed.npz ] && ~/seqrec/ltr_env/bin/python -W ignore ltr_fit.py ~/seqrec/results $ds $seed | cut -c1-600
  done
done
echo ALL_LTR_DONE
