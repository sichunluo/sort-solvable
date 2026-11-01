#!/bin/bash
# Round 8 (second review): conventional SASRec and GRU4Rec under the fixed-target order control, GRU4Rec with random
# GTS training ties, and two more fixed-target permutations for the training-free models.
cd ~/seqrec/code
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mt
for b in beauty sports toys ml1m; do for v in _fr1 _fr2; do mkdir -p ~/seqrec/results/$b$v; done; done
./run_queue.sh 0 jobs_r8.txt 3 &
export CUDA_VISIBLE_DEVICES=0
for b in beauty sports toys ml1m; do for v in _fr1 _fr2; do
  python run_pctm.py $b$v | tail -1; python run_baselines.py $b$v | tail -1
done; done
wait
echo R8_DONE
