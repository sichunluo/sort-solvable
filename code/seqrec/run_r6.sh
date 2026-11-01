#!/bin/bash
# Round 6: fixed-target tie control under GTS (training), plus training-free controls (permutations, data access).
cd ~/seqrec/code
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mt
./run_queue.sh 0 jobs_r6.txt 2 &
export CUDA_VISIBLE_DEVICES=0
for b in beauty sports toys ml1m; do python run_pctm.py ${b}_tsx | tail -1; python run_baselines.py ${b}_tsx | tail -1; done
for b in beauty sports toys ml1m; do for k in 1 2 3 4; do python run_pctm.py ${b}_shuf$k | tail -1; python run_baselines.py ${b}_shuf$k | tail -1; done; done
for b in beauty sports toys ml1m; do python expert_access.py $b | tail -1; done
wait
echo R6_DONE
