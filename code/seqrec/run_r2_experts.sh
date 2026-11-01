#!/bin/bash
# Round-2 expert-only evaluations. Usage: run_r2_experts.sh <gpu>
cd ~/seqrec/code
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mt
export CUDA_VISIBLE_DEVICES=$1
for b in beauty sports toys ml1m; do python run_pctm.py ${b}_shuf | tail -1; done
for b in beauty sports toys ml1m; do for k in 0 1; do python run_pctm.py ${b}_tl$k | tail -1; done; done
for b in beauty sports toys ml1m; do for v in "" _t _shuf; do python run_baselines.py $b$v | tail -1; done; done
for b in beauty sports toys ml1m; do python ensemble.py $b | tail -1; done
for b in beauty sports toys ml1m; do for v in "" _t; do python tune_pctm.py $b$v | tail -1; done; done
echo R2_EXPERTS_DONE
