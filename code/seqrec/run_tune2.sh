#!/bin/bash
# PCTM re-tuning with a 162-config grid (3x3x3x3 around the frozen config, directional or direction-free counts).
cd ~/seqrec/code
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mt
export CUDA_VISIBLE_DEVICES=$1
for b in beauty sports toys ml1m; do python tune_pctm.py ${b}_t --small | tail -1; done
for b in beauty sports toys ml1m; do python tune_pctm.py ${b}_ts --small | tail -1; done
for b in beauty sports toys ml1m; do python tune_pctm.py $b --small | tail -1; done
echo TUNE2_DONE
