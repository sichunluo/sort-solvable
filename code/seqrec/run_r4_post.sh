#!/bin/bash
# After round 4 training: rolling evaluation on the tie-shuffled temporal splits and the stacking test. Usage: <gpu>
cd ~/seqrec/code
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mt
export CUDA_VISIBLE_DEVICES=$1
while pgrep -f "jobs_r4.txt" > /dev/null; do sleep 60; done
for b in beauty sports toys ml1m; do python stacking_test.py $b | tail -3; done
for b in beauty sports toys ml1m; do python eval_rolling.py ${b}_ts | grep -v "^{" | tail -1; python eval_rolling.py ${b}_t | grep -v "^{" | tail -1; done
echo R4_POST_DONE
