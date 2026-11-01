#!/bin/bash
# Round 7b: after round 7, re-run every round-7 job that did not finish (killed by host OOM before the Data.times
# fix; train.py skips finished jobs), then redo all cross-evaluations so that they include the re-run seeds.
cd ~/seqrec/code
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mt
until grep -q R7_DONE ~/seqrec/logs/r7.log; do sleep 60; done
./run_queue.sh 0 jobs_r7_g0.txt 3 &
./run_queue.sh 1 jobs_r7_g1.txt 2 &
wait
echo R7B_TRAIN_DONE
export CUDA_VISIBLE_DEVICES=0
for b in beauty sports toys ml1m; do
  for tr in $b $b"_fr" $b"_fv"; do for h in $b $b"_fr" $b"_fv"; do python eval_cross.py $tr $h | tail -1; done; done
  python eval_cross.py $b"_day" $b"_day" | tail -1
  for tr in $b"_t" $b"_tsx"; do for h in $b"_t" $b"_tsx"; do
    python eval_cross.py $tr $h | tail -1; python eval_cross.py $tr $h $b"_tset" | tail -1
  done; done
done
echo R7B_DONE
