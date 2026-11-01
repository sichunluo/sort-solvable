#!/bin/bash
# Round 9 (third review): earlier GTS cutoffs, prospective 5-core filtering (Beauty), seed counts equalized across
# protocols, SASRec on two more fixed-target permutations, refit-epoch sensitivity, validation ranks for every
# baseline (validation-selected strongest baseline), and the cross-evaluations that use the new seeds.
cd ~/seqrec/code
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mt
until grep -q R8_DONE ~/seqrec/logs/r8.log; do sleep 60; done
for d in beauty_tp beauty_tpx; do mkdir -p ~/seqrec/results/$d; done
for b in beauty sports toys ml1m; do for v in _tq _tqx; do mkdir -p ~/seqrec/results/$b$v; done; done
./run_queue.sh 0 jobs_r9.txt 4 &
export CUDA_VISIBLE_DEVICES=0
for b in beauty sports toys ml1m; do for v in _tq _tqx; do python run_pctm.py $b$v | tail -1; done; done
for d in beauty_tp beauty_tpx; do python run_pctm.py $d | tail -1; done
for b in beauty sports toys ml1m; do
  for v in "" _shuf _fr _fr1 _fr2 _fv _day _t _tsx _tq _tqx _ts _tl0 _tl1; do python run_baselines.py $b$v | tail -1; done
done
for d in beauty_tp beauty_tpx; do python run_baselines.py $d | tail -1; done
wait
echo R9_TRAIN_DONE
for b in beauty sports toys ml1m; do
  python eval_cross.py $b"_fr" $b"_fr" | tail -1; python eval_cross.py $b"_day" $b"_day" | tail -1
  python eval_cross.py $b"_t" $b"_t" $b"_tset" | tail -1; python eval_cross.py $b"_tsx" $b"_tsx" $b"_tset" | tail -1
  python eval_cross.py $b $b | tail -1
done
echo R9_DONE
