#!/bin/bash
# Round 7 (reviewer controls): order-controlled LOO/GTS variants, set-valued targets, val-fitted PoE weight,
# more permutations/seeds, ML-20M refit and tie shuffle; then cross-evaluation of train order x history order x targets.
cd ~/seqrec/code
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mt
for b in beauty sports toys ml1m; do for v in _fr _fv _day _tset; do mkdir -p ~/seqrec/results/$b$v; done; done
mkdir -p ~/seqrec/results/ml20m_shuf
./run_queue.sh 0 jobs_r7_g0.txt 3 &
./run_queue.sh 1 jobs_r7_g1.txt 2 &
export CUDA_VISIBLE_DEVICES=0
# experts: new variants, and every variant again with SHA-256 tie-breaking for MC / SeqRules
for b in beauty sports toys ml1m; do
  for v in _fr _fv _day _shuf _shuf1 _shuf2 _shuf3 _shuf4 _t _ts _tsx _tl0 _tl1; do
    python run_pctm.py $b$v | tail -1; python run_baselines.py $b$v | tail -1
  done
done
python run_pctm.py ml20m_shuf | tail -1; python run_baselines.py ml20m_shuf | tail -1
wait
echo R7_TRAIN_DONE
for b in beauty sports toys ml1m; do
  for tr in $b $b"_fr" $b"_fv"; do for h in $b $b"_fr" $b"_fv"; do python eval_cross.py $tr $h | tail -1; done; done
  python eval_cross.py $b"_day" $b"_day" | tail -1
  for tr in $b"_t" $b"_tsx"; do for h in $b"_t" $b"_tsx"; do
    python eval_cross.py $tr $h | tail -1; python eval_cross.py $tr $h $b"_tset" | tail -1
  done; done
done
echo R7_DONE
