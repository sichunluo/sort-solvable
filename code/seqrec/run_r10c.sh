#!/bin/bash
# Round 10c: the remaining three-seed fusion diagnostics. Analyses that reload saved LOO checkpoints use SASRec seeds
# 1-3 (seed-0 checkpoints were not kept), as LTR, stacking and the seed ensemble already do: joint PoE seed 3 (for
# the re-tuned-beta row, which the seed-0 run predates), the weight-scale diagnostic for seeds 2-3 and the contextual
# gate for seeds 1-3 on all four datasets.
cd ~/seqrec/code
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mt
waitmem() { until [ "$(nvidia-smi -i $1 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge "$2" ]; do sleep 120; done; }
R=~/seqrec/results; L=~/seqrec/logs
poe3() {  # poe3 <gpu> <dataset>...: PoE seed 3, retried once if it fails
  for b in "${@:2}"; do
    for a in 1 2; do
      [ -f $R/$b/poe_luo1_s3.json ] && break
      waitmem $1 6000
      CUDA_VISIBLE_DEVICES=$1 python train.py $b --method poe --luo 1 --seed 3 > $L/${b}_method_poe_luo_1_seed_3.log 2>&1
    done
  done
}
poe3 0 beauty sports & p0=$!
poe3 1 toys ml1m & p1=$!
wait $p0 $p1
for b in beauty sports toys ml1m; do
  for s in 2 3; do
    [ -f $R/$b/diag_weights_s$s.json ] && continue
    waitmem 0 6000; CUDA_VISIBLE_DEVICES=0 python diag_weights.py $b $s 2>&1 | grep "std(beta"
  done
  for s in 1 2 3; do
    [ -f $R/$b/gate_s$s.json ] && continue
    waitmem 0 6000; CUDA_VISIBLE_DEVICES=0 python gate.py $b --seed $s > $L/gate_${b}_s$s.log 2>&1; tail -1 $L/gate_${b}_s$s.log
  done
done
echo R10C_DONE
