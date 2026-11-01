#!/bin/bash
# BERT4Rec (train.py --arch bert): cloze masking probability, last-item-only share and architecture (SASRec's or
# Sun et al.'s) chosen on validation with seed 0, then seeds 0-4 on LOO (conventional, refit), the fixed-target order
# splits (_fr, _fv) and the GTS (_t, _tsx), then the order cross-evaluation over {ds, ds_fr, ds_fv}^2. Patience 40
# (SASRec: 20) because the cloze objective improves validation slowly.
cd ~/seqrec/code
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
S=~/seqrec; R=$S/results; L=$S/logs/bert; CL=$S/bert_claims
mkdir -p $CL $L
MARK=$L/bert.start
[ -e $MARK ] || touch $MARK
logname() { echo $L/$(echo "$*" | tr ' ' '_' | tr -d '-').log; }
waitmem() { until [ "$(nvidia-smi -i $1 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge "$2" ]; do sleep 120; done; }
done_() { grep -qE "^RESULT|^done already" "$1" 2>/dev/null && [ "$1" -nt "$MARK" ]; }
worker() {  # worker <gpu> <jobs file> <free MB>: as in run_r11.sh, mkdir claims and three tries per job
  for pass in 1 2 3 4; do
    while read -r l; do
      [ -z "$l" ] && continue
      lg=$(logname $l); n=$(basename "$lg" .log)
      done_ "$lg" && continue
      mkdir "$CL/$n" 2>/dev/null || continue
      waitmem $1 $3
      CUDA_VISIBLE_DEVICES=$1 python train.py $l > "$lg" 2>&1 < /dev/null
      done_ "$lg" && continue
      t=$(( $(cat "$CL/$n.tries" 2>/dev/null || echo 0) + 1 )); echo $t > "$CL/$n.tries"
      [ $t -lt 3 ] && rmdir "$CL/$n"
    done < "$2"
  done
}
DS="beauty sports toys ml1m"

echo "BERT_TUNE_START $(date)"
for ds in $DS; do for p in 0.2 0.4 0.6; do for q in 0 0.2; do
  echo "$ds --arch bert --mask_prob $p --mask_last $q --epochs 1000 --patience 40 --tag bert_m${p}_l${q}_sas --seed 0"
  echo "$ds --arch bert --mask_prob $p --mask_last $q --blocks 2 --heads 2 --dropout 0.1 --epochs 1000 --patience 40 --tag bert_m${p}_l${q}_orig --seed 0"
done; done; done > jobs_bert_tune.txt
for g in 0 1 0 1 0 1; do worker $g jobs_bert_tune.txt 4000 & sleep 5; done; wait
echo "BERT_TUNE_DONE $(date)"

python bert_select.py $R jobs_bert_main.txt || { echo BERT_SELECT_FAILED; exit 1; }
for g in 0 1 0 1 0 1 0 1; do worker $g jobs_bert_main.txt 4000 & sleep 5; done; wait
echo "BERT_MAIN_DONE $(date)"

for g in 0 1; do
  (i=0
   for ds in $DS; do for a in "" _fr _fv; do for b in "" _fr _fv; do
     i=$((i + 1)); [ $((i % 2)) = $g ] || continue
     waitmem $g 6000
     CUDA_VISIBLE_DEVICES=$g python eval_cross.py $ds$a $ds$b > $L/cross_$ds${a}__$ds$b.log 2>&1 < /dev/null || echo "CROSS_FAILED $ds$a $ds$b"
   done; done; done) &
done; wait
echo "BERT_ALL_DONE $(date)"
