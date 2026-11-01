#!/bin/bash
# Extra workers for run_bert.sh (same claims, logs and done test): run_bert_extra.sh <jobs file> <gpu> [<gpu> ...].
# Waits until the jobs file exists (the main list is written after the tuning round).
cd ~/seqrec/code
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
S=~/seqrec; L=$S/logs/bert; CL=$S/bert_claims; MARK=$L/bert.start
logname() { echo $L/$(echo "$*" | tr ' ' '_' | tr -d '-').log; }
waitmem() { until [ "$(nvidia-smi -i $1 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge "$2" ]; do sleep 120; done; }
done_() { grep -qE "^RESULT|^done already" "$1" 2>/dev/null && [ "$1" -nt "$MARK" ]; }
worker() {
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
J=$1; shift
until [ -s "$J" ]; do sleep 120; done
for g in "$@"; do worker $g "$J" 4000 & sleep 5; done; wait
echo "EXTRA_DONE $J $(date)"
