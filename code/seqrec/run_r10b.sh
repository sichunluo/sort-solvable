#!/bin/bash
# Round 10b. (1) ML-20M one run at a time per GPU (a run needs ~20 GB), each started only when the GPU has room,
# retrying runs that ran out of memory. (2) After the small round-10 runs: GTS SASRec seeds 0-2 retrained with the
# refit checkpoints kept (the first GTS runs kept them only for seeds 3+; their results are moved to r1_seed012/),
# then the cross and rolling evaluations of the GTS splits, so that every analysis uses seeds 0-2.
cd ~/seqrec/code
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MARK=~/seqrec/logs/r10.start  # a job is done once a log written after this logs a RESULT (older logs share names)
logname() { echo ~/seqrec/logs/$(echo "$*" | tr ' ' '_' | tr -d '-').log; }
waitmem() { until [ "$(nvidia-smi -i $1 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge "$2" ]; do sleep 120; done; }
done_() { grep -q "^RESULT" "$1" 2>/dev/null && [ "$1" -nt "$MARK" ]; }
queue() {  # queue <gpu> <jobs file> <MiB needed>: one job at a time, retried until it logs a RESULT (max 6 tries)
  for a in 1 2 3 4 5 6; do
    while read -r l; do
      [ -z "$l" ] && continue
      done_ "$(logname $l)" && continue
      waitmem $1 $3
      CUDA_VISIBLE_DEVICES=$1 python train.py $l > "$(logname $l)" 2>&1 < /dev/null
    done < "$2"
  done
}
queue 0 jobs_r10_ml20m_g0.txt 26000 &
(while pgrep -f "[t]rain.py ml20m --method sasrec --seed 2 --refit 1" > /dev/null; do sleep 300; done
 queue 1 jobs_r10_ml20m_g1.txt 26000) &
while pgrep -f "[r]un_queue.sh [01] jobs_r10_g[01].txt" > /dev/null; do sleep 300; done
queue 0 jobs_r10_g0.txt 6000 & q0=$!  # retries of runs that ran out of memory
queue 1 jobs_r10_g1.txt 6000 & q1=$!
wait $q0 $q1
for b in beauty sports toys ml1m; do
  mkdir -p ~/seqrec/results/${b}_t/r1_seed012
  for s in 0 1 2; do mv ~/seqrec/results/${b}_t/sasrec_s${s}{.json,_ranks.npz,.pt} ~/seqrec/results/${b}_t/r1_seed012/ 2>/dev/null; done
done
queue 0 jobs_r10_t_g0.txt 6000 & p0=$!
queue 1 jobs_r10_t_g1.txt 6000 & p1=$!
wait $p0 $p1
echo R10B_GTS_TRAINED
waitmem 0 12000
for b in beauty sports toys ml1m; do
  CUDA_VISIBLE_DEVICES=0 python eval_cross.py ${b}_t ${b}_t | tail -1
  CUDA_VISIBLE_DEVICES=0 python eval_cross.py ${b}_t ${b}_t ${b}_tset | tail -1
  CUDA_VISIBLE_DEVICES=0 python eval_cross.py ${b}_t ${b}_tsx | tail -1
  CUDA_VISIBLE_DEVICES=0 python eval_cross.py ${b}_t ${b}_tsx ${b}_tset | tail -1
  CUDA_VISIBLE_DEVICES=0 python eval_rolling.py ${b}_t | grep -v "^{" | tail -1
done
echo R10B_DONE
