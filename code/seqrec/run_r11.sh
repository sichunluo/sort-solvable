#!/bin/bash
# Round 11: five seeds for every reported neural condition: seeds 0-4 (1-5 for the analyses that reload saved LOO
# checkpoints), eSASRec 32-36, BSARec code base 42-46, ML-20M 0-4. Starts once the three-seed runs are done: the
# ML-20M runs of a GPU when the round-10b ML-20M queue of that GPU has exited, everything else when
# run_r10_retry.sh has finished. Run at low CPU priority (nice 10) with four threads per process.
cd ~/seqrec/code
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
S=~/seqrec; R=$S/results; L=$S/logs; CL=$S/r11_claims
mkdir -p $CL
MARK=$L/r11.start  # a job is done once a log written after this logs a RESULT (older logs share names)
touch $MARK
logname() { echo $L/$(echo "$*" | tr ' ' '_' | tr -d '-').log; }
waitmem() { until [ "$(nvidia-smi -i $1 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge "$2" ]; do sleep 120; done; }
done_() { grep -qE "^RESULT|^done already" "$1" 2>/dev/null && [ "$1" -nt "$MARK" ]; }
gone() { while kill -0 $1 2>/dev/null && grep -q run_r10b /proc/$1/cmdline 2>/dev/null; do sleep 300; done; }
worker() {  # worker <gpu> <jobs file> <free MB>: train.py jobs claimed with mkdir, so several workers can share a
            # list; a failed job is released for another try, three tries in all
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

# ML-20M seeds 3 (GPU 0) and 4 (GPU 1): conventional, refit, shuffled ties
(gone 3660466; worker 0 jobs_r11_ml20m_g0.txt 26000; echo R11_ML20M_G0_DONE) & m0=$!
(gone 3660467; worker 1 jobs_r11_ml20m_g1.txt 26000; echo R11_ML20M_G1_DONE) & m1=$!

until grep -q R10_RETRY_DONE $L/r10_retry.log 2>/dev/null; do sleep 300; done
echo "R11_START $(date)"

# BSARec code base seeds 45-46 (FEARec and BERT4Rec stay blocked); then up to three retry passes that release
# claims which left no output and have no running process
(cd $S
 for c in bsarec/claims/*_s4[56]; do case ${c##*/} in FEARec_*|BERT4Rec_*) ;; *) rm -f $c/skip; rmdir $c;; esac; done
 for g in 0 1 0 1 0 1 0 1 0 1; do bash code/run_bsarec_queue.sh $g $S/code/jobs_bsarec5.txt & sleep 15; done; wait
 for a in 1 2 3; do
   n=0
   for c in bsarec/claims/*_s4[56]; do
     j=${c##*/}
     [ -e $c/skip ] || [ -f bsarec/output/${j}_test.npz ] || pgrep -f "[t]rain_name $j( |$)" > /dev/null && continue
     rmdir $c && n=$((n + 1))
   done
   [ -f bsarec/logs/failed.txt ] && mv bsarec/logs/failed.txt bsarec/logs/failed_r11_try$a.txt
   [ $n = 0 ] && break
   for g in 0 1 0 1; do waitmem $g 4000; bash code/run_bsarec_queue.sh $g $S/code/jobs_bsarec5.txt & sleep 15; done; wait
 done
 echo R11_BSAREC_DONE) & b=$!

# eSASRec seeds 35-36 on the eight reported splits, three workers per seed; failed splits re-run up to three times
(cd $S
 for w in "0 35" "1 35" "0 35" "1 36" "0 36" "1 36"; do bash code/run_esasrec_queue.sh $w & sleep 5; done; wait
 for a in 1 2 3; do
   [ -s esasrec_logs/failed.txt ] || break
   while read -r _ ds mode seed; do rmdir esasrec_claims/${ds}_s$seed 2>/dev/null; done < esasrec_logs/failed.txt
   mv esasrec_logs/failed.txt esasrec_logs/failed_r11_try$a.txt
   for seed in 35 36; do waitmem 0 8000; bash code/run_esasrec_queue.sh 0 $seed; done
 done
 echo R11_ESASREC_DONE) & e=$!

# Our models: seeds 3-4 of every three-seed condition, PoE seeds 4-5 and LOO SASRec seed 5 (for the reload analyses)
(for g in 0 1 0 1 0 1; do worker $g jobs_r11.txt 6000 & sleep 20; done; wait
 worker 0 jobs_r11.txt 6000  # picks up runs released after the other workers had finished their passes
 # analyses that reload LOO checkpoints: LTR, contextual gate and weight-scale diagnostic for seeds 4-5
 for b in beauty sports toys ml1m; do
   for s in 4 5; do
     [ -f $R/$b/sasrec_s$s.pt ] || continue
     [ -f $R/$b/ltr_s$s.json ] || { waitmem 0 6000; [ -f $R/$b/ltr_test_s$s.npz ] || CUDA_VISIBLE_DEVICES=0 python ltr_dump.py $b $s
       $S/ltr_env/bin/python -W ignore ltr_fit.py $R $b $s | cut -c1-300; }
     [ -f $R/$b/gate_s$s.json ] || { waitmem 0 6000; CUDA_VISIBLE_DEVICES=0 python gate.py $b --seed $s > $L/gate_${b}_s$s.log 2>&1
       tail -1 $L/gate_${b}_s$s.log | cut -c1-200; }
     [ -f $R/$b/diag_weights_s$s.json ] || { waitmem 0 6000; CUDA_VISIBLE_DEVICES=0 python diag_weights.py $b $s 2>&1 | grep "std(beta"; }
   done
 done
 # cross evaluations among the original, random-tie and reversed-tie LOO splits with all five seeds
 for b in beauty sports toys ml1m; do for tr in "" _fr _fv; do for hi in "" _fr _fv; do
   waitmem 0 8000; CUDA_VISIBLE_DEVICES=0 python eval_cross.py $b$tr $b$hi | tail -1 | cut -c1-200
 done; done; done
 echo R11_SMALL_DONE) & t=$!

wait $m0 $m1 $b $e $t
echo R11_DONE
