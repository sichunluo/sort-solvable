#!/bin/bash
# Round 10: three seeds for every reported neural condition. SASRec seed 2 on the extra tie permutations and the
# refit-length check, fixed-beta PoE seeds 1-2, ML-20M seeds 1-2 (conventional, refit, shuffled ties), eSASRec seeds
# 33-34 on the reported LOO splits, BSARec code base seeds 43-44 (jobs_bsarec.txt).
cd ~/seqrec/code
./run_queue.sh 0 jobs_r10_ml20m_g0.txt 3 &
./run_queue.sh 1 jobs_r10_ml20m_g1.txt 3 &
./run_queue.sh 0 jobs_r10_g0.txt 2 &
./run_queue.sh 1 jobs_r10_g1.txt 2 &
cd ~/seqrec
for w in "0 33" "1 33" "0 34" "1 34"; do bash code/run_esasrec_queue.sh $w & sleep 5; done
# BSARec code-base workers: started separately (10 workers, run_bsarec_queue.sh on GPUs 0/1 in turn).
wait
echo R10_DONE
