#!/bin/bash
# Round 10 retries for runs that ran out of GPU memory. After the eSASRec workers exit, failed eSASRec splits are
# released and re-run on a GPU with room; after the BSARec workers exit, claims of BSARec jobs that left no output
# and no running process are released and four fresh workers re-run them. Up to three passes each.
cd ~/seqrec
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mt
waitmem() { until [ "$(nvidia-smi -i $1 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge "$2" ]; do sleep 120; done; }
while pgrep -f "[r]un_esasrec_queue.sh" > /dev/null; do sleep 300; done
for a in 1 2 3; do
  [ -s esasrec_logs/failed.txt ] || break
  while read -r _ ds mode seed; do rmdir esasrec_claims/${ds}_s$seed 2>/dev/null; done < esasrec_logs/failed.txt
  mv esasrec_logs/failed.txt esasrec_logs/failed_try$a.txt
  for seed in 33 34; do waitmem 0 8000; bash code/run_esasrec_queue.sh 0 $seed; done
done
echo R10_RETRY_ESASREC_DONE
while pgrep -f "[r]un_bsarec_queue.sh" > /dev/null; do sleep 300; done
for a in 1 2 3; do
  n=0
  for c in bsarec/claims/*_s4[234]; do
    j=${c##*/}
    [ -e $c/skip ] || [ -f bsarec/output/${j}_test.npz ] || pgrep -f "[t]rain_name $j( |$)" > /dev/null && continue
    rmdir $c && n=$((n + 1))
  done
  [ -f bsarec/logs/failed.txt ] && mv bsarec/logs/failed.txt bsarec/logs/failed_try$a.txt
  [ $n = 0 ] && break
  for g in 0 1 0 1; do waitmem $g 4000; bash code/run_bsarec_queue.sh $g & sleep 15; done; wait
done
echo R10_RETRY_DONE
