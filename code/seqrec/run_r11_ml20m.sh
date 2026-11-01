#!/bin/bash
# Round 11, ML-20M seeds 3-4 (conventional, refit, shuffled ties) on whichever GPU has room, replacing the two
# per-GPU chains of run_r11.sh: at most two ML-20M runs per GPU, the refits (longest) first. A refit is started
# only when the GPU has 26 GB free (its refit phase needs up to ~26 GB), any other run when it has 16 GB (~13 GB).
# Runs already in progress (e.g. started by an earlier copy of this script) are left alone; a failed run is
# retried, three tries in all.
cd ~/seqrec/code
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
L=~/seqrec/logs
MARK=$L/r11.start
logname() { echo $L/$(echo "$*" | tr ' ' '_' | tr -d '-').log; }
done_() { grep -qE "^RESULT|^done already" "$1" 2>/dev/null && [ "$1" -nt "$MARK" ]; }
running_() { pgrep -fx "python train.py $*" > /dev/null; }
freemem() { nvidia-smi -i $1 --query-gpu=memory.free --format=csv,noheader,nounits; }
n_ml20m() {  # ML-20M runs on GPU $1
  for p in $(pgrep -f "[t]rain.py ml20m"); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -x "CUDA_VISIBLE_DEVICES=$1"; done | wc -l
}
JOBS=("ml20m --method sasrec --seed 3 --refit 1" "ml20m --method sasrec --seed 4 --refit 1"
      "ml20m --method sasrec --seed 3" "ml20m --method sasrec --seed 4"
      "ml20m_shuf --method sasrec --seed 3" "ml20m_shuf --method sasrec --seed 4")
declare -A tries
while :; do
  left=0
  for j in "${JOBS[@]}"; do
    lg=$(logname $j)
    done_ "$lg" && continue
    running_ $j && { left=1; continue; }
    [ "${tries[$j]:-0}" -ge 3 ] && continue
    left=1
    need=$([[ $j == *refit* ]] && echo 26000 || echo 16000)
    g=""
    for c in 0 1; do [ "$(n_ml20m $c)" -lt 2 ] && [ "$(freemem $c)" -ge $need ] && { g=$c; break; }; done
    [ -z "$g" ] && continue
    tries[$j]=$(( ${tries[$j]:-0} + 1 ))
    echo "$(date +%m-%d_%H:%M) GPU$g: $j (try ${tries[$j]})"
    CUDA_VISIBLE_DEVICES=$g python train.py $j > "$lg" 2>&1 < /dev/null &
    sleep 600  # let the run reach its memory footprint before placing the next one
  done
  [ $left = 0 ] && break
  sleep 120
done
echo R11_ML20M_DONE
