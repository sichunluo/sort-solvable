#!/bin/bash
# TGH (haoyuhan1/GraphRec, src/ unchanged) on the GRID-format splits written by prep_grid.py: flan-t5-xl item
# embeddings of the original items (shared by all variants of a dataset), the transition graph of each variant,
# TGH-1 and TGH-2 with the paper's settings, and the top-10 lists converted to npz (tgh_npz.py).
# Usage: bash run_tgh.sh <gpu> <dataset: beauty|sports|toys> [variants...]   (run from ~/seqrec)
GPU=$1; DS=$2; shift 2
VARIANTS=${@:-"$DS ${DS}_shuf ${DS}_fr"}
PY=$HOME/seqrec/tgh_env/bin/python
SRC=$HOME/seqrec/graphrec/src
G=$HOME/seqrec/grid
mkdir -p $HOME/seqrec/tgh_logs $HOME/seqrec/tgh_out
export CUDA_VISIBLE_DEVICES=$GPU TF_CPP_MIN_LOG_LEVEL=3
if [ ! -f $G/$DS/t5xl.pkl ]; then  # embeddings from the local model copy (MODEL_NAME is the hub id upstream)
  $PY -c "
import sys; sys.path.insert(0, '$SRC'); sys.argv = ['generate_embeddings.py', '--dataset', '$DS', '--data_dir', '$G/$DS']
import generate_embeddings as g; g.MODEL_NAME = '$HOME/seqrec/flan-t5-xl'; g.main()
" > $HOME/seqrec/tgh_logs/${DS}_embed.log 2>&1 || exit 1
fi
for v in $VARIANTS; do
  $PY $SRC/prepare_data.py --dataset $DS --data_dir $G/$v > $HOME/seqrec/tgh_logs/${v}_graph.log 2>&1 || exit 1
  $PY $SRC/tgh.py --dataset $DS --data_dir $G/$v --emb_path $G/$DS/t5xl.pkl --method TGH-1 --hop_k 7 2 1 \
    --edge_weight_alpha 0.5 --out_dir $HOME/seqrec/tgh_out/$v > $HOME/seqrec/tgh_logs/${v}_TGH-1.log 2>&1 || exit 1
  $PY $SRC/tgh.py --dataset $DS --data_dir $G/$v --emb_path $G/$DS/t5xl.pkl --method TGH-2 --src2_hop_k 5 1 \
    --src3_hop_k 3 1 --edge_weight_alpha 0.5 --out_dir $HOME/seqrec/tgh_out/$v > $HOME/seqrec/tgh_logs/${v}_TGH-2.log 2>&1 || exit 1
  $PY $HOME/seqrec/code/tgh_npz.py $HOME/seqrec/tgh_out/$v/$DS || exit 1
done
echo "TGH_DONE $DS"
