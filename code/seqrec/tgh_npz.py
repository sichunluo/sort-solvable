"""Convert TGH's predictions/<dataset>/{TGH-1,TGH-2}.pkl (written by tgh.py in the same run) to npz with each user's
top-10 list and target, so that the analysis never unpickles files on another machine.
Usage: python tgh_npz.py <predictions dir>
"""
import os
import pickle
import sys

import numpy as np

for m in ("TGH-1", "TGH-2"):
    path = os.path.join(sys.argv[1], f"{m}.pkl")
    with open(path, "rb") as f:
        preds = pickle.load(f)
    preds.sort(key=lambda p: p["user_id"])
    top = np.full((len(preds), 10), -1, dtype=np.int64)
    for i, p in enumerate(preds):
        top[i, :len(p["item_ids"][:10])] = p["item_ids"][:10]
    np.savez(path[:-4] + ".npz", users=np.asarray([p["user_id"] for p in preds]), top10=top,
             answers=np.asarray([p["gt_ids"][0] for p in preds]))
    print("wrote", path[:-4] + ".npz")
