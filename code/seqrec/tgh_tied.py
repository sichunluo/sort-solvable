"""TGH-1/TGH-2 NDCG@10 on the GRID splits (original, fixed targets with random ties, ties shuffled; prep_grid.py),
overall and split into tied test targets (same review day as the previous event of that split's test sequence) and
the others. Reads tgh_out/<variant>/<ds>/TGH-*.npz (tgh_npz.py) and the split files; writes results/tgh.json.
Usage: python tgh_tied.py <grid split dir> <tgh_out dir> <results dir>
"""
import glob
import json
import os
import sys

import numpy as np

import tfrecord_lite as T

GRID, TOUT, RES = sys.argv[1:4]

NAMES = ("beauty", "sports", "toys")


def ndcg(z):
    hit = z["top10"] == z["answers"][:, None]
    return np.where(hit.any(1), 1 / np.log2(hit.argmax(1) + 2.0), 0.0)


res = {}
for ds in NAMES:
    if not os.path.exists(f"{GRID}/{ds}_days.npy"):
        continue
    day = {(u, i): t for u, i, t in np.load(f"{GRID}/{ds}_days.npy")}  # (GRID user, GRID item) -> review day
    for v in ("", "_fr", "_shuf"):
        test = {}
        for f in glob.glob(f"{GRID}/{ds}{v}/testing/*.tfrecord.gz"):
            for ex in T.read(f):
                test[ex["user_id"][0]] = ex["sequence_data"]
        tied = np.asarray([day[(u, test[u][-1])] == day[(u, test[u][-2])] for u in range(len(test))])
        for m in ("TGH-1", "TGH-2"):
            f = f"{TOUT}/{ds}{v}/{ds}/{m}.npz"
            if not os.path.exists(f):
                continue
            z = np.load(f)
            assert (z["users"] == np.arange(len(test))).all() and (z["answers"] == [test[u][-1] for u in range(len(test))]).all()
            n = ndcg(z)
            res.setdefault(ds, {}).setdefault(m, {})[v or "orig"] = dict(
                all=float(n.mean()), tied=float(n[tied].mean()), other=float(n[~tied].mean()),
                tied_share=float(tied.mean()), users=int(len(n)))
json.dump(res, open(f"{RES}/tgh.json", "w"), indent=1)
for ds, r in res.items():
    for m, rr in r.items():
        for v, x in rr.items():
            print(f"{ds:7s} {m} {v:5s} all {x['all']:.4f}  tied {x['tied']:.4f}  other {x['other']:.4f}  "
                  f"(tied {x['tied_share']:.1%})")
