"""Scores of the BSARec code-base runs (run_bsarec.py) on the original S3-Rec files and on the files with ties at
random (prep_bsarec.py), overall and split into tied test targets (same review day as the previous interaction in
that file) and the others. Reads the saved top-20 lists; writes results/bsarec.json.
Usage: python bsarec_tied.py <raw dir> <processed dir> <bsarec output dir> <bsarec data dir> <results dir>
"""
import glob
import json
import os
import sys

RAW, PROC, OUTD, DATA, RES = sys.argv[1:6]
sys.argv = [sys.argv[0], RAW, PROC]  # prep_ties reads <raw> <processed> at import

import numpy as np  # noqa: E402

from prep_ties import amazon_events  # noqa: E402

NAMES = {"Beauty": "beauty", "Sports_and_Outdoors": "sports", "Toys_and_Games": "toys"}


def tied_targets(name, suffix):
    """Per line of <name><suffix>.txt: is the last item reviewed on the same day as the one before it?"""
    t = {(u, i): v for u, i, v in amazon_events(name)[["user", "item", "t"]].itertuples(index=False)}
    out = []
    for line in open(f"{DATA}/{name}{suffix}.txt"):
        u, *items = map(int, line.split())
        out.append(t[(u, items[-1])] == t[(u, items[-2])])
    return np.asarray(out)


def ndcg(f):
    z = np.load(f)
    hit = z["top20"][:, :10] == z["answers"][:, None]
    return np.where(hit.any(1), 1 / np.log2(hit.argmax(1) + 2.0), 0.0)


res = {}
for name, ds in NAMES.items():
    masks = {suf: tied_targets(name, suf) for suf in ("", "_shuf")}
    for f in sorted(glob.glob(f"{OUTD}/*_{name}_s*_test.npz")):
        model, seed = os.path.basename(f).split(f"_{name}_")[0], os.path.basename(f).rsplit("_s", 1)[1][:-9]
        g = f.replace(f"_{name}_", f"_{name}_shuf_")
        if not os.path.exists(g):
            continue
        a, b = ndcg(f), ndcg(g)
        ma, mb = masks[""], masks["_shuf"]
        res.setdefault(ds, {})[f"{model}_s{seed}"] = dict(
            orig=float(a.mean()), shuf=float(b.mean()), change=float(b.mean() / a.mean() - 1),
            orig_tied=float(a[ma].mean()), orig_other=float(a[~ma].mean()),
            shuf_tied=float(b[mb].mean()), shuf_other=float(b[~mb].mean()),
            tied_share=[float(ma.mean()), float(mb.mean())], users=int(len(a)))
json.dump(res, open(f"{RES}/bsarec.json", "w"), indent=1)
for ds, r in res.items():
    for k, v in r.items():
        print(f"{ds:7s} {k:22s} orig {v['orig']:.4f} shuf {v['shuf']:.4f} ({100 * v['change']:+.1f}%)  "
              f"tied {v['orig_tied']:.4f}->{v['shuf_tied']:.4f}  other {v['orig_other']:.4f}->{v['shuf_other']:.4f}")
