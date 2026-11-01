"""Numbers quoted in the text that no table shows (round 9): paired gains from random training ties under the earlier
GTS cutoffs and the pre-cutoff 5-core split, SASRec across tie permutations of the fixed-target LOO, and the
sensitivity of the refit to its number of epochs. Reads only per-user rank files. Writes results/text_numbers.json.
Usage: python text_numbers.py <results dir> <processed dir>
"""
import glob
import json
import os
import sys

import numpy as np

R, PROC = sys.argv[1:3]
DS = ["beauty", "sports", "toys", "ml1m"]
rng = np.random.default_rng(0)
nd = lambda r: np.where(r < 10, 1 / np.log2(r + 2.0), 0.0)
SEEDS = (0, 1, 2, 3, 4)  # every neural condition is averaged over the same five seeds
seed_of = lambda name: int(name.rsplit("_s", 1)[1].split("_")[0])


def runs(ds, tag):
    """Seed -> per-user NDCG@10 of one tag."""
    out = {}
    for f in sorted(glob.glob(f"{R}/{ds}/{tag}_s[0-9]_ranks.npz")):
        if seed_of(f) in SEEDS:
            out[seed_of(f)] = nd(np.load(f)["alone"])
    return out


def experts(ds):
    z, b = np.load(f"{R}/{ds}/experts_ranks.npz"), np.load(f"{R}/{ds}/baselines_ranks.npz")
    return {"pctm": nd(z["pctm_test"]), "mc": nd(z["mc_test"]), "pctm_sym": nd(b["pctm_sym"]),
            "seqrules": nd(b["seqrules"])}


def gain(xs, ys, n=2000):
    """Relative change mean(xs)/mean(ys)-1 on the same users, two-level interval (seeds, then users), in %."""
    X, Y = np.atleast_2d(xs), np.atleast_2d(ys)
    g = np.empty(n)
    for i in range(n):
        a, b = X[rng.integers(0, len(X), len(X))].mean(0), Y[rng.integers(0, len(Y), len(Y))].mean(0)
        u = rng.integers(0, X.shape[1], X.shape[1])
        g[i] = a[u].mean() / b[u].mean() - 1
    return [round(100 * v, 1) for v in (X.mean() / Y.mean() - 1, *np.percentile(g, [2.5, 97.5]))]


def tied(ds):
    gap = np.load(f"{PROC}/{ds}_meta.npz")["ho_gap"]
    return gap == 0 if ds.startswith("ml1m") else gap < 86400


out = {}
# random training ties under the earlier cutoffs (_tq -> _tqx) and the pre-cutoff 5-core split (_tp -> _tpx)
for a, b, dss in (("_tq", "_tqx", DS), ("_tp", "_tpx", ["beauty"]), ("_t", "_tsx", DS)):
    for ds in dss:
        sa, sb = runs(ds + a, "sasrec"), runs(ds + b, "sasrec")
        common = sorted(set(sa) & set(sb))
        ea, eb = experts(ds + a), experts(ds + b)
        out[f"{ds}{a}->{b}"] = dict(seeds=common, sasrec=gain([sb[s] for s in common], [sa[s] for s in common]),
                                    **{k: gain(eb[k], ea[k]) for k in ea},
                                    users=int(len(ea["pctm"])), sasrec_over_pctm=[
                                        round(np.mean([sa[s].mean() for s in common]) / ea["pctm"].mean(), 3),
                                        round(np.mean([sb[s].mean() for s in common]) / eb["pctm"].mean(), 3)])
# GTS Beauty with global vs pre-cutoff 5-core filtering
for ds in ("beauty_t", "beauty_tp"):
    s, e = runs(ds, "sasrec"), experts(ds)
    out[ds] = dict(users=int(len(e["pctm"])), sasrec=round(float(np.mean([x.mean() for x in s.values()])), 4),
                   **{k: round(float(v.mean()), 4) for k, v in e.items()})
# fixed-target LOO, three tie permutations: refit SASRec on tied targets relative to the original order
for ds in DS:
    base, m = runs(ds, "sasrec_refit"), tied(ds)
    pc = experts(ds)["pctm"]
    row = {}
    for suf in ("_fr", "_fr1", "_fr2"):
        s = runs(ds + suf, "sasrec_refit")
        common = sorted(set(s) & set(base))
        row[suf] = dict(seeds=common, tied=gain([s[k][m] for k in common], [base[k][m] for k in common]),
                        all=gain([s[k] for k in common], [base[k] for k in common]),
                        pctm_tied=gain(experts(ds + suf)["pctm"][m], pc[m]))
    out[f"{ds}_perm"] = row
# refit length: retrain for 0.8x / 1.2x of the selected number of epochs (LOO)
for ds in DS:
    base = runs(ds, "sasrec_refit")
    row = {}
    for sc in ("0.8", "1.2"):
        s = runs(ds, f"sasrec_refit_rs{sc}")
        common = sorted(set(s) & set(base))
        if common:
            row[sc] = dict(seeds=common, change=gain([s[k] for k in common], [base[k] for k in common]))
    out[f"{ds}_refit_scale"] = row
json.dump(out, open(f"{R}/text_numbers.json", "w"), indent=1)
for k, v in out.items():
    print(k, v)
