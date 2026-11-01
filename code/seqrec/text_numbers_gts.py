"""Numbers quoted only in the text that are not in text_numbers.py or text_numbers_loo.py, over seeds 0-4: the GTS
order crossing (training order x test-history order on the same checkpoints), GRU4Rec with random GTS training
ties, the leaky-GTS change of the SASRec/PCTM ratio (two-level paired bootstrap), and the refit SASRec relative to
the validation-selected baseline across three tie-shuffle permutations. Writes results/text_numbers_gts.json.
Usage: python text_numbers_gts.py <results> <processed>
"""
import glob
import json
import os
import sys

import numpy as np

R, PROC = sys.argv[1:3]
DS = ["beauty", "sports", "toys", "ml1m"]
SEEDS = (0, 1, 2, 3, 4)
rng = np.random.default_rng(0)
nd = lambda r: np.where(r < 10, 1 / np.log2(r + 2.0), 0.0)
seed_of = lambda f: int(os.path.basename(f).rsplit("_s", 1)[1].split("_")[0])
rnd = lambda x, k=3: round(float(x), k)


def runs(ds, tag, key="alone"):
    return {seed_of(f): nd(np.load(f)[key]) for f in sorted(glob.glob(f"{R}/{ds}/{tag}_s[0-9]_ranks.npz"))
            if seed_of(f) in SEEDS and key in np.load(f).files}


def cross_sasrec(target, train, hist):
    z = np.load(f"{R}/{target}/cross_{train}__{hist}.npz")
    return np.mean([nd(z[f"sasrec_s{s}"]).mean() for s in SEEDS])


out = {}
for ds in DS:
    o = out[ds] = {}
    t, x = f"{ds}_t", f"{ds}_tsx"
    base = cross_sasrec(t, t, t)
    o["cross_train_rnd_hist_id_%"] = rnd(100 * (cross_sasrec(t, x, t) / base - 1), 1)
    o["cross_train_id_hist_rnd_%"] = rnd(100 * (cross_sasrec(x, t, x) / base - 1), 1)
    o["cross_both_rnd_%"] = rnd(100 * (cross_sasrec(x, x, x) / base - 1), 1)
    g0, g1 = runs(t, "gru"), runs(x, "gru")
    o["gru_tsx_vs_t_%"] = rnd(100 * (np.mean([v.mean() for v in g1.values()]) / np.mean([v.mean() for v in g0.values()]) - 1), 1)
    # leaky GTS: the union of the two folds, aligned to the GTS test users
    users = np.load(f"{PROC}/{t}.npz")["ho_users"]
    fold_users = np.concatenate([np.load(f"{PROC}/{ds}_tl{k}.npz")["ho_users"] for k in (0, 1)])
    pos = {u: i for i, u in enumerate(fold_users)}
    order = np.asarray([pos[u] for u in users])
    leak = [np.concatenate([runs(f"{ds}_tl{k}", "sasrec")[s] for k in (0, 1)])[order] for s in SEEDS]
    gts = [runs(t, "sasrec")[s] for s in SEEDS]
    pl = np.concatenate([nd(np.load(f"{R}/{ds}_tl{k}/experts_ranks.npz")["pctm_test"]) for k in (0, 1)])[order]
    pg = nd(np.load(f"{R}/{t}/experts_ranks.npz")["pctm_test"])
    L, G = np.asarray(leak), np.asarray(gts)
    d = L.mean() / pl.mean() - G.mean() / pg.mean()
    bs = []
    for _ in range(2000):
        k = rng.integers(0, len(SEEDS), len(SEEDS))
        u = rng.integers(0, len(users), len(users))
        a, b = L[k].mean(0)[u], G[k].mean(0)[u]
        bs.append(a.mean() / pl[u].mean() - b.mean() / pg[u].mean())
    o["leaky_ratio_change"] = [rnd(d, 3), *[rnd(v, 3) for v in np.percentile(bs, [2.5, 97.5])]]
    o["leaky_change_%"] = {"sasrec": rnd(100 * (L.mean() / G.mean() - 1), 1), "pctm": rnd(100 * (pl.mean() / pg.mean() - 1), 1)}
    # refit SASRec vs the validation-selected baseline, per tie-shuffle permutation (the extra permutations have no
    # validation scores for every baseline, so they use the baseline selected on the first one)
    perm, best = {}, None
    for suf in ("_shuf", "_shuf1", "_shuf2"):
        dd = f"{ds}{suf}"
        if not os.path.exists(f"{R}/{dd}/baselines_ranks.npz"):
            continue
        e, b = np.load(f"{R}/{dd}/experts_ranks.npz"), np.load(f"{R}/{dd}/baselines_ranks.npz")
        test = {"pctm": e["pctm_test"], "mc": e["mc_test"], **{k: b[k] for k in ("pctm_sym", "seqrules", "mostpop")}}
        if best is None:
            val = {"pctm": e["pctm_val"], "mc": e["mc_val"], **{k: b[f"{k}_val"] for k in ("pctm_sym", "seqrules", "mostpop")}}
            best = max(val, key=lambda k: nd(val[k]).mean())
        s = runs(dd, "sasrec_refit")
        perm[suf] = [best, rnd(np.mean([v.mean() for v in s.values()]) / nd(test[best]).mean()), len(s)]
    o["perm_refit_vs_strongest"] = perm
    print(ds, json.dumps(o), flush=True)
json.dump(out, open(f"{R}/text_numbers_gts.json", "w"), indent=1)
