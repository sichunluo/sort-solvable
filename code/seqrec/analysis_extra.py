"""Reviewer analyses from existing data and per-user ranks (CPU only). Writes results/analysis_extra.json.

  refit:     SASRec refit gain and refit/PCTM ratio by the share of the target item's training occurrences that sit
             in withheld positions (each user's last training event, which the non-refit model never trains on);
             and how often withheld vs. other training events are later than another user's LOO test target.
  direction: directional vs. direction-free PCTM on tied and other LOO targets (same targets, same counts up to C^T).
  gts:       how GTS test targets are selected (first post-cutoff event or not, several eligible same-day events).
  rolling:   rolling evaluation with each user's first window target (= the GTS validation target used for model
             selection) excluded.
Usage: python analysis_extra.py <raw dir> <processed dir> <results dir>
"""
import json
import os
import sys

RAW, PROC, RES = sys.argv[1:4]
os.environ["SEQREC_DATA"] = PROC
sys.argv = [sys.argv[0], RAW, PROC]  # prep_ties / prep_time read <raw> <processed> at import

import numpy as np
import pandas as pd

import prep_time
from common import Data
from prep_ties import SOURCES

DS = os.environ.get("DATASETS", "beauty,sports,toys,ml1m").split(",")
nd = lambda r: np.where(r < 10, 1 / np.log2(r + 2.0), 0.0)
load = lambda ds, f: np.load(f"{RES}/{ds}/{f}")
SEEDS = (0, 1, 2, 3, 4)  # every neural condition is averaged over the same five seeds
seeds = lambda ds, tag, key="alone": [nd(load(ds, f)[key]) for f in sorted(os.listdir(f"{RES}/{ds}"))
                                     if f.startswith(tag + "_s") and f.endswith("_ranks.npz") and f[len(tag) + 2].isdigit()
                                     and int(f[len(tag) + 2:-len("_ranks.npz")]) in SEEDS]
out = {"refit": {}, "direction": {}, "gts": {}, "rolling": {}}

for ds in DS:
    z = np.load(f"{PROC}/{ds}.npz")
    users, items, ho_items = z["users"], z["items"], z["ho_items"]
    withheld = np.r_[users[1:] != users[:-1], True]  # last training event of each user
    occ = np.bincount(items, minlength=len(z["catalog"]))
    wocc = np.bincount(items[withheld], minlength=len(z["catalog"]))
    share = wocc[ho_items] / np.maximum(occ[ho_items], 1)
    p = nd(load(ds, "experts_ranks.npz")["pctm_test"])
    conv, refit = np.mean(seeds(ds, "sasrec"), 0), np.mean(seeds(ds, "sasrec_refit"), 0)
    strata = {"<10%": share < .1, "10-25%": (share >= .1) & (share < .25), "25-50%": (share >= .25) & (share < .5),
              ">=50%": share >= .5}
    row = {k: {"share_of_targets": float(m.mean()), "refit_gain": float(refit[m].mean() / conv[m].mean() - 1),
               "refit_over_pctm": float(refit[m].mean() / p[m].mean())} for k, m in strata.items()}
    # event times in LOO order (same stable sort as prep_ties.loo; hold-out filters do not touch training events)
    df = SOURCES[ds]().assign(order=lambda x: np.arange(len(x)))
    df = df.sort_values(["user", "t", "order"], kind="stable").reset_index(drop=True)
    last = ~df.user.duplicated(keep="last")
    t_train, t_test = df.t[~last].values, np.sort(df.t[last].values)
    assert len(t_train) == len(items)
    later = np.searchsorted(t_test, t_train, side="left") / len(t_test)  # share of test targets strictly earlier
    row["withheld_later_than_test_target"] = float(later[withheld].mean())
    row["other_train_later_than_test_target"] = float(later[~withheld].mean())
    out["refit"][ds] = row

    gap = np.load(f"{PROC}/{ds}_meta.npz")["ho_gap"]
    tie = gap == 0 if ds == "ml1m" else gap < 86400
    sym = nd(load(ds, "baselines_ranks.npz")["pctm_sym"])
    out["direction"][ds] = {"tied": float(p[tie].mean() / sym[tie].mean() - 1),
                            "other": float(p[~tie].mean() / sym[~tie].mean() - 1),
                            "share_of_gain_on_tied": float(tie.mean() * (p[tie].mean() - sym[tie].mean())
                                                           / (p.mean() - sym.mean()))}

    # GTS target selection
    g = prep_time.LOADERS[ds]()
    g = g.assign(order=np.arange(len(g))).sort_values(["user", "t", "order"], kind="stable")
    cut = np.quantile(g.t, 0.9)
    past, fut = g[g.t < cut], g[g.t >= cut]
    fut = fut[fut.user.isin(past.user.unique())]
    first = fut[~fut.user.duplicated(keep="first")].set_index("user")
    test = prep_time.first_valid(fut, past).set_index("user")
    same = (first.loc[test.index, "item"].values == test.item.values)
    ts = np.load(f"{PROC}/{ds}_tset.npz")["ho_users"]
    out["gts"][ds] = {"users_with_future_events": int(len(first)), "test_users": int(len(test)),
                      "target_is_first_future_event": float(same.mean()),
                      "first_future_event_new_item": float((~first.item.isin(past.item.unique())).mean()),
                      "several_eligible_same_day": float(np.mean(np.unique(ts, return_counts=True)[1] > 1))}

    # rolling evaluation without the validation targets
    for suf in ("_t", "_ts"):
        d = Data(ds + suf)
        in_inner = np.zeros(d.n_items, bool)
        in_inner[np.concatenate([s for s in d.inner if len(s)])] = True
        first_of_user = []
        for u, s in enumerate(d.seqs):
            got = False
            for k in range(max(1, len(d.inner[u])), len(s)):
                if in_inner[s[k]] and not np.any(s[:k] == s[k]):
                    first_of_user.append(not got)
                    got = True
        keep = ~np.asarray(first_of_user)
        np.save(f"{RES}/{ds}{suf}/rolling_keep.npy", keep)  # used by make_figs.py
        r = np.load(f"{RES}/{ds}{suf}/rolling_ranks.npz")
        assert len(keep) == len(r["gaps"])
        sas = np.mean([nd(r[k]) for k in r.files if k.startswith("sasrec_s") and int(k[len("sasrec_s"):]) in SEEDS], 0)
        pc, gaps = nd(r["pctm"]), r["gaps"]
        tied = gaps == 0
        res = {}
        for name, m in (("all", keep), ("tied", keep & tied), ("other", keep & ~tied),
                        ("all_incl_val", np.ones(len(keep), bool))):
            res[name] = float(sas[m].mean() / pc[m].mean())
        res["share_excluded"] = float(1 - keep.mean())
        out["rolling"][ds + suf] = res
    print(ds, json.dumps({k: out[k][ds] for k in ("refit", "direction", "gts")}), out["rolling"][ds + "_t"],
          out["rolling"][ds + "_ts"], flush=True)

# Recency vs. leakage inside LOO: the refit gain by how much of the withheld data lies after the user's own target,
# and the old-to-new direction of consecutive cross-day pairs (base rate for the ASIN-age direction).
out["refit_by_leak"], out["cross_day_newer"] = {}, {}
for ds in ["beauty", "sports", "toys"]:
    z = np.load(f"{PROC}/{ds}.npz")
    users, items = z["users"], z["items"]
    withheld = np.r_[users[1:] != users[:-1], True]
    df = SOURCES[ds]().assign(order=lambda x: np.arange(len(x)))
    df = df.sort_values(["user", "t", "order"], kind="stable").reset_index(drop=True)
    last = ~df.user.duplicated(keep="last")
    t_train = df.t[~last].values
    t_w = np.sort(t_train[withheld])
    tgt = df[last].set_index("user").t.loc[z["ho_users"]].values
    later = 1 - np.searchsorted(t_w, tgt, side="right") / len(t_w)  # share of withheld events after this target
    conv, refit = np.mean(seeds(ds, "sasrec"), 0), np.mean(seeds(ds, "sasrec_refit"), 0)
    q = np.quantile(later, [0, .2, .4, .6, .8, 1])
    rows = []
    for k in range(5):
        m = (later >= q[k]) & (later <= q[k + 1]) if k == 4 else (later >= q[k]) & (later < q[k + 1])
        rows.append({"withheld_after_target": [float(later[m].min()), float(later[m].max())],
                     "refit_gain": float(refit[m].mean() / conv[m].mean() - 1), "n": int(m.sum())})
    out["refit_by_leak"][ds] = rows
    first = df.groupby("item").t.min()
    nxt = df.shift(-1)
    pair = (df.user.values == nxt.user.values)
    cross = pair & (nxt.t.values > df.t.values)
    newer = first.loc[nxt.item[cross].astype(int)].values > first.loc[df.item[cross]].values
    same = first.loc[nxt.item[cross].astype(int)].values == first.loc[df.item[cross]].values
    out["cross_day_newer"][ds] = {"newer": float(newer.mean()), "same_first_day": float(same.mean()),
                                  "older": float(1 - newer.mean() - same.mean()), "pairs": int(cross.sum())}
    print(ds, [round(100 * r["refit_gain"]) for r in rows], [r["withheld_after_target"] for r in rows],
          out["cross_day_newer"][ds], flush=True)

json.dump(out, open(f"{RES}/analysis_extra.json", "w"), indent=1)
