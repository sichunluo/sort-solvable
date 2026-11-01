"""LightGBM LambdaRank re-ranking of the SASRec+PCTM candidate union, and the predictability probe.

Trained on 80% of val users, early-stopped on the other 20%, evaluated on test users (a target outside the candidate
set counts as a miss). Runs in the separate LightGBM venv (numpy, lightgbm, sklearn only).
Usage: python ltr_fit.py <results dir> <dataset> <sasrec seed>
"""
import json
import sys

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

R, ds, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
V, T = np.load(f"{R}/{ds}/ltr_val_s{seed}.npz"), np.load(f"{R}/{ds}/ltr_test_s{seed}.npz")
names = list(V["cand_features"]) + list(V["user_features"])
nd = lambda r: np.where(r < 10, 1 / np.log2(r + 2.0), 0.0)
rng = np.random.default_rng(0)


def table(Z):
    return np.hstack([Z["X"], Z["U"][Z["row"]]]), Z["label"].astype(np.int32), Z["row"]


def groups(rows):
    return np.unique(rows, return_counts=True)[1]


Xv, yv, rv = table(V)
n_val = len(V["U"])
tr_user = rng.random(n_val) < 0.8
m_tr, m_ho = tr_user[rv], ~tr_user[rv]
late_v = Xv[:, names.index("late_gap")]


def holdout_ndcg(score, rows, label):
    """NDCG@10 over holdout users from candidate scores (users without a positive count as 0)."""
    order = np.lexsort((-score, rows))
    r, lab = rows[order], label[order]
    starts = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
    pos = np.arange(len(r)) - np.repeat(starts, np.diff(np.r_[starts, len(r)]))
    gain = np.where(lab.astype(bool) & (pos < 10), 1 / np.log2(pos + 2.0), 0.0)
    return gain.sum() / (~tr_user).sum()


# Boost on top of the late-fusion score (stacking residual); pick its scale on the holdout users.
fits = {}
for scale in (0.3, 1.0, 3.0):
    rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=2000, learning_rate=0.02, num_leaves=31,
                        min_child_samples=200, reg_lambda=10.0, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.8, lambdarank_truncation_level=20, n_jobs=16, verbose=-1)
    rk.fit(Xv[m_tr], yv[m_tr], group=groups(rv[m_tr]), init_score=scale * late_v[m_tr],
           eval_set=[(Xv[m_ho], yv[m_ho])], eval_group=[groups(rv[m_ho])], eval_init_score=[scale * late_v[m_ho]],
           eval_at=[10], callbacks=[lgb.early_stopping(200, verbose=False)])
    ho = holdout_ndcg(scale * late_v[m_ho] + rk.predict(Xv[m_ho], num_iteration=rk.best_iteration_),
                      rv[m_ho], yv[m_ho])
    fits[scale] = (ho, rk)
scale = max(fits, key=lambda s: fits[s][0])
ranker = fits[scale][1]
late_ho = holdout_ndcg(late_v[m_ho], rv[m_ho], yv[m_ho])

Xt, yt, rt = table(T)
score = scale * Xt[:, names.index("late_gap")] + ranker.predict(Xt, num_iteration=ranker.best_iteration_)
late_logrank = Xt[:, names.index("late_logrank")]
n_test = len(T["U"])
r_ltr = np.full(n_test, 10 ** 9)
order = np.lexsort((late_logrank, -score, rt))  # by user, then LTR score desc, ties by late-fusion rank
pos = np.empty(len(order), dtype=np.int64)
starts = np.r_[0, np.flatnonzero(np.diff(rt[order])) + 1]
for a, b in zip(starts, np.r_[starts[1:], len(order)]):
    pos[order[a:b]] = np.arange(b - a)
hit = yt.astype(bool)
r_ltr[rt[hit]] = pos[hit]


def boot(a, b, n=2000):
    dlt = nd(a) - nd(b)
    m = dlt[rng.integers(0, len(dlt), (n, len(dlt)))].mean(1)
    return [float(dlt.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


res = {"dataset": ds, "seed": seed, "best_iter": int(ranker.best_iteration_), "init_scale": scale,
       "holdout_val_ndcg10": {"late": float(late_ho), **{f"ltr_x{s}": float(v[0]) for s, v in fits.items()}},
       "target_in_candidates_test": float(np.mean(r_ltr < 10 ** 9)),
       "test_ndcg10": {"sasrec": float(nd(T["rank_z"]).mean()), "pctm": float(nd(T["rank_s"]).mean()),
                       "late": float(nd(T["rank_late"]).mean()), "ltr": float(nd(r_ltr).mean()),
                       "oracle_selector": float(np.maximum(nd(T["rank_z"]), nd(T["rank_s"])).mean())},
       "ltr_minus_late": boot(r_ltr, T["rank_late"]),
       "importance_gain": dict(sorted(zip(names, ranker.booster_.feature_importance("gain").round(1).tolist()),
                                      key=lambda kv: -kv[1])[:10])}

# Predictability: from per-user features only, which model hits@10?
Uv, Ut = V["U"], T["U"]
hz_v, hs_v, hz_t, hs_t = V["rank_z"] < 10, V["rank_s"] < 10, T["rank_z"] < 10, T["rank_s"] < 10
auc = {}
for name, lv, lt, mv, mt in (("sasrec_only_vs_pctm_only", hz_v, hz_t, hz_v ^ hs_v, hz_t ^ hs_t),
                             ("sasrec_hits", hz_v, hz_t, np.ones_like(hz_v), np.ones_like(hz_t)),
                             ("pctm_hits", hs_v, hs_t, np.ones_like(hs_v), np.ones_like(hs_t))):
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                             subsample=0.8, subsample_freq=1, n_jobs=16, verbose=-1)
    clf.fit(Uv[mv], lv[mv])
    auc[name] = {"auc": float(roc_auc_score(lt[mt], clf.predict_proba(Ut[mt])[:, 1])), "n_test": int(mt.sum()),
                 "pos_rate": float(lt[mt].mean())}
res["predictability_auc"] = auc
print(json.dumps(res), flush=True)
with open(f"{R}/{ds}/ltr_s{seed}.json", "w") as f:
    json.dump(res, f, indent=1)
np.savez(f"{R}/{ds}/ltr_s{seed}_ranks.npz", ltr=r_ltr)
