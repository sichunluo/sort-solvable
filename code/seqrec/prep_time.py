"""Global temporal splits (no access to the future) in the same npz format as prep.py, plus the inner split.

Cutoffs at the 80th / 90th percentile of event times. Train = events before t_test. Test target = each user's first
event at or after t_test that is a train-catalogue item not already in the user's train history. Inner = events
before t_val; val target = first event in [t_val, t_test) passing the same filters against the inner data.
Amazon events come from the 2014 5-core review files (the source of the S3Rec sequences), with unixReviewTime.

Extra arrays: inner_mask and times (per train event), val_users, val_items, cutoffs; leaky folds also
cand_catalog (items that exist before t_test; the only items scored).
"""
import gzip
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])


def read_amazon(name):
    rows = []
    with gzip.open(RAW / "amazon5" / f"reviews_{name}_5.json.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["reviewerID"], r["asin"], int(r["unixReviewTime"])))
    df = pd.DataFrame(rows, columns=["user", "item", "t"])
    df["user"] = pd.factorize(df.user, sort=True)[0] + 1
    df["item"] = pd.factorize(df.item, sort=True)[0] + 1
    return df


def read_ml1m():
    with zipfile.ZipFile(RAW / "ml" / "ml-1m.zip") as z, z.open("ml-1m/ratings.dat") as f:
        df = pd.read_csv(f, sep="::", engine="python", names=["user", "item", "rating", "t"])
    return df[["user", "item", "t"]]


def first_valid(future, past):
    """Each user's first future event whose item is in the past catalogue and not in the user's past items."""
    future = future[future.user.isin(past.user.unique()) & future.item.isin(past.item.unique())]
    seen = pd.MultiIndex.from_frame(past[["user", "item"]])
    future = future[~pd.MultiIndex.from_frame(future[["user", "item"]]).isin(seen)]
    return future[~future.user.duplicated(keep="first")]


def split(df, leak_fold=None, shuffle_ties=False, fixed_test=False, q=(0.8, 0.9), cut=None):
    """leak_fold=f: same cutoffs and test targets, but keep test users of fold f (of 2) only and add every other
    user's events after t_test to train, i.e. the 'other users' future' a leave-one-out split exposes.
    shuffle_ties: order events with identical timestamps at random (seed 0) instead of by file order (= ASIN).
    fixed_test: with shuffle_ties, keep the test targets of the file-order split (only training order changes)."""
    ordered = lambda o: df.assign(order=o).sort_values(["user", "t", "order"], kind="stable").reset_index(drop=True)
    base = ordered(np.arange(len(df)))
    df = ordered(np.random.default_rng(0).permutation(len(df))) if shuffle_ties else base
    t_val, t_test = cut if cut is not None else (np.quantile(df.t, q[0]), np.quantile(df.t, q[1]))
    past = df[df.t < t_test]
    src = base if fixed_test else df
    test = first_valid(src[src.t >= t_test], past)
    train, cand = past, None
    if leak_fold is not None:
        users = np.random.default_rng(0).permutation(np.sort(test.user.unique()))
        held = users[leak_fold::2]
        train = df[(df.t < t_test) | ~df.user.isin(held)]
        test = test[test.user.isin(held)]
        cand = np.sort(past.item.unique()).astype(np.int64)  # score only items that exist before t_test
    inner_mask = (train.t < t_val).values
    val = first_valid(train[~inner_mask & (train.t < t_test).values], train[inner_mask])
    catalog = np.sort(train.item.unique()).astype(np.int64)
    idx = lambda s: np.searchsorted(catalog, s.values).astype(np.int32)
    out = dict(catalog=catalog, users=train.user.values.astype(np.int64), items=idx(train.item),
               inner_mask=inner_mask, ho_users=test.user.values.astype(np.int64), ho_items=idx(test.item),
               val_users=val.user.values.astype(np.int64), val_items=idx(val.item),
               cutoffs=np.array([t_val, t_test]), times=train.t.values.astype(np.int64))
    if cand is not None:
        out["cand_catalog"] = cand
    return out


LOADERS = {"beauty": lambda: read_amazon("Beauty"), "sports": lambda: read_amazon("Sports_and_Outdoors"),
           "toys": lambda: read_amazon("Toys_and_Games"), "ml1m": read_ml1m}
VARIANTS = {"_t": None, "_tl0": 0, "_tl1": 1, "_ts": "shuffle", "_tsx": "shuffle_fixed", "_tq": None, "_tqx": "shuffle_fixed"}
EARLY = (0.7, 0.8)  # _tq / _tqx: the same protocol with earlier cutoffs (sensitivity to the time window)

if __name__ == "__main__":
    for name in [f"{b}{v}" for b in LOADERS for v in VARIANTS] if len(sys.argv) < 4 else sys.argv[3:]:
        base, var = name.split("_")[0], "_" + name.split("_")[1]
        v = VARIANTS[var]
        d = split(LOADERS[base](), leak_fold=v if isinstance(v, int) else None,
                  shuffle_ties=isinstance(v, str), fixed_test=v == "shuffle_fixed",
                  q=EARLY if var.startswith("_tq") else (0.8, 0.9))
        cut = pd.to_datetime(d["cutoffs"], unit="s").strftime("%Y-%m-%d").tolist()
        print(f"{name}: cutoffs val/test {cut}  train users={len(np.unique(d['users']))} items={len(d['catalog'])} "
              f"events={len(d['items'])} (inner {d['inner_mask'].mean():.2f})  val={len(d['val_items'])} "
              f"test={len(d['ho_items'])}", flush=True)
        np.savez(OUT / f"{name}.npz", **d)
