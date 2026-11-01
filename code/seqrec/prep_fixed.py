"""Order-controlled variants with fixed or order-free targets (reviewer controls for the tie artifact).

  <ds>_fr / <ds>_fv : the original LOO split (same test targets, same training events) with the tied *training*
                      events ordered at random (seed 0) / in reverse file order (descending ASIN on Amazon);
                      <ds>_fr1, <ds>_fr2 use permutation seeds 1 and 2 (training-free models only).
  <ds>_day          : order-free LOO. Each user's whole last day (Amazon) / last second (ML-1M) is held out as a
                      set of targets (one ho row per item); earlier tied events are ordered at random (seed 0).
  <ds>_tset         : the GTS split of <ds>_t with set-valued targets: every eligible event on the target's day
                      (Amazon) / second (ML-1M), instead of the first one in file order (= lowest ASIN).
  ml20m_shuf        : ML-20M LOO with ties broken at random (seed 0).
Usage: python prep_fixed.py <raw dir> <processed dir>
"""
import sys

import numpy as np
import pandas as pd

import prep
import prep_time
from prep_ties import OUT, SOURCES, loo


def fixed_target(df, mode, seed=0):
    """LOO with the original targets; only the order of tied training events changes."""
    base = df.assign(order=np.arange(len(df))).sort_values(["user", "t", "order"], kind="stable")
    key = base.order.values.astype(np.float64)
    if mode == "rnd":
        key = np.random.default_rng(seed).permutation(len(base)).astype(np.float64)
    elif mode == "rev":
        key = -key
    key[(~base.user.duplicated(keep="last")).values] = np.inf  # each user's original last event stays last
    order = np.empty(len(df))
    order[base.index.values] = key
    return loo(df, order)


def day_split(df, unit):
    """Hold out each user's last day/second as an unordered set; ties in the training part at random."""
    df = df.assign(order=np.random.default_rng(0).permutation(len(df)))
    df = df.sort_values(["user", "t", "order"], kind="stable").reset_index(drop=True)
    last = (df.t // unit) == (df.groupby("user").t.transform("max") // unit)
    train, hold = df[~last], df[last]
    hold = hold[hold.user.isin(train.user.unique()) & hold.item.isin(train.item.unique())]
    seen = pd.MultiIndex.from_frame(train[["user", "item"]])
    hold = hold[~pd.MultiIndex.from_frame(hold[["user", "item"]]).isin(seen)].sort_values(["user", "item"])
    catalog = np.sort(train.item.unique()).astype(np.int64)
    return dict(catalog=catalog, users=train.user.values.astype(np.int64),
                items=np.searchsorted(catalog, train.item.values).astype(np.int32),
                ho_users=hold.user.values.astype(np.int64),
                ho_items=np.searchsorted(catalog, hold.item.values).astype(np.int32))


def gts_sets(name, unit):
    """All eligible post-cutoff events on each GTS test user's target day/second."""
    df = prep_time.LOADERS[name]()
    df = df.assign(order=np.arange(len(df))).sort_values(["user", "t", "order"], kind="stable")
    t_test = np.quantile(df.t, 0.9)
    past, fut = df[df.t < t_test], df[df.t >= t_test]
    first = prep_time.first_valid(fut, past).set_index("user").t
    cand = fut[fut.user.isin(first.index) & fut.item.isin(past.item.unique())]
    seen = pd.MultiIndex.from_frame(past[["user", "item"]])
    cand = cand[~pd.MultiIndex.from_frame(cand[["user", "item"]]).isin(seen)]
    cand = cand[(cand.t.values // unit) == (first.loc[cand.user].values // unit)]
    return cand.drop_duplicates(["user", "item"]).sort_values(["user", "item"])


if __name__ == "__main__":
    for name, load in SOURCES.items():
        df = load()
        ref = np.load(OUT / f"{name}.npz")
        for suf, mode, seed in (("_fr", "rnd", 0), ("_fv", "rev", 0), ("_fr1", "rnd", 1), ("_fr2", "rnd", 2)):
            d = fixed_target(df, mode, seed)
            assert np.array_equal(d["ho_users"], ref["ho_users"]) and np.array_equal(d["ho_items"], ref["ho_items"])
            assert np.array_equal(d["catalog"], ref["catalog"]) and np.array_equal(d["users"], ref["users"])
            assert np.array_equal(np.sort(d["items"]), np.sort(ref["items"]))
            moved = np.mean(d["items"] != ref["items"])
            np.savez(OUT / f"{name}{suf}.npz", **{k: v for k, v in d.items() if k != "ho_gap"})
            print(f"{name}{suf}: same targets and events, {moved:.1%} of training positions reordered", flush=True)
        unit = 1 if name == "ml1m" else 86400
        d = day_split(df, unit)
        k = np.unique(d["ho_users"], return_counts=True)[1]
        np.savez(OUT / f"{name}_day.npz", **d)
        print(f"{name}_day: {len(k)} test users, {len(d['ho_items'])} targets ({k.mean():.2f} per user, "
              f"{np.mean(k > 1):.1%} of users with >1)", flush=True)
        t = dict(np.load(OUT / f"{name}_t.npz"))
        sets = gts_sets(name, unit)
        cat = t["catalog"]
        assert np.all(np.isin(sets.item.values, cat)) and set(sets.user) == set(t["ho_users"])
        t["ho_users"] = sets.user.values.astype(np.int64)
        t["ho_items"] = np.searchsorted(cat, sets.item.values).astype(np.int32)
        np.savez(OUT / f"{name}_tset.npz", **t)
        k = np.unique(t["ho_users"], return_counts=True)[1]
        print(f"{name}_tset: {len(k)} test users, {len(t['ho_items'])} targets ({np.mean(k > 1):.1%} with >1)",
              flush=True)
    df = prep.LOADERS["ml20m"]()
    d = loo(df, np.random.default_rng(0).permutation(len(df)))
    np.savez(OUT / "ml20m_shuf.npz", **{k: v for k, v in d.items() if k != "ho_gap"})
    print(f"ml20m_shuf: train {len(d['items'])}, test {len(d['ho_items'])}", flush=True)
