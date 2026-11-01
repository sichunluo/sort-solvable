"""Timestamp ties in the LOO benchmarks.

S3Rec built the Amazon sequences from the 2014 5-core review files with a stable sort on the day-resolution review
time, so same-day reviews keep the file order (ascending ASIN). We rebuild those sequences exactly (user k = k-th
reviewer in file order, items numbered by first appearance) to recover each event's timestamp, and write
  <name>_meta.npz : per LOO test user, the gap (seconds) between the target and the previous history event;
  <name>_shuf.npz : the LOO split of the same interactions with ties broken uniformly at random (seed 0).
ML-1M is handled the same way (ties = identical rating seconds, broken by file order in the eSASRec split).
Usage: python prep_ties.py <raw dir> <processed dir>
"""
import gzip
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])


def amazon_events(raw_name):
    """S3Rec-identical (user, item, time, file order) events."""
    users, order = {}, []
    for line in gzip.open(RAW / "amazon5" / f"reviews_{raw_name}_5.json.gz", "rt"):
        r = json.loads(line)
        if r["reviewerID"] not in users:
            users[r["reviewerID"]] = []
            order.append(r["reviewerID"])
        users[r["reviewerID"]].append((r["asin"], int(r["unixReviewTime"])))
    item2id, rows = {}, []
    for k, u in enumerate(order, 1):
        for a, t in sorted(users[u], key=lambda x: x[1]):
            item2id.setdefault(a, len(item2id) + 1)
            rows.append((k, item2id[a], t))
    df = pd.DataFrame(rows, columns=["user", "item", "t"])
    s3 = [l.split() for l in open(RAW / "s3rec" / f"{raw_name}.txt")]
    got = df.groupby("user").item.apply(lambda s: " ".join(map(str, s))).tolist()
    assert got == [" ".join(l[1:]) for l in s3], "S3Rec sequences not reproduced"
    return df


def ml1m_events():
    with zipfile.ZipFile(RAW / "ml" / "ml-1m.zip") as z, z.open("ml-1m/ratings.dat") as f:
        df = pd.read_csv(f, sep="::", engine="python", names=["user", "item", "rating", "t"])
    return df[["user", "item", "t"]].reset_index(drop=True)


def loo(df, order):
    """eSASRec LastNSplitter(n=1) with cold-user/cold-item/already-seen filtering, ties broken by `order`."""
    df = df.assign(order=order).sort_values(["user", "t", "order"], kind="stable").reset_index(drop=True)
    last = ~df.user.duplicated(keep="last")
    train, hold = df[~last], df[last]
    hold = hold[hold.user.isin(train.user.unique()) & hold.item.isin(train.item.unique())]
    seen = pd.MultiIndex.from_frame(train[["user", "item"]])
    hold = hold[~pd.MultiIndex.from_frame(hold[["user", "item"]]).isin(seen)]
    prev_t = train.groupby("user").t.last()
    catalog = np.sort(train.item.unique()).astype(np.int64)
    return dict(catalog=catalog, users=train.user.values.astype(np.int64),
                items=np.searchsorted(catalog, train.item.values).astype(np.int32),
                ho_users=hold.user.values.astype(np.int64),
                ho_items=np.searchsorted(catalog, hold.item.values).astype(np.int32),
                ho_gap=(hold.t.values - prev_t.loc[hold.user].values).astype(np.int64))


SOURCES = {"beauty": lambda: amazon_events("Beauty"), "sports": lambda: amazon_events("Sports_and_Outdoors"),
           "toys": lambda: amazon_events("Toys_and_Games"), "ml1m": ml1m_events}
if __name__ == "__main__":
    for name, load in SOURCES.items():
        df = load()
        # Original LOO tie order: file order of the (time-stable-sorted) sequences.
        orig = loo(df, np.arange(len(df)))
        ref = np.load(OUT / f"{name}.npz")
        assert len(orig["items"]) == len(ref["items"]) and np.array_equal(orig["ho_users"], ref["ho_users"])
        np.savez(OUT / f"{name}_meta.npz", ho_users=orig["ho_users"], ho_gap=orig["ho_gap"])
        for seed in range(5):  # seed 0 is the main shuffled split; 1-4 measure permutation variance (experts only)
            shuf = loo(df, np.random.default_rng(seed).permutation(len(df)))
            tag = "_shuf" if seed == 0 else f"_shuf{seed}"
            np.savez(OUT / f"{name}{tag}.npz", **{k: v for k, v in shuf.items() if k != "ho_gap"})
            np.savez(OUT / f"{name}{tag}_meta.npz", ho_users=shuf["ho_users"], ho_gap=shuf["ho_gap"])
        tie = lambda g: np.mean(g == 0)
        print(f"{name}: LOO test {len(orig['ho_items'])} (tie with previous event {tie(orig['ho_gap']):.1%}, "
              f"same day {np.mean(orig['ho_gap'] < 86400):.1%}) | shuffled-tie LOO test {len(shuf['ho_items'])} "
              f"(tie {tie(shuf['ho_gap']):.1%})", flush=True)
