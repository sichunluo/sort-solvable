"""Rebuild the eSASRec leave-one-out splits (as used by Petrov et al., RecSys'26) into compact npz files.

Mirrors transformer_benchmark@2b03992 src/datasets/{s3_repro,ml_1m,ml_20m}.py + LastNSplitter(n=1,
filter_cold_users, filter_cold_items, filter_already_seen). Row counts are checked against the
capacity-probes data manifest.

Output per dataset (data/seqrec/processed/<name>.npz):
  catalog   sorted external ids of items present in train; internal id = index into catalog
  users     external user id per train event, ordered by (user, datetime, source order)
  items     internal item id per train event (same order)
  ho_users  external user id per holdout row (sorted)
  ho_items  internal item id per holdout row
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(sys.argv[1])
OUT = Path(sys.argv[2])
EXPECTED = {  # (train rows, holdout rows) from capacity-probes data/manifest.json
    "beauty": (176139, 22311), "sports": (260739, 35539), "toys": (148185, 19365),
    "ml1m": (994169, 6038), "ml20m": (19861770, 138456),
}


def read_s3(name):
    users, items, times = [], [], []
    with open(RAW / "s3rec" / f"{name}.txt") as f:
        for line in f:
            user, rest = line.strip().split(" ", 1)
            seq = rest.split(" ")
            users += [int(user)] * len(seq)
            items += [int(i) for i in seq]
            times += list(range(1, len(seq) + 1))
    return pd.DataFrame({"user": users, "item": items, "t": times})


def read_ml(zip_name, member, sep, header):
    with zipfile.ZipFile(RAW / "ml" / zip_name) as z, z.open(member) as f:
        if header:
            df = pd.read_csv(f)
        else:
            df = pd.read_csv(f, sep=sep, engine="python", names=["userId", "movieId", "rating", "timestamp"])
    return pd.DataFrame({"user": df.userId.values, "item": df.movieId.values, "t": df.timestamp.values})


def split(df):
    df = df.assign(order=np.arange(len(df)))
    df = df.sort_values(["user", "t", "order"], kind="stable").reset_index(drop=True)
    last = ~df.user.duplicated(keep="last")
    train, hold = df[~last], df[last]
    hold = hold[hold.user.isin(train.user.unique())]                     # cold users
    hold = hold[hold.item.isin(train.item.unique())]                     # cold items
    seen = pd.MultiIndex.from_frame(train[["user", "item"]])
    hold = hold[~pd.MultiIndex.from_frame(hold[["user", "item"]]).isin(seen)]  # already seen
    catalog = np.sort(train.item.unique()).astype(np.int64)
    return dict(
        catalog=catalog,
        users=train.user.values.astype(np.int64),
        items=np.searchsorted(catalog, train.item.values).astype(np.int32),
        ho_users=hold.user.values.astype(np.int64),
        ho_items=np.searchsorted(catalog, hold.item.values).astype(np.int32),
    )


LOADERS = {
    "beauty": lambda: read_s3("Beauty"),
    "sports": lambda: read_s3("Sports_and_Outdoors"),
    "toys": lambda: read_s3("Toys_and_Games"),
    "ml1m": lambda: read_ml("ml-1m.zip", "ml-1m/ratings.dat", "::", False),
    "ml20m": lambda: read_ml("ml-20m.zip", "ml-20m/ratings.csv", ",", True),
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[3:] or LOADERS:
        d = split(LOADERS[name]())
        got = (len(d["items"]), len(d["ho_items"]))
        status = "OK" if got == EXPECTED[name] else f"MISMATCH expected {EXPECTED[name]}"
        print(f"{name}: users={len(np.unique(d['users']))} items={len(d['catalog'])} "
              f"train={got[0]} holdout={got[1]} {status}", flush=True)
        np.savez(OUT / f"{name}.npz", **d)
