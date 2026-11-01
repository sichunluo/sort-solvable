"""Tie audit for interaction logs (checklist item 1): how often consecutive events of a user share a timestamp, and
whether a stable sort by time would leave those ties in identifier order.

For each source we read the events in file order, sort each user's events by time with a stable sort (as S3-Rec,
SASRec's DataProcessing.py, RecBole's "TO" ordering and eSASRec do), and report
  tie_pairs     share of consecutive pairs with identical timestamps (at the file's resolution)
  tied_targets  share of users whose last event ties with the previous one (the LOO target)
  file_sorted   share of adjacent file rows whose item key is non-decreasing (1.0 = file sorted by item)
  tied_asc      share of tied consecutive pairs in ascending item-key order (0.5 = no identifier order)
Usage: python audit_ties.py <name> <format> <path> [<resolution seconds>]
formats: amazon14 / amazon18 (JSON lines with asin, reviewerID, unixReviewTime), amazon23 (csv user_id,
parent_asin, rating, timestamp in ms), movielens (csv or :: separated), sasrec (reviews json -> SASRec Beauty.txt).
"""
import ast
import csv
import gzip
import io
import json
import sys
import zipfile

import numpy as np
import pandas as pd


def read(fmt, path):
    if fmt in ("amazon14", "amazon18"):
        rows = []
        with gzip.open(path, "rt") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    r = ast.literal_eval(line)  # 2014 full files are Python literals; never eval()
                rows.append((r["reviewerID"], r["asin"], int(r["unixReviewTime"])))
        return pd.DataFrame(rows, columns=["user", "item", "t"])
    if fmt == "amazon23":
        df = pd.read_csv(path)
        return pd.DataFrame({"user": df.user_id, "item": df.parent_asin, "t": df.timestamp // 1000})
    if fmt == "movielens":
        zpath, member = path.split("::", 1)
        with zipfile.ZipFile(zpath) as z, z.open(member) as f:
            if member.endswith(".csv"):
                df = pd.read_csv(f)
            else:
                df = pd.read_csv(f, sep="::", engine="python", names=["userId", "movieId", "rating", "timestamp"])
        return pd.DataFrame({"user": df.userId, "item": df.movieId, "t": df.timestamp})
    raise ValueError(fmt)


def audit(df, res=1):
    file_sorted = float(np.mean(df.item.values[1:] >= df.item.values[:-1]))
    df = df.assign(t=df.t // res, order=np.arange(len(df)))
    df = df.sort_values(["user", "t", "order"], kind="stable")
    u, t, it = df.user.values, df.t.values, df.item.values
    same_u = u[1:] == u[:-1]
    tied = same_u & (t[1:] == t[:-1])
    last = np.r_[u[1:] != u[:-1], True]
    prev_tied = np.r_[False, tied]  # event i ties with event i-1 of the same user
    return {"events": int(len(df)), "users": int(len(np.unique(u))), "items": int(df.item.nunique()),
            "tie_pairs": float(tied.sum() / same_u.sum()), "tied_targets": float(prev_tied[last].mean()),
            "file_sorted": file_sorted, "tied_asc": float(np.mean(it[1:][tied] > it[:-1][tied])) if tied.any() else None}


def sasrec_beauty(path, published):
    """Re-run kang205/SASRec data/DataProcessing.py on the 2014 review file and check it against Beauty.txt."""
    df = read("amazon14", path)
    cu, cp = df.user.value_counts(), df.item.value_counts()
    df = df[(df.user.map(cu) >= 5).values & (df.item.map(cp) >= 5).values].copy()
    df["uid"] = pd.factorize(df.user)[0] + 1   # ids by first appearance, as in the script
    df["iid"] = pd.factorize(df.item)[0] + 1
    out = df.assign(order=np.arange(len(df))).sort_values(["uid", "t", "order"], kind="stable")
    pub = pd.read_csv(published, sep=" ", header=None, names=["uid", "iid"])
    assert np.array_equal(out.uid.values, pub.uid.values) and np.array_equal(out.iid.values, pub.iid.values), \
        "SASRec Beauty.txt not reproduced"
    return df[["uid", "item", "t"]].rename(columns={"uid": "user"})


if __name__ == "__main__":
    name, fmt, path = sys.argv[1:4]
    res = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    if fmt == "sasrec":
        raw, published = path.split("::")
        df = sasrec_beauty(raw, published)
        print(f"{name}: SASRec Beauty.txt reproduced exactly from the review file", flush=True)
    else:
        df = read(fmt, path)
    print(json.dumps({"name": name, "resolution_s": res, **audit(df, res)}), flush=True)
