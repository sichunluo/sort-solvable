"""Prospective GTS for Beauty: 5-core filtering uses only events before the test cutoff (reviewer control for the
globally filtered 2014 5-core files). Starts from the full 2014 review file, keeps the cutoff dates of beauty_t,
filters users and items with fewer than 5 pre-cutoff events iteratively, keeps the post-cutoff events of the
surviving users and items, and builds the same GTS split (beauty_tp) and its training-ties-shuffled twin
(beauty_tpx). Usage: python prep_prospective.py <raw dir> <processed dir> <reviews_Beauty.json.gz>
"""
import sys

import numpy as np
import pandas as pd

RAW, PROC, FULL = sys.argv[1:4]
sys.argv = [sys.argv[0], RAW, PROC]
import prep_time  # noqa: E402
from audit_ties import read  # noqa: E402

df = read("amazon14", FULL)  # file order (sorted by ASIN), day-resolution times
t_val, t_test = np.load(f"{PROC}/beauty_t.npz")["cutoffs"]
past = df[df.t < t_test]
while True:  # iterative 5-core on the past only
    u, i = past.user.value_counts(), past.item.value_counts()
    keep = past.user.map(u).ge(5).values & past.item.map(i).ge(5).values
    if keep.all():
        break
    past = past[keep]
fut = df[(df.t >= t_test) & df.user.isin(past.user.unique()) & df.item.isin(past.item.unique())]
ev = pd.concat([past, fut]).sort_index()  # back to file order
ev = ev.drop_duplicates(["user", "item"], keep="first")
ev["user"] = pd.factorize(ev.user, sort=True)[0] + 1
ev["item"] = pd.factorize(ev.item, sort=True)[0] + 1
ev = ev.reset_index(drop=True)
g = np.load(f"{PROC}/beauty_t.npz")
for name, kw in (("beauty_tp", {}), ("beauty_tpx", dict(shuffle_ties=True, fixed_test=True))):
    d = prep_time.split(ev, cut=(t_val, t_test), **kw)
    np.savez(f"{PROC}/{name}.npz", **d)
    print(f"{name}: train users={len(np.unique(d['users']))} items={len(d['catalog'])} events={len(d['items'])} "
          f"val={len(d['val_items'])} test={len(d['ho_items'])} (globally filtered beauty_t: test={len(g['ho_items'])}, "
          f"items={len(g['catalog'])})", flush=True)
