"""Inputs for re-running TGH (haoyuhan1/GraphRec) on de-tied data. TGH evaluates on the GRID/P5 TFRecord splits of
Beauty/Sports/Toys (training = seq[:-2], evaluation = seq[:-1][-21:], testing = seq[-21:]). For Beauty and Sports
these are the S3-Rec sequences with user and item ids shifted by one; Toys uses other user and item ids, which we
align to the S3-Rec ids through the item text (Amazon 2014 metadata) and the sequences. Every original GRID record
is checked. The script writes, in the same layout and with the fields TGH reads (user_id, sequence_data; items: id,
text), with GRID's own user and item ids:

  <ds>        the original sequences (must reproduce the GRID files),
  <ds>_shuf   ties (same-day reviews) ordered at random before splitting (seed 0, as <ds>_shuf and the BSARec runs),
  <ds>_fr     the original test target kept last, all earlier ties of the user ordered at random (seed 0),

and <ds>_days.npy, the review day of every (GRID user, GRID item) pair, for the tied/other split of tgh_tied.py.
Usage: python prep_grid.py <raw dir> <processed dir> <GRID amazon_data dir> <out dir> [<meta_Toys_and_Games.json.gz>]
"""
import ast
import collections
import glob
import gzip
import json
import os
import sys

RAW, PROC, GRID, OUT = sys.argv[1:5]
META = sys.argv[5] if len(sys.argv) > 5 else None
sys.argv = [sys.argv[0], RAW, PROC]  # prep_ties reads <raw> <processed> at import

import numpy as np  # noqa: E402

import tfrecord_lite as T  # noqa: E402
from prep_ties import amazon_events  # noqa: E402

NAMES = {"beauty": "Beauty", "sports": "Sports_and_Outdoors", "toys": "Toys_and_Games"}
CUTS = (("training", lambda s: s[:-2]), ("evaluation", lambda s: s[:-1][-21:]), ("testing", lambda s: s[-21:]))


def read_split(ds, split, key="sequence_data"):
    out = {}
    for f in glob.glob(f"{GRID}/{ds}/{split}/*.tfrecord.gz"):
        for ex in T.read(f):
            out[ex["user_id" if split != "items" else "id"][0]] = ex[key]
    return out


def asin_ids(name):
    """ASIN -> S3-Rec item id (numbering by first appearance, as amazon_events)."""
    users, order = {}, []
    for line in gzip.open(f"{RAW}/amazon5/reviews_{name}_5.json.gz", "rt"):
        r = json.loads(line)
        if r["reviewerID"] not in users:
            users[r["reviewerID"]] = []
            order.append(r["reviewerID"])
        users[r["reviewerID"]].append((r["asin"], int(r["unixReviewTime"])))
    ids = {}
    for u in order:
        for a, _ in sorted(users[u], key=lambda x: x[1]):
            ids.setdefault(a, len(ids) + 1)
    return ids


def toys_alignment(df, items):
    """GRID item id -> S3-Rec item id and GRID user id -> S3-Rec user id for Toys."""
    ids = asin_ids("Toys_and_Games")
    by_title = collections.defaultdict(list)
    for line in gzip.open(META, "rt"):
        m = ast.literal_eval(line)  # the 2014 metadata files are Python literals
        if m["asin"] in ids and "title" in m:
            by_title[m["title"].strip()].append((m, ids[m["asin"]]))
    item = {}
    for g, text in items.items():
        title = text[0].decode().split("; Brand: ")[0][len("Title: "):].strip()
        cands = by_title.get(title, [])
        if len(cands) > 1:  # disambiguate repeated titles by the rest of the text
            cands = [c for c in cands if f"Price: {c[0].get('price')}" in text[0].decode()] or cands
        if len(cands) == 1:
            item[g] = cands[0][1]
    s3 = {u: list(g.item.values) for u, g in df.groupby("user", sort=True)}
    test = read_split("toys", "testing")
    index = collections.defaultdict(list)  # (length, last item) -> S3-Rec users
    for u, s in s3.items():
        index[(min(len(s), 21), s[-1])].append(u)
    user = {}
    for g, seq in test.items():
        last = item.get(seq[-1])
        cands = index.get((len(seq), last), []) if last else [u for (n, _), us in index.items() if n == len(seq) for u in us]
        cands = [u for u in cands if all(item.get(a, b) == b for a, b in zip(seq, s3[u][-len(seq):]))]
        if len(cands) == 1:
            user[g] = cands[0]
    for g, u in user.items():  # items left unresolved by the text, from the aligned users' sequences
        for a, b in zip(test[g], s3[u][-len(test[g]):]):
            item.setdefault(a, b)
    taken = set(user.values())
    for g in sorted(set(test) - set(user)):  # users with identical sequences (0.2%): paired in id order
        seq = [item[a] for a in test[g]]
        cands = [u for u in sorted(s3) if u not in taken and s3[u][-21:] == seq]
        if cands:
            user[g] = cands[0]
            taken.add(cands[0])
    print(f"toys: {len(item)}/{len(items)} items aligned by text and sequence, {len(user)}/{len(test)} users",
          flush=True)
    return item, user


def write(ds, seqs, items):
    """seqs: per GRID user (0..n-1) the full sequence in GRID item ids."""
    for split, cut in CUTS:
        os.makedirs(f"{OUT}/{ds}/{split}", exist_ok=True)
        T.write(f"{OUT}/{ds}/{split}/data_0.tfrecord.gz",
                (T.encode_example({"user_id": [u], "sequence_data": cut(s)}) for u, s in enumerate(seqs)))
    os.makedirs(f"{OUT}/{ds}/items", exist_ok=True)
    T.write(f"{OUT}/{ds}/items/data_0.tfrecord.gz",
            (T.encode_example({"id": [i], "text": [items[i][0]]}) for i in sorted(items)))


for ds, name in NAMES.items():
    if ds == "toys" and META is None:
        continue
    df = amazon_events(name)  # S3-Rec ids, review days; asserts the S3-Rec sequences are reproduced
    items = read_split(ds, "items", "text")
    if ds == "toys":
        to_s3, user_s3 = toys_alignment(df, items)
        assert len(to_s3) == len(items) and len(user_s3) == df.user.nunique(), "Toys alignment incomplete"
        to_grid = {b: a for a, b in to_s3.items()}
        grid_user = {u: g for g, u in user_s3.items()}
    else:
        to_grid = {i: i - 1 for i in df.item.unique()}
        grid_user = {u: u - 1 for u in df.user.unique()}
    df = df.assign(gu=df.user.map(grid_user), gi=df.item.map(to_grid)).sort_values("gu", kind="stable")
    groups = [g for _, g in df.groupby("gu", sort=True)]
    orig = [list(g.gi.values) for g in groups]
    for split, cut in CUTS:
        grid = read_split(ds, split)
        assert len(grid) == len(orig) and all(grid[u] == cut(s) for u, s in enumerate(orig)), (ds, split)
    rng = np.random.default_rng(0)
    shuf = df.assign(order=rng.permutation(len(df))).sort_values(["gu", "t", "order"], kind="stable")
    shuf = [list(g.gi.values) for _, g in shuf.groupby("gu", sort=True)]
    rng = np.random.default_rng(0)
    fr = []
    for g in groups:
        head = g.iloc[:-1].assign(order=rng.permutation(len(g) - 1)).sort_values(["t", "order"], kind="stable")
        fr.append(list(head.gi.values) + [g.gi.values[-1]])
    for suf, seqs in (("", orig), ("_shuf", shuf), ("_fr", fr)):
        assert all(sorted(a) == sorted(b) for a, b in zip(seqs, orig))
        write(ds + suf, seqs, items)
    np.save(f"{OUT}/{ds}_days.npy", df[["gu", "gi", "t"]].to_numpy(dtype=np.int64))
    tied = np.mean([g.t.values[-1] == g.t.values[-2] for g in groups])
    print(f"{ds}: {len(orig)} users, GRID files reproduced; tied test targets {tied:.1%}", flush=True)
