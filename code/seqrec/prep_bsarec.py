"""Inputs for re-running the BSARec code base (yehjin-shin/BSARec, which ships byte-identical S3-Rec files) on
de-tied data: <Name>_shuf.txt has the same users and items as <Name>.txt with ties (same-day reviews) ordered at
random (seed 0, as <ds>_shuf), and <Name>[_shuf]_same_target.npy is the contrastive-learning index that DuoRec and
FEARec expect, built here in one pass with the same semantics as the repository's builder (we never load the
repository's own pickled .npy files). Inputs are our verified copies of the S3-Rec files.
Usage: python prep_bsarec.py <raw dir> <processed dir> <out dir>
"""
import os
import sys

import numpy as np

RAW, PROC, OUT = sys.argv[1:4]
sys.argv = [sys.argv[0], RAW, PROC]  # prep_ties reads <raw> <processed> at import
from prep_ties import amazon_events  # noqa: E402


def same_target_index(user_seq, max_len=50):
    """For every item, the training prefixes (BSARec RecDataset 'train') that end in it, in dataset order."""
    prefixes = [s[-(max_len + 2):-2][:i + 1] for s in user_seq for i in range(len(s[-(max_len + 2):-2]))]
    buckets = [[] for _ in range(max(p[-1] for p in prefixes) + 2)]
    for p in prefixes:
        buckets[p[-1]].append(p)
    out = np.empty(len(buckets), dtype=object)
    out[:] = buckets
    return out


os.makedirs(OUT, exist_ok=True)
for name in ("Beauty", "Sports_and_Outdoors", "Toys_and_Games"):
    df = amazon_events(name)  # S3-Rec ids, asserts the published sequences are reproduced
    orig = [l.split() for l in open(f"{RAW}/s3rec/{name}.txt")]
    shuf = df.assign(order=np.random.default_rng(0).permutation(len(df)))
    shuf = shuf.sort_values(["user", "t", "order"], kind="stable")
    with open(f"{OUT}/{name}_shuf.txt", "w") as f:
        for u, g in shuf.groupby("user", sort=True):
            f.write(f"{u} " + " ".join(map(str, g.item.values)) + "\n")
    shuf_lines = [l.split() for l in open(f"{OUT}/{name}_shuf.txt")]
    assert [l[0] for l in shuf_lines] == [l[0] for l in orig]
    assert all(sorted(a[1:]) == sorted(b[1:]) for a, b in zip(shuf_lines, orig))
    moved = np.mean([x != y for a, b in zip(shuf_lines, orig) for x, y in zip(a[1:], b[1:])])
    np.save(f"{OUT}/{name}_same_target.npy", same_target_index([[int(i) for i in l[1:]] for l in orig]))
    np.save(f"{OUT}/{name}_shuf_same_target.npy", same_target_index([[int(i) for i in l[1:]] for l in shuf_lines]))
    print(f"{name}: {len(orig)} users, {moved:.1%} of positions moved", flush=True)
