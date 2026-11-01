"""Go/no-go summary: test metrics over seeds, PoE vs late fusion, hit overlap with PCTM, hop-distance buckets."""
import glob
import json
import os
import sys

import numpy as np
import torch

from common import RESULTS, Data

K = 10
datasets = sys.argv[1:] or ["beauty", "sports", "toys", "ml1m", "ml20m"]
VARIANTS = [("pctm", None, None), ("mc", None, None), ("sasrec", "alone", "SASRec"), ("sasrec", "late", "Late fusion"),
            ("poe_luo1", "fused", "PoE (LUO)"), ("poe_luo1", "alone", "PoE (LUO), expert dropped"),
            ("poe_luo0", "fused", "PoE (no LUO)"), ("poe_luo1", "tuned", "PoE (LUO) + val-tuned β"),
            ("poe_luo1_fix", "fused", "PoE (LUO), β fixed at late β*"), ("gate", "gate", "Contextual gate")]
TAGS = ("sasrec", "poe_luo1", "poe_luo0", "poe_luo1_fix", "gate")


def ndcg(r):
    return np.where(r < K, 1.0 / np.log2(r + 2.0), 0.0)


def hops(d, max_hop=3, bs=2048):
    """Shortest directed distance from each test user's last train item to the target in the 1-step graph."""
    dev = "cuda"
    users = d.seqs
    n = d.n_items
    src = np.concatenate([s[:-1] for s in users]).astype(np.int64)
    dst = np.concatenate([s[1:] for s in users]).astype(np.int64)
    A = torch.zeros(n, n, dtype=torch.float16, device=dev)
    A[torch.from_numpy(src).to(dev), torch.from_numpy(dst).to(dev)] = 1
    last = torch.tensor([users[u][-1] for u in d.test_u], device=dev)
    y = torch.from_numpy(d.test_y).to(dev)
    out = torch.full((len(y),), max_hop + 1, dtype=torch.long, device=dev)
    for s in range(0, len(y), bs):
        F = torch.zeros(min(bs, len(y) - s), n, dtype=torch.float16, device=dev)
        F[torch.arange(F.shape[0], device=dev), last[s:s + bs]] = 1
        o = out[s:s + bs]
        for h in range(1, max_hop + 1):
            F = ((F @ A) > 0).half()
            hit = F.gather(1, y[s:s + bs, None])[:, 0] > 0
            o[(o > max_hop) & hit] = h
    return out.cpu().numpy()


lines = []
for ds in datasets:
    if not os.path.exists(f"{RESULTS}/{ds}/experts_ranks.npz"):
        continue
    d = Data(ds)
    ex = np.load(f"{RESULTS}/{ds}/experts_ranks.npz")
    ranks = {("pctm", None): [ex["pctm_test"]], ("mc", None): [ex["mc_test"]]}
    for tag in TAGS:
        for f in sorted(glob.glob(f"{RESULTS}/{ds}/{tag}_s*_ranks.npz")):
            z = np.load(f)
            for key in z.files:
                ranks.setdefault((tag, key), []).append(z[key])
    hop = hops(d)
    lines += [f"\n## {ds}  (test users {len(d.test_u)}, items {d.n_items})\n",
              "| method | seeds | NDCG@10 | Recall@10 | " + " | ".join(
                  f"hop {h} ({np.mean(hop == h):.0%})" for h in (1, 2, 3)) + f" | hop ≥4/∞ ({np.mean(hop > 3):.0%}) |",
              "|---|---|---|---|---|---|---|---|"]
    for tag, key, label in VARIANTS:
        rs = ranks.get((tag, key))
        if not rs:
            continue
        nd = [ndcg(r).mean() for r in rs]
        rc = [np.mean(r < K) for r in rs]
        buckets = [np.mean([ndcg(r)[hop == h].mean() for r in rs]) for h in (1, 2, 3)]
        buckets.append(np.mean([ndcg(r)[hop > 3].mean() for r in rs]))
        sd = f" ± {np.std(nd):.4f}" if len(rs) > 1 else ""
        lines.append(f"| {label or tag.upper()} | {len(rs)} | {np.mean(nd):.4f}{sd} | {np.mean(rc):.4f} | "
                     + " | ".join(f"{b:.4f}" for b in buckets) + " |")
    # paired comparison per seed
    poe, late, alone = ranks.get(("poe_luo1", "fused"), []), ranks.get(("sasrec", "late"), []), ranks.get(("sasrec", "alone"), [])
    m = min(len(poe), len(late))
    if m:
        diffs = [ndcg(poe[i]).mean() - ndcg(late[i]).mean() for i in range(m)]
        lines.append(f"\nPoE(LUO) − late fusion, NDCG@10 per seed: {', '.join(f'{x:+.4f}' for x in diffs)}")
    best_expert = ndcg(ex["pctm_test"]).mean()
    if poe:
        lines.append(f"PoE(LUO) vs PCTM: {np.mean([ndcg(r).mean() for r in poe]) / best_expert - 1:+.1%};  "
                     f"late fusion vs PCTM: {np.mean([ndcg(r).mean() for r in late]) / best_expert - 1:+.1%}" if late else "")
    if alone:
        hs, hp = alone[0] < K, ex["pctm_test"] < K
        lines.append(f"Hit@10 overlap SASRec vs PCTM (seed 0): Jaccard {np.sum(hs & hp) / np.sum(hs | hp):.2f}; "
                     f"P(PCTM hit | SASRec hit) {np.mean(hp[hs]):.2f}; P(SASRec hit | PCTM hit) {np.mean(hs[hp]):.2f}; "
                     f"oracle union Recall@10 {np.mean(hs | hp):.4f} (SASRec {np.mean(hs):.4f}, PCTM {np.mean(hp):.4f})")
    for tag in TAGS[:4]:
        js = sorted(glob.glob(f"{RESULTS}/{ds}/{tag}_s[0-9].json"))
        if js:
            info = [json.load(open(f)) for f in js]
            lines.append(f"{tag}: best epochs {[i['best_epoch'] for i in info]}, "
                         f"beta {[round(i['test']['beta'], 2) for i in info]}")
text = "\n".join(lines)
print(text)
with open(f"{RESULTS}/summary.md", "w") as f:
    f.write(text + "\n")
