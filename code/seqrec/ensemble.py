"""Variance-reduction null for late fusion: SASRec seed ensembles (mean logits) vs SASRec + PCTM, on LOO test.
Also the seed-vs-seed oracle selector, as a noise floor for the SASRec-vs-PCTM oracle. Oracle and Jaccard values are
means over the seeds (seed vs seed: over the pairs of consecutive seeds, cyclically).
Usage: python ensemble.py <dataset> (needs sasrec_s1..s5.pt)"""
import json
import sys

import numpy as np
import torch

from common import CONFIGS, PCTM, RESULTS, Data, metrics, save_json, target_ranks
from model import SASRec, right_aligned

ds = sys.argv[1]
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
d, cfg = Data(ds), CONFIGS[ds]
seeds = [1, 2, 3, 4, 5]
models = []
for s in seeds:
    m = SASRec(d.n_items, cfg["max_len"], **cfg["sas"]).to(dev)
    m.load_state_dict(torch.load(f"{RESULTS}/{ds}/sasrec_s{s}.pt", map_location=dev))
    models.append(m.eval())
betas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
sha = torch.from_numpy(d.sha_rank).to(dev)


@torch.no_grad()
def run(split, pctm):
    users, ys, hists = d.split(split)
    out = {}
    for s in range(0, len(users), 512):
        hs = [hists[u] for u in users[s:s + 512]]
        y = torch.from_numpy(ys[s:s + 512]).to(dev)
        ex = d.exclude(hs, dev)
        X = right_aligned(hs, cfg["max_len"])
        zs = [m.scores(m.encode(X)[:, -1]) for m in models]
        S = pctm.score_histories(hs, dtype=torch.float32)
        cands = {"single": zs[0], "ens2": (zs[0] + zs[1]) / 2, "ens3": sum(zs[:3]) / 3, "ens_all": sum(zs) / len(zs)}
        for name, z in cands.items():
            for b in betas:
                out.setdefault((name, b), []).append(
                    target_ranks((z + b * S).masked_fill(ex, -float("inf")), y, sha).cpu().numpy())
        for i in range(len(zs)):
            out.setdefault((f"seed{i}", 0.0), []).append(target_ranks(zs[i].masked_fill(ex, -float("inf")), y,
                                                                      sha).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


val = run("val", PCTM(d.inner, d.n_items, cfg["pctm"], cfg["max_len"], dev))
test = run("test", PCTM(d.seqs, d.n_items, cfg["pctm"], cfg["max_len"], dev))
nd = lambda r: np.where(r < 10, 1 / np.log2(r + 2.0), 0.0)
res = {"seeds": seeds}
for name in ("single", "ens2", "ens3", "ens_all"):
    b = max(betas, key=lambda x: metrics(val[(name, x)])["ndcg@10"])
    res[name] = {"alone": metrics(test[(name, 0.0)])["ndcg@10"], "late": metrics(test[(name, b)])["ndcg@10"],
                 "beta": b}
p = np.load(f"{RESULTS}/{ds}/experts_ranks.npz")["pctm_test"]
r = [test[(f"seed{i}", 0.0)] for i in range(len(seeds))]
pairs = [(i, (i + 1) % len(r)) for i in range(len(r))]
hit = lambda x: x < 10
jac = lambda a, b: (hit(a) & hit(b)).sum() / (hit(a) | hit(b)).sum()
res["oracle_seed_vs_seed"] = float(np.mean([np.maximum(nd(r[i]), nd(r[j])).mean() for i, j in pairs]))
res["oracle_sasrec_vs_pctm"] = float(np.mean([np.maximum(nd(x), nd(p)).mean() for x in r]))
res["jaccard_seed_vs_seed"] = float(np.mean([jac(r[i], r[j]) for i, j in pairs]))
res["jaccard_sasrec_vs_pctm"] = float(np.mean([jac(x, p) for x in r]))
print(ds, json.dumps(res), flush=True)
save_json(f"{RESULTS}/{ds}/ensemble.json", res)
