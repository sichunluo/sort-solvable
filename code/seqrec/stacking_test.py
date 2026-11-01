"""Direct test of the stacking explanation for joint-training failure: the late-fusion weight that is optimal on
in-sample positions (each user's last inner item, which SASRec was trained to predict; expert with leave-user-out
counts, i.e. out of sample) vs. the weight that is optimal on validation (both out of sample).
Usage: python stacking_test.py <dataset> (uses sasrec_s1..s5.pt)"""
import json
import sys

import numpy as np
import torch

from common import CONFIGS, PCTM, RESULTS, Data, TrainExpert, metrics, save_json, target_ranks
from model import SASRec, right_aligned

ds = sys.argv[1]
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
d, cfg = Data(ds), CONFIGS[ds]
BETAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
pctm = PCTM(d.inner, d.n_items, cfg["pctm"], cfg["max_len"], dev)
users = np.asarray([u for u, s in enumerate(d.inner) if len(s) >= 2])
sha = torch.from_numpy(d.sha_rank).to(dev)
res = {}
for seed in (1, 2, 3, 4, 5):
    m = SASRec(d.n_items, cfg["max_len"], **cfg["sas"]).to(dev)
    m.load_state_dict(torch.load(f"{RESULTS}/{ds}/sasrec_s{seed}.pt", map_location=dev))
    m.eval()
    out = {}
    with torch.no_grad():
        for luo in (True, False):
            ex = TrainExpert(pctm, cfg["max_len"], leave_user_out=luo)
            ranks = {b: [] for b in BETAS}
            for s in range(0, len(users), 256):
                seqs = [d.inner[u] for u in users[s:s + 256]]
                _, Y, S = ex.batch(seqs)
                hs = [q[:-1] for q in seqs]
                y = torch.tensor([q[-1] for q in seqs], device=dev)
                z = m.scores(m.encode(right_aligned(hs, cfg["max_len"]))[:, -1])
                excl = d.exclude(hs, dev)
                for b in BETAS:
                    ranks[b].append(target_ranks((z + b * S[:, -1]).masked_fill(excl, -float("inf")), y, sha).cpu().numpy())
            grid = {b: metrics(np.concatenate(v))["ndcg@10"] for b, v in ranks.items()}
            out["luo" if luo else "full"] = {"beta_star": max(grid, key=grid.get), "grid": grid}
    out["val_beta_star"] = json.load(open(f"{RESULTS}/{ds}/sasrec_s{seed}.json"))["val"]["beta"]
    res[seed] = out
    print(ds, seed, {k: (v["beta_star"] if isinstance(v, dict) else v) for k, v in out.items()}, flush=True)
save_json(f"{RESULTS}/{ds}/stacking_test.json", res)
