"""Training-free side of the data-access control: build the experts from the inner split (each user's last training
interaction withheld, exactly the data the non-refit neural model never trains on) and evaluate on the LOO test
targets with full histories. Usage: python expert_access.py <dataset>"""
import sys

import numpy as np
import torch

from common import CONFIGS, PCTM, RESULTS, Data, metrics, save_json, target_ranks
from seqrules import SEQRULES, SeqRules

ds = sys.argv[1]
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
d, cfg = Data(ds), CONFIGS[ds]
users, ys, hists = d.split("test")
sha, ext = torch.from_numpy(d.sha_rank).to(dev), torch.arange(d.n_items, device=dev)
out = {}
for counts_from, seqs in (("full", d.seqs), ("inner", d.inner)):
    pctm = PCTM(seqs, d.n_items, cfg["pctm"], cfg["max_len"], dev)
    rules = SeqRules(seqs, d.n_items, SEQRULES[ds.split("_")[0]], dev)
    rp, rr = [], []
    for s in range(0, len(users), 1024):
        hs = [hists[u] for u in users[s:s + 1024]]
        y = torch.from_numpy(ys[s:s + 1024]).to(dev)
        ex = d.exclude(hs, dev)
        rp.append(target_ranks(pctm.score_histories(hs).masked_fill(ex, -float("inf")), y, sha).cpu().numpy())
        rr.append(target_ranks(rules.score_histories(hs).masked_fill(ex, -float("inf")), y, ext).cpu().numpy())
    out[counts_from] = {"pctm": metrics(np.concatenate(rp)), "seqrules": metrics(np.concatenate(rr))}
    del pctm, rules
    torch.cuda.empty_cache() if dev == "cuda" else None
print(ds, {k: {m: round(v["ndcg@10"], 4) for m, v in r.items()} for k, r in out.items()}, flush=True)
save_json(f"{RESULTS}/{ds}/expert_access.json", out)
