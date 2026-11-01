"""Extra training-free baselines on a split's test targets: MostPop, RecentPop (30 days before the last train event,
when event times are available), a direction-free PCTM (counts C + C^T, same configuration) and Sequential Rules
(frozen capacity-probes configuration; equal scores broken by ascending external item id on the original splits, by
SHA-256 order on all variants).
Usage: python run_baselines.py <dataset>"""
import sys

import numpy as np
import torch

from common import CONFIGS, PCTM, REPORTED_NDCG10, RESULTS, Data, metrics, save_json, target_ranks
from seqrules import SEQRULES, SeqRules

ds = sys.argv[1]
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
d, cfg = Data(ds), CONFIGS[ds]
sha = torch.from_numpy(d.sha_rank).to(dev)


def baselines(split):
    """Baseline ranks on a split, with counts from that split's history (inner data for val, full for test)."""
    users, ys, hists = d.split(split)
    seqs = d.inner if split == "val" else d.seqs
    pctm = PCTM(seqs, d.n_items, cfg["pctm"], cfg["max_len"], dev)
    sym = PCTM(seqs, d.n_items, cfg["pctm"], cfg["max_len"], dev)
    sym.C = sym.C + sym.C.T
    scores = {"mostpop": torch.log1p(pctm.freq).float()}
    if d.times is not None:
        t_all = np.concatenate([t[:len(s)] for t, s in zip(d.times, seqs)])
        items = np.concatenate(seqs)
        recent = items[t_all >= t_all.max() - 30 * 86400]
        scores["recentpop"] = torch.from_numpy(np.log1p(np.bincount(recent, minlength=d.n_items))
                                               + 1e-6 * np.log1p(pctm.freq.cpu().numpy())).float().to(dev)
    rules = SeqRules(seqs, d.n_items, SEQRULES[ds.split("_")[0]], dev)
    ranks = {k: [] for k in list(scores) + ["pctm_sym", "seqrules"]}
    for s in range(0, len(users), 1024):
        hs = [hists[u] for u in users[s:s + 1024]]
        y = torch.from_numpy(ys[s:s + 1024]).to(dev)
        ex = d.exclude(hs, dev)
        for k, v in scores.items():
            ranks[k].append(target_ranks(v.expand(len(hs), -1).masked_fill(ex, -float("inf")), y, sha).cpu().numpy())
        ranks["pctm_sym"].append(target_ranks(sym.score_histories(hs).masked_fill(ex, -float("inf")), y, sha).cpu().numpy())
        ranks["seqrules"].append(target_ranks(rules.score_histories(hs).masked_fill(ex, -float("inf")), y, ext).cpu().numpy())
    return {k: np.concatenate(v) for k, v in ranks.items()}


ext = torch.arange(d.n_items, device=dev) if ds in REPORTED_NDCG10 else sha  # as in run_pctm.py
ranks = baselines("test")
ranks.update({f"{k}_val": v for k, v in baselines("val").items()})

out = {k: metrics(v) for k, v in ranks.items()}
print(ds, {k: round(v["ndcg@10"], 4) for k, v in out.items()}, flush=True)
save_json(f"{RESULTS}/{ds}/baselines.json", out)
np.savez(f"{RESULTS}/{ds}/baselines_ranks.npz", **ranks)
