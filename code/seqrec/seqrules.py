"""Dense GPU Sequential Rules (Ludewig & Jannach), as configured in capacity-probes configs/table1.json.

R = sum_d g(d) N^(d) over distances d <= steps; rows optionally scaled by min-max normalized IDF and pruned to the
K strongest targets (ties by ascending item id); score(a|h) = sum_j v(j) R[h_j, a] over the newest L history items
with v(1) = 1 and v(j) = weighting(j) for j >= 2. Equal scores are broken by ascending external item id.
"""
import numpy as np
import torch

SEQRULES = {  # frozen capacity-probes configurations
    "beauty": dict(steps=10, weighting="log", pruning=100, idf=False, history_length=10, history_weighting="quadratic"),
    "sports": dict(steps=30, weighting="log", pruning=0, idf=False, history_length=20, history_weighting="div"),
    "toys": dict(steps=30, weighting="log", pruning=0, idf=False, history_length=10, history_weighting="div"),
    "ml1m": dict(steps=10, weighting="div", pruning=200, idf=True, history_length=10, history_weighting="quadratic"),
    "ml20m": dict(steps=20, weighting="div", pruning=100, idf=True, history_length=5, history_weighting="quadratic"),
}


def positional(weighting, positions):
    positions = np.asarray(positions, dtype=np.float64)
    if weighting == "div":
        return 1.0 / positions
    if weighting == "quadratic":
        return 1.0 / positions ** 2
    if weighting == "log":
        return 1.0 / np.log10(positions + 1.7)
    raise ValueError(weighting)


class SeqRules:
    def __init__(self, seqs, n_items, cfg, device):
        self.n, self.cfg, self.device = n_items, cfg, device
        users = np.repeat(np.arange(len(seqs)), [len(s) for s in seqs])
        items = np.concatenate(seqs).astype(np.int64)
        R = torch.zeros(n_items * n_items, dtype=torch.float32, device=device)
        for d, w in enumerate(positional(cfg["weighting"], np.arange(1, cfg["steps"] + 1)), 1):
            same = users[:-d] == users[d:]
            idx = torch.from_numpy(items[:-d][same] * n_items + items[d:][same]).to(device)
            R.index_add_(0, idx, torch.full(idx.shape, float(w), device=device))
        R = R.view(n_items, n_items)
        if cfg["idf"]:
            freq = np.bincount(items, minlength=n_items).astype(np.float64)
            raw = np.log(len(seqs) / np.maximum(freq, 1e-12))
            idf = (raw - raw.min()) / (raw.max() - raw.min()) if raw.max() > raw.min() else np.ones(n_items)
            R *= torch.from_numpy(idf.astype(np.float32)).to(device)[:, None]
        if cfg["pruning"] > 0:
            for s in range(0, n_items, 2048):
                block = R[s:s + 2048]
                order = torch.sort(block, dim=1, descending=True, stable=True).indices
                drop = order[:, cfg["pruning"]:]
                block.scatter_(1, drop, 0.0)
        self.R = R
        L = cfg["history_length"]
        self.hw = {l: np.r_[1.0, positional(cfg["history_weighting"], np.arange(2, l + 1))] if l > 1 else np.ones(1)
                   for l in range(1, L + 1)}

    def score_histories(self, hists):
        L = self.cfg["history_length"]
        hs = [h[-L:][::-1] for h in hists]
        Lb = max(len(h) for h in hs)
        idx = np.zeros((len(hs), Lb), dtype=np.int64)
        w = np.zeros((len(hs), Lb), dtype=np.float32)
        for b, h in enumerate(hs):
            idx[b, :len(h)] = h
            w[b, :len(h)] = self.hw[len(h)]
        idx, w = torch.from_numpy(idx).to(self.device), torch.from_numpy(w).to(self.device)
        out = torch.empty(len(hs), self.n, dtype=torch.float32, device=self.device)
        step = max(1, int(1.2e8 // (Lb * self.n)))
        for s in range(0, len(hs), step):
            out[s:s + step] = torch.bmm(w[s:s + step, None, :], self.R[idx[s:s + step]])[:, 0]
        return out
