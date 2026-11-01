"""Plan-B probe: context-dependent fusion of a trained SASRec and PCTM, fitted on held-out (inner val) positions.

fused = alpha(f) * z_sasrec + beta(f) * S_pctm with alpha, beta = softplus(linear(f)) over a few per-user features.
The gate is fitted on 80% of val users, early-stopped on the other 20%, and compared with the global late-fusion beta.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from common import CONFIGS, PCTM, RESULTS, Data, metrics, save_json, target_ranks
from model import SASRec, right_aligned

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
cfg = CONFIGS[args.dataset]
d = Data(args.dataset)
model = SASRec(d.n_items, cfg["max_len"], **cfg["sas"]).to(dev)
model.load_state_dict(torch.load(f"{RESULTS}/{args.dataset}/sasrec_s{args.seed}.pt", map_location=dev))
model.eval()
late = json.load(open(f"{RESULTS}/{args.dataset}/sasrec_s{args.seed}.json"))
beta_late = late["val"]["beta"]


@torch.no_grad()
def collect(split, pctm, bs=512):
    users, ys, hists = d.split(split)
    Z, S, M, feats = [], [], [], []
    deg = (pctm.C > 0).sum(1).float()
    for s in range(0, len(users), bs):
        hs = [hists[u] for u in users[s:s + bs]]
        z = model.scores(model.encode(right_aligned(hs, model.max_len))[:, -1])
        e = pctm.score_histories(hs, dtype=torch.float32)
        seen = d.exclude(hs, dev)
        z[seen], e[seen] = -1e4, -1e4
        tz, te = z.topk(10, 1), e.topk(10, 1)
        last = torch.tensor([h[-1] for h in hs], device=dev)
        overlap = (tz.indices[:, :, None] == te.indices[:, None, :]).any(2).float().mean(1)
        f = torch.stack([torch.log1p(torch.tensor([float(len(h)) for h in hs], device=dev)),
                         te.values[:, 0] - te.values[:, 9], tz.values[:, 0] - tz.values[:, 9],
                         torch.log1p(deg[last]), torch.log1p(pctm.freq[last].float()), overlap], 1)
        Z.append(z.half()), S.append(e.half()), M.append(seen), feats.append(f)
    return torch.cat(Z), torch.cat(S), torch.cat(M), torch.cat(feats), torch.from_numpy(ys).to(dev)


def fused(Z, S, M, f, w, rows):
    g = F.softplus(f[rows] @ w[:, :-1].T + w[:, -1])  # (B, 2): alpha, beta
    out = g[:, :1] * Z[rows].float() + g[:, 1:] * S[rows].float()
    return out.masked_fill(M[rows], -float("inf"))


pctm = PCTM(d.inner, d.n_items, cfg["pctm"], cfg["max_len"], dev)
Zv, Sv, Mv, fv, yv = collect("val", pctm)
mu, sd = fv.mean(0), fv.std(0) + 1e-6
fv = (fv - mu) / sd
perm = torch.from_numpy(np.random.default_rng(0).permutation(len(yv))).to(dev)
tr, ho = perm[: int(0.8 * len(yv))], perm[int(0.8 * len(yv)):]
inv_sp = lambda x: float(np.log(np.expm1(x)))
w = torch.zeros(2, fv.shape[1] + 1, device=dev)
w[0, -1], w[1, -1] = inv_sp(1.0), inv_sp(beta_late)
w.requires_grad_(True)
opt = torch.optim.Adam([w], lr=0.02)
sha = torch.from_numpy(d.sha_rank).to(dev)


def ndcg_on(rows, w_):
    with torch.no_grad():
        r = torch.cat([target_ranks(fused(Zv, Sv, Mv, fv, w_, rows[i:i + 1024]), yv[rows[i:i + 1024]], sha)
                       for i in range(0, len(rows), 1024)])
    return metrics(r.cpu().numpy())["ndcg@10"]


best, best_w, bad = ndcg_on(ho, w), w.detach().clone(), 0
start = best
for step in range(400):
    rows = tr[torch.randint(len(tr), (1024,), device=dev)]
    loss = F.cross_entropy(fused(Zv, Sv, Mv, fv, w, rows), yv[rows])
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 20 == 19:
        cur = ndcg_on(ho, w)
        if cur > best:
            best, best_w, bad = cur, w.detach().clone(), 0
        else:
            bad += 1
            if bad >= 5:
                break
del Zv, Sv, Mv, pctm
torch.cuda.empty_cache()
pctm = PCTM(d.seqs, d.n_items, cfg["pctm"], cfg["max_len"], dev)
Zt, St, Mt, ft, yt = collect("test", pctm)
ft = (ft - mu) / sd
rows = torch.arange(len(yt), device=dev)
with torch.no_grad():
    r_gate = torch.cat([target_ranks(fused(Zt, St, Mt, ft, best_w, rows[i:i + 1024]), yt[i:i + 1024], sha)
                        for i in range(0, len(yt), 1024)]).cpu().numpy()
res = {"seed": args.seed, "beta_late": beta_late, "heldout_val_ndcg": {"late_start": start, "gate": best},
       "test": {"gate": metrics(r_gate), "late": late["test"]["late"], "sasrec": late["test"]["alone"]},
       "w": best_w.tolist(),
       "features": ["log_len", "expert_gap", "sasrec_gap", "log_outdeg_last", "log_pop_last", "top10_overlap"]}
print("GATE", args.dataset, json.dumps(res["test"]), "heldout val", res["heldout_val_ndcg"], flush=True)
save_json(f"{RESULTS}/{args.dataset}/gate_s{args.seed}.json", res)
np.savez(f"{RESULTS}/{args.dataset}/gate_s{args.seed}_ranks.npz", gate=r_gate)
