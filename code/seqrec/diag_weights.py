"""Two diagnostics on test users: (1) effective expert weight in late fusion vs PoE, measured as
std(beta*S) / std(z) over unseen items (whole catalogue and the fused top-100); (2) hop distance split beyond 3
(reachable 4-5, 6+, unreachable) with NDCG@10 of the saved test ranks."""
import json
import sys

import numpy as np
import torch

from common import CONFIGS, PCTM, RESULTS, Data, save_json
from model import SASRec, right_aligned

ds, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
cfg, d = CONFIGS[ds], Data(ds)
pctm = PCTM(d.seqs, d.n_items, cfg["pctm"], cfg["max_len"], dev)


def load(tag):
    m = SASRec(d.n_items, cfg["max_len"], **cfg["sas"]).to(dev)
    m.load_state_dict(torch.load(f"{RESULTS}/{ds}/{tag}_s{seed}.pt", map_location=dev))
    return m.eval()


@torch.no_grad()
def ratios(model, beta, bs=512):
    full, top = [], []
    for s in range(0, len(d.test_u), bs):
        hs = [d.seqs[u] for u in d.test_u[s:s + bs]]
        z = model.scores(model.encode(right_aligned(hs, model.max_len))[:, -1])
        e = beta * pctm.score_histories(hs, dtype=torch.float32)
        keep = ~d.exclude(hs, dev)
        zm, em = z.masked_fill(~keep, float("nan")), e.masked_fill(~keep, float("nan"))
        sd = lambda x: (x - x.nanmean(1, keepdim=True)).pow(2).nanmean(1).sqrt()
        full.append((sd(em) / sd(zm)).cpu())
        idx = (z + e).masked_fill(~keep, -float("inf")).topk(100, 1).indices
        top.append((e.gather(1, idx).std(1) / z.gather(1, idx).std(1)).cpu())
    return float(torch.cat(full).median()), float(torch.cat(top).median())


out = {}
late = json.load(open(f"{RESULTS}/{ds}/sasrec_s{seed}.json"))["val"]["beta"]
out["late"] = dict(beta=late, ratio_full_top100=ratios(load("sasrec"), late))
poe = load("poe_luo1")
out["poe"] = dict(beta=poe.beta.item(), ratio_full_top100=ratios(poe, poe.beta.item()))
try:
    fix = load("poe_luo1_fix")
    out["poe_fix"] = dict(beta=fix.beta.item(), ratio_full_top100=ratios(fix, fix.beta.item()))
except FileNotFoundError:
    pass
print(ds, "seed", seed, "median std(beta*S)/std(z) [catalogue, fused top-100]:", json.dumps(out), flush=True)

# hop distances up to 10 in the 1-step transition graph of full train
n = d.n_items
src = np.concatenate([s[:-1] for s in d.seqs]).astype(np.int64)
dst = np.concatenate([s[1:] for s in d.seqs]).astype(np.int64)
A = torch.zeros(n, n, dtype=torch.float16, device=dev)
A[torch.from_numpy(src).to(dev), torch.from_numpy(dst).to(dev)] = 1
indeg0 = (A.sum(0) == 0).cpu().numpy()
last = torch.tensor([d.seqs[u][-1] for u in d.test_u], device=dev)
y = torch.from_numpy(d.test_y).to(dev)
hop = torch.full((len(y),), 99, dtype=torch.long, device=dev)
for s in range(0, len(y), 2048):
    F = torch.zeros(min(2048, len(y) - s), n, dtype=torch.float16, device=dev)
    F[torch.arange(F.shape[0], device=dev), last[s:s + 2048]] = 1
    o = hop[s:s + 2048]
    for h in range(1, 11):
        F = ((F @ A) > 0).half()
        o[(o == 99) & (F.gather(1, y[s:s + 2048, None])[:, 0] > 0)] = h
hop = hop.cpu().numpy()
nd = lambda r: np.where(r < 10, 1 / np.log2(r + 2.0), 0.0)
ranks = {"PCTM": np.load(f"{RESULTS}/{ds}/experts_ranks.npz")["pctm_test"]}
for tag, key, name in (("sasrec", "alone", "SASRec"), ("sasrec", "late", "late"), ("poe_luo1", "alone", "z_theta")):
    ranks[name] = np.load(f"{RESULTS}/{ds}/{tag}_s{seed}_ranks.npz")[key]
buckets = {"4-5": (hop >= 4) & (hop <= 5), "6-10": (hop >= 6) & (hop <= 10), "unreachable": hop == 99}
print(f"{ds}: hop>=4 split (share of test; target in-degree 0 among unreachable: "
      f"{indeg0[d.test_y[hop == 99]].mean() if (hop == 99).any() else float('nan'):.2f})")
hop_split = {}
for b, m in buckets.items():
    print(f"  {b:<12} share {m.mean():.3f}  " + "  ".join(f"{k} {nd(r)[m].mean():.4f}" for k, r in ranks.items()))
    hop_split[b] = {"share": float(m.mean()), **{k: float(nd(r)[m].mean()) for k, r in ranks.items()}}
save_json(f"{RESULTS}/{ds}/diag_weights_s{seed}.json", {"std_ratio_catalogue_top100": out, "hop_split": hop_split})
