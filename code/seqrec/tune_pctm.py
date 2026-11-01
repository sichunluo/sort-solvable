"""Re-tune PCTM on a split's own validation targets (grid over count window/decay, tau, history kernel, popularity
boost), then evaluate the selected configuration on test. Used to check that temporal-split conclusions do not hinge
on configurations frozen for leave-one-out. Usage: python tune_pctm.py <dataset>"""
import itertools
import sys

import numpy as np
import torch

from common import CONFIGS, PCTM, RESULTS, Data, kernel_weights, metrics, save_json, target_ranks

ds = sys.argv[1]
small = "--small" in sys.argv  # 3x3x3x3 grid around the frozen configuration (used for the large LOO splits)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
d, cfg = Data(ds), CONFIGS[ds]
COUNTS = ["exp:3:0.5", "inv:5:0.5", "inv:10:0.5", "exp:10:0.5", "inv:20:1.0", "exp:20:0.7"]
TAU = [f * cfg["pctm"]["tau"] for f in (0.1, 0.3, 1.0, 3.0, 10.0)]
KERNELS = ["exp:0.3", "exp:0.6", "exp:0.9", "tail:7:0.8:0.6", "tail:10:0.8:0.7"]
BOOSTS = [0.0, -0.04, -0.2, -0.4, -0.5]
if small:
    f = cfg["pctm"]
    COUNTS = list(dict.fromkeys([f["counts"], "inv:10:0.5", "exp:20:0.7"]))[:3]
    TAU = [f["tau"] * x for x in (0.3, 1.0, 3.0)]
    KERNELS = list(dict.fromkeys([f["kernel"], "exp:0.6", "tail:7:0.8:0.6", "tail:10:0.8:0.7"]))[:3]
    BOOSTS = list(dict.fromkeys([f["pop_boost"], 0.0, -0.2, -0.04]))[:3]


def configure(p, tau, kernel, boost):
    p.scale, p._E, p.boost, p.kspec = d.n_items / tau, None, boost, kernel
    p.kern = {l: kernel_weights(kernel, l) for l in range(1, p.L + 1)}
    p.pop = (boost * torch.log(p.freq + 1.0)).float()


def ranks(p, split):
    users, ys, hists = d.split(split)
    sha = torch.from_numpy(d.sha_rank).to(dev)
    out = []
    for s in range(0, len(users), 1024):
        hs = [hists[u] for u in users[s:s + 1024]]
        sc = p.score_histories(hs).masked_fill(d.exclude(hs, dev), -float("inf"))
        out.append(target_ranks(sc, torch.from_numpy(ys[s:s + 1024]).to(dev), sha).cpu().numpy())
    return np.concatenate(out)


SYMS = [False, True]  # True: direction-free counts C + C^T
grid = {}
for counts, sym in itertools.product(COUNTS, SYMS):
    p = PCTM(d.inner, d.n_items, dict(cfg["pctm"], counts=counts), cfg["max_len"], dev)
    if sym:
        p.C = p.C + p.C.T
    for tau, kernel, boost in itertools.product(TAU, KERNELS, BOOSTS):
        configure(p, tau, kernel, boost)
        grid[(counts + (":sym" if sym else ""), tau, kernel, boost)] = metrics(ranks(p, "val"))["ndcg@10"]
    del p
    torch.cuda.empty_cache()
best = max(grid, key=grid.get)
frozen = (cfg["pctm"]["counts"], cfg["pctm"]["tau"], cfg["pctm"]["kernel"], cfg["pctm"]["pop_boost"])
p = PCTM(d.seqs, d.n_items, dict(cfg["pctm"], counts=best[0].replace(":sym", "")), cfg["max_len"], dev)
if best[0].endswith(":sym"):
    p.C = p.C + p.C.T
configure(p, *best[1:])
r = ranks(p, "test")
res = {"dataset": ds, "best": dict(zip(("counts", "tau", "kernel", "pop_boost"), best)), "val_best": grid[best],
       "val_frozen": grid.get(frozen), "test": metrics(r), "n_configs": len(grid)}
print(f"{ds}: tuned PCTM val {grid[best]:.4f} (frozen {grid.get(frozen)}) test {res['test']['ndcg@10']:.4f} "
      f"config {best}", flush=True)
save_json(f"{RESULTS}/{ds}/pctm_tuned.json", res)
np.savez(f"{RESULTS}/{ds}/pctm_tuned_ranks.npz", test=r)
