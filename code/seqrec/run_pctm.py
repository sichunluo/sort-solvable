"""Training-free experts on val and test: PCTM (frozen config) and first-order MC. Checks PCTM/MC against Table 1."""
import sys

import numpy as np
import torch

from common import CONFIGS, PCTM, REPORTED_NDCG10, RESULTS, Data, metrics, save_json, target_ranks

name = sys.argv[1]
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
d, cfg = Data(name), CONFIGS[name]
out = {"dataset": name, "n_items": d.n_items, "n_val": len(d.val_u), "n_test": len(d.test_u)}
ranks = {}
for split in ("val", "test"):
    users, ys, hists = d.split(split)
    pctm = PCTM(hists, d.n_items, cfg["pctm"], cfg["max_len"], dev)
    mc = PCTM(hists, d.n_items, dict(cfg["pctm"], counts="exp:1:1"), 1, dev)
    sha = torch.from_numpy(d.sha_rank).to(dev)
    # MC ties: ascending external id reproduces the published numbers on the original splits; elsewhere the id can
    # encode item age (GTS ids are sorted ASINs), so variants use the SHA-256 order like PCTM.
    ext = torch.arange(d.n_items, device=dev) if name in REPORTED_NDCG10 else sha
    rp, rm = [], []
    for s in range(0, len(users), 1024):
        hs = [hists[u] for u in users[s:s + 1024]]
        y = torch.from_numpy(ys[s:s + 1024]).to(dev)
        seen = d.exclude(hs, dev)
        sc = pctm.score_histories(hs)
        sc[seen] = -float("inf")
        rp.append(target_ranks(sc, y, sha).cpu().numpy())
        sc = mc.C[torch.tensor([h[-1] for h in hs], device=dev)].clone()
        sc[seen] = -float("inf")
        rm.append(target_ranks(sc, y, ext).cpu().numpy())
    ranks[f"pctm_{split}"], ranks[f"mc_{split}"] = np.concatenate(rp), np.concatenate(rm)
    out[f"pctm_{split}"], out[f"mc_{split}"] = metrics(ranks[f"pctm_{split}"]), metrics(ranks[f"mc_{split}"])
    del pctm, mc
    torch.cuda.empty_cache()
out["reported"] = REPORTED_NDCG10.get(name, {})
rep = lambda k: f" (reported {out['reported'][k]:.4f})" if k in out["reported"] else ""
print(f"{name}: PCTM test ndcg@10={out['pctm_test']['ndcg@10']:.4f}{rep('pctm')}  "
      f"MC={out['mc_test']['ndcg@10']:.4f}{rep('mc')}  val PCTM={out['pctm_val']['ndcg@10']:.4f}", flush=True)
save_json(f"{RESULTS}/{name}/experts.json", out)
np.savez(f"{RESULTS}/{name}/experts_ranks.npz", **ranks)
