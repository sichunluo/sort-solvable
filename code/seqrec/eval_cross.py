"""Cross-evaluation of training order, test-history order and target definition on identical events.

Models and expert counts come from <train_ds> (its final SASRec and BERT4Rec checkpoints and its training sequences), test
histories from <hist_ds>, and test rows (possibly several per user, e.g. set-valued targets) from <target_ds>. All
three must share the catalogue and the training events; only their order or the targets differ. All methods break
equal scores by SHA-256 order. Ranks go to results/<target_ds>/cross_<train_ds>__<hist_ds>.npz.
Usage: python eval_cross.py <train_ds> <hist_ds> [<target_ds>]
"""
import glob
import json
import sys

import numpy as np
import torch

from common import CONFIGS, PCTM, RESULTS, Data, metrics, save_json, target_ranks
from model import ARCHS, right_aligned
from seqrules import SEQRULES, SeqRules

train_ds, hist_ds = sys.argv[1], sys.argv[2]
target_ds = sys.argv[3] if len(sys.argv) > 3 else hist_ds
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
cfg = CONFIGS[train_ds]
dT, dH, dY = Data(train_ds), Data(hist_ds), Data(target_ds)
for o in (dH, dY):
    assert np.array_equal(o.catalog, dT.catalog) and np.array_equal(o.user_ids, dT.user_ids)
    assert all(np.array_equal(np.sort(a), np.sort(b)) for a, b in zip(o.seqs, dT.seqs))
users, ys = dY.test_u, dY.test_y
sha = torch.from_numpy(dT.sha_rank).to(dev)

pctm = PCTM(dT.seqs, dT.n_items, cfg["pctm"], cfg["max_len"], dev)
sym = PCTM(dT.seqs, dT.n_items, cfg["pctm"], cfg["max_len"], dev)
sym.C = sym.C + sym.C.T
mc = PCTM(dT.seqs, dT.n_items, dict(cfg["pctm"], counts="exp:1:1"), 1, dev)
rules = SeqRules(dT.seqs, dT.n_items, SEQRULES[train_ds.split("_")[0]], dev)
models = {}
FINAL = {"sasrec": "sasrec", "sasrec_refit": "sasrec", "bert": "bert", "bert_refit": "bert"}  # tag -> arch
for f in sorted(glob.glob(f"{RESULTS}/{train_ds}/*_s[0-9]_refit.pt")):
    tag, seed = f.rsplit("/", 1)[1][:-len("_refit.pt")].rsplit("_s", 1)
    if tag not in FINAL:
        continue
    res = json.load(open(f"{RESULTS}/{train_ds}/{tag}_s{seed}.json"))
    sas = dict(cfg["sas"], **{k: res["args"][k] for k in ("blocks", "heads", "dropout") if res["args"].get(k) is not None})
    m = ARCHS[FINAL[tag]](dT.n_items, cfg["max_len"], **sas).to(dev)
    m.load_state_dict(torch.load(f, map_location=dev))
    beta = res["val"]["beta"]
    models[f"{FINAL[tag]}_s{seed}"] = (m.eval(), beta)
print(f"{train_ds} -> histories {hist_ds}, targets {target_ds}: {len(ys)} rows, models {sorted(models)}", flush=True)

ranks = {k: [] for k in ["pctm", "pctm_sym", "mc", "seqrules", "mostpop"] + list(models) + [f"late_{n}" for n in models]}
pop = torch.log1p(pctm.freq).float()
with torch.no_grad():
    for s in range(0, len(ys), 512):
        hs = [dH.seqs[u] for u in users[s:s + 512]]
        y = torch.from_numpy(ys[s:s + 512]).to(dev)
        ex = dY.exclude(hs, dev)
        rk = lambda sc: target_ranks(sc.float().masked_fill(ex, -float("inf")), y, sha)
        S = pctm.score_histories(hs, dtype=torch.float32)
        ranks["pctm"].append(rk(pctm.score_histories(hs)))
        ranks["pctm_sym"].append(rk(sym.score_histories(hs)))
        ranks["mc"].append(rk(mc.C[torch.tensor([h[-1] for h in hs], device=dev)]))
        ranks["seqrules"].append(rk(rules.score_histories(hs)))
        ranks["mostpop"].append(rk(pop.expand(len(hs), -1)))
        X = right_aligned(hs, cfg["max_len"])
        for n, (m, beta) in models.items():
            z = m.scores(m.last(X))
            ranks[n].append(rk(z))
            ranks[f"late_{n}"].append(rk(z + beta * S))
ranks = {k: torch.cat(v).cpu().numpy() for k, v in ranks.items()}
out = {"train": train_ds, "hist": hist_ds, "targets": target_ds, "n_rows": int(len(ys)),
       "n_users": int(len(np.unique(users))), "row_metrics": {k: metrics(v) for k, v in ranks.items()}}
print({k: round(v["ndcg@10"], 4) for k, v in out["row_metrics"].items()}, flush=True)
save_json(f"{RESULTS}/{target_ds}/cross_{train_ds}__{hist_ds}.json", out)
np.savez(f"{RESULTS}/{target_ds}/cross_{train_ds}__{hist_ds}.npz", users=dY.user_ids[users], **ranks)
