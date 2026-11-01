"""Rolling evaluation inside the temporal validation window, stratified by the gap to the previous event.

Models and counts use only events before t_val (stage-1 SASRec/GRU checkpoints of the _t runs, PCTM/MC/MostPop on
the inner split). Every event in [t_val, t_test) whose item exists before t_val and is new to the user is a target,
scored with the user's full history up to that event. This keeps short-gap (same-day) continuations, which LOO
over-represents, without any access to the future. Usage: python eval_rolling.py <dataset>_t
"""
import glob
import json
import sys

import numpy as np
import torch

from common import CONFIGS, PCTM, RESULTS, Data, metrics, save_json, target_ranks
from model import ARCHS, right_aligned

ds = sys.argv[1]
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
d, cfg = Data(ds), CONFIGS[ds]
amazon = not ds.startswith("ml")
in_inner = np.zeros(d.n_items, bool)
in_inner[np.concatenate([s for s in d.inner if len(s)])] = True
hist_u, hist_k, ys, gaps = [], [], [], []
for u, (s, t) in enumerate(zip(d.seqs, d.times)):
    for k in range(max(1, len(d.inner[u])), len(s)):
        if in_inner[s[k]] and not np.any(s[:k] == s[k]):
            hist_u.append(u), hist_k.append(k), ys.append(s[k]), gaps.append(t[k] - t[k - 1])
ys, gaps = np.asarray(ys, dtype=np.int64), np.asarray(gaps)
if amazon:
    strata = {"same day": gaps == 0, "1-7 d": (gaps > 0) & (gaps <= 7 * 86400),
              "8-30 d": (gaps > 7 * 86400) & (gaps <= 30 * 86400), ">30 d": gaps > 30 * 86400}
else:
    strata = {"same second": gaps == 0, "<1 h": (gaps > 0) & (gaps <= 3600),
              "1 h-1 d": (gaps > 3600) & (gaps <= 86400), ">1 d": gaps > 86400}
print(f"{ds}: {len(ys)} rolling targets; " + ", ".join(f"{k} {m.mean():.1%}" for k, m in strata.items()), flush=True)

pctm = PCTM(d.inner, d.n_items, cfg["pctm"], cfg["max_len"], dev)
mc = PCTM(d.inner, d.n_items, dict(cfg["pctm"], counts="exp:1:1"), 1, dev)
sha = torch.from_numpy(d.sha_rank).to(dev)
ext = torch.arange(d.n_items, device=dev)
models = {}
for f in sorted(glob.glob(f"{RESULTS}/{ds}/*_s[0-9].pt")):
    tag, seed = f.rsplit("/", 1)[1][:-3].rsplit("_s", 1)
    if tag not in ("sasrec", "gru"):
        continue
    m = ARCHS[tag](d.n_items, cfg["max_len"], **cfg["sas"]).to(dev)
    m.load_state_dict(torch.load(f, map_location=dev))
    beta = json.load(open(f"{RESULTS}/{ds}/{tag}_s{seed}.json"))["val"]["beta"]
    models[f"{tag}_s{seed}"] = (m.eval(), beta)

ranks = {k: [] for k in ["pctm", "mc", "mostpop"] + [n for n in models] + [f"late_{n}" for n in models]}
with torch.no_grad():
    for s in range(0, len(ys), 512):
        hs = [d.seqs[u][:k] for u, k in zip(hist_u[s:s + 512], hist_k[s:s + 512])]
        y = torch.from_numpy(ys[s:s + 512]).to(dev)
        ex = d.exclude(hs, dev)
        S = pctm.score_histories(hs, dtype=torch.float32)
        ranks["pctm"].append(target_ranks(S.masked_fill(ex, -float("inf")), y, sha))
        mcs = mc.C[torch.tensor([h[-1] for h in hs], device=dev)].float()
        ranks["mc"].append(target_ranks(mcs.masked_fill(ex, -float("inf")), y, ext))
        pop = torch.log1p(pctm.freq).float().expand(len(hs), -1)
        ranks["mostpop"].append(target_ranks(pop.masked_fill(ex, -float("inf")), y, sha))
        X = right_aligned(hs, cfg["max_len"])
        for n, (m, beta) in models.items():
            z = m.scores(m.encode(X)[:, -1])
            ranks[n].append(target_ranks(z.masked_fill(ex, -float("inf")), y, sha))
            ranks[f"late_{n}"].append(target_ranks((z + beta * S).masked_fill(ex, -float("inf")), y, sha))
ranks = {k: torch.cat(v).cpu().numpy() for k, v in ranks.items()}
nd = lambda r: np.where(r < 10, 1 / np.log2(r + 2.0), 0.0)
rng = np.random.default_rng(0)


def boot(a, b, n=2000):
    dl = a - b
    m = dl[rng.integers(0, len(dl), (n, len(dl)))].mean(1)
    return [float(dl.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


res = {"dataset": ds, "n_targets": int(len(ys)), "strata": {}}
for arch in ("sasrec", "gru"):
    seeds = [n for n in models if n.startswith(arch + "_")]
    if not seeds:
        continue
    neural = np.mean([nd(ranks[n]) for n in seeds], 0)
    late = np.mean([nd(ranks[f"late_{n}"]) for n in seeds], 0)
    p = nd(ranks["pctm"])
    for name, m in [("all", np.ones(len(ys), bool))] + list(strata.items()):
        row = res["strata"].setdefault(name, {"share": float(m.mean()), "n": int(m.sum()),
                                              "pctm": float(p[m].mean()), "mc": float(nd(ranks["mc"])[m].mean()),
                                              "mostpop": float(nd(ranks["mostpop"])[m].mean())})
        row[arch] = float(neural[m].mean())
        row[f"late_{arch}"] = float(late[m].mean())
        row[f"{arch}_minus_pctm"] = boot(neural[m], p[m])
    res[f"{arch}_seeds"] = seeds
print(json.dumps(res["strata"], indent=None), flush=True)
save_json(f"{RESULTS}/{ds}/rolling.json", res)
np.savez(f"{RESULTS}/{ds}/rolling_ranks.npz", gaps=gaps, **ranks)
