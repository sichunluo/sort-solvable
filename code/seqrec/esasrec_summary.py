"""eSASRec (upstream code) over seeds 32-36 on the reported LOO splits: conventional and refit NDCG@10, their ratio
to PCTM / PCTM-sym, the refit gain, the gap to the published value, and the best training-free baseline."""
import json, os, sys
import numpy as np
R = sys.argv[1]
SEEDS = (32, 33, 34, 35, 36)
PUB = {"beauty": 0.0524, "sports": 0.0324, "toys": 0.0533, "ml1m": 0.1739}  # probes paper, eSASRec column
nd = lambda r: np.where(r < 10, 1 / np.log2(r + 2.0), 0.0)
out = {}
for ds in ("beauty", "sports", "toys", "ml1m", "beauty_fr", "sports_fr", "toys_fr", "ml1m_fr"):
    d = f"{R}/{ds}"
    row = {}
    for tag in ("esasrec", "esasrec_refit"):
        v = [nd(np.load(f"{d}/{tag}_s{s}_ranks.npz")["alone"]).mean() for s in SEEDS if os.path.exists(f"{d}/{tag}_s{s}_ranks.npz")]
        row[tag] = (float(np.mean(v)), float(np.std(v)), len(v))
        row[tag + "_per_seed"] = [round(float(x), 4) for x in v]
    e = np.load(f"{d}/experts_ranks.npz"); b = np.load(f"{d}/baselines_ranks.npz")
    base = {"pctm": nd(e["pctm_test"]).mean(), **{k: nd(b[k]).mean() for k in b.files if k in ("seqrules", "pctm_sym", "mc", "mostpop")}}
    row["base"] = {k: round(float(v), 4) for k, v in base.items()}
    c, f = row["esasrec"][0], row["esasrec_refit"][0]
    row["conv/pctm"], row["refit/pctm"] = round(c / base["pctm"], 3), round(f / base["pctm"], 3)
    if "pctm_sym" in base:
        row["refit/pctm_sym"] = round(f / base["pctm_sym"], 3)
    row["refit_gain_%"] = round(100 * (f / c - 1), 1)
    row["refit/best_base"] = round(f / max(base.values()), 3)
    row["best_base"] = max(base, key=base.get)
    if ds in PUB:
        row["conv-published"] = round(c - PUB[ds], 4)
    out[ds] = row
    print(ds, json.dumps({k: v for k, v in row.items() if not k.endswith("per_seed")}))
json.dump(out, open(f"{R}/esasrec_summary.json", "w"), indent=1)
