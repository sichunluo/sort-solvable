"""Numbers quoted only in the text of the LOO sections (ties, data access, fusion), over the paper's seeds: seeds
0-4 for every trained model, 1-5 for the analyses that reload saved LOO checkpoints (gate, weight diagnostic,
stacking test, LTR, re-tuned PoE). Seed means are taken per user before ratios. Writes results/text_numbers_loo.json.
Usage: python text_numbers_loo.py <results> <processed>
"""
import glob
import json
import os
import sys

import numpy as np

R, PROC = sys.argv[1:3]
DS = ["beauty", "sports", "toys", "ml1m"]
SEEDS, RELOAD_SEEDS = (0, 1, 2, 3, 4), (1, 2, 3, 4, 5)
SASPLUS = {"beauty": 0.0537, "sports": 0.0315, "toys": 0.0575, "ml1m": 0.1662}  # published SAS+ (probe study)
seed_of = lambda f: int(os.path.basename(f).rsplit("_s", 1)[1].split("_")[0].split(".")[0])
nd = lambda r: np.where(r < 10, 1 / np.log2(r + 2.0), 0.0)
rnd = lambda x, k=3: round(float(x), k)


def runs(ds, tag, key="alone", seeds=SEEDS):
    out = {}
    for f in sorted(glob.glob(f"{R}/{ds}/{tag}_s[0-9]_ranks.npz")):
        z = np.load(f)
        if seed_of(f) in seeds and key in z.files:
            out[seed_of(f)] = nd(z[key])
    return out


def mean_users(d):
    return np.mean(list(d.values()), 0)


def tie(ds):
    g = np.load(f"{PROC}/{ds}_meta.npz")["ho_gap"]
    return g == 0 if ds.startswith("ml1m") else g < 86400


def expert(ds, key="pctm_test"):
    return nd(np.load(f"{R}/{ds}/experts_ranks.npz")[key])


def base(ds, key):
    return nd(np.load(f"{R}/{ds}/baselines_ranks.npz")[key])


out = {}
for ds in DS:
    t, p, ps = tie(ds), expert(ds), base(ds, "pctm_sym")
    s, sr = mean_users(runs(ds, "sasrec")), mean_users(runs(ds, "sasrec_refit"))
    late, lr = mean_users(runs(ds, "sasrec", "late")), mean_users(runs(ds, "sasrec_refit", "late"))
    g, gr = runs(ds, "gru"), runs(ds, "gru_refit")
    o = out[ds] = {}
    # ties: where the NDCG and the expert's lead sit
    o["ndcg_share_tied"] = dict(pctm=rnd(p[t].sum() / p.sum()), sasrec=rnd(s[t].sum() / s.sum()),
                                sasrec_refit=rnd(sr[t].sum() / sr.sum()))
    o["pctm_lead_share_tied_conv"] = rnd((p - s)[t].sum() / (p - s).sum())
    # data access
    o["sasrec_conv"], o["sasrec_refit"] = rnd(s.mean(), 4), rnd(sr.mean(), 4)
    o["sasrec_conv_minus_sasplus"] = rnd(s.mean() - SASPLUS[ds], 4)
    o["refit_gain_%"] = rnd(100 * (sr.mean() / s.mean() - 1), 1)
    o["gru_refit_gain_%"] = rnd(100 * (np.mean([x.mean() for x in gr.values()]) / np.mean([x.mean() for x in g.values()]) - 1), 1)
    o["gru_refit/pctm"] = rnd(np.mean([x.mean() for x in gr.values()]) / p.mean())
    o["refit/pctm_tied_other"] = [rnd(sr[t].mean() / p[t].mean()), rnd(sr[~t].mean() / p[~t].mean())]
    o["conv/pctm"] = rnd(s.mean() / p.mean())
    ea = json.load(open(f"{R}/{ds}/expert_access.json"))
    pin = ea["inner"]["pctm"]["ndcg@10"]
    o["inner_pctm_change_%"] = rnd(100 * (pin / ea["full"]["pctm"]["ndcg@10"] - 1), 1)
    o["conv/inner_pctm"] = rnd(s.mean() / pin)
    # fixed targets, random training ties and histories (_fr): conventional gap, refit tied/other
    pf, psf = expert(f"{ds}_fr"), base(f"{ds}_fr", "pctm_sym")
    sf, srf = mean_users(runs(f"{ds}_fr", "sasrec")), mean_users(runs(f"{ds}_fr", "sasrec_refit"))
    o["fr_conv/pctm"] = rnd(sf.mean() / pf.mean())
    o["fr_refit/pctm_tied"] = rnd(srf[t].mean() / pf[t].mean())
    o["fr_refit/pctm_sym_tied_other"] = [rnd(srf[t].mean() / psf[t].mean()), rnd(srf[~t].mean() / psf[~t].mean())]
    grf = mean_users(runs(f"{ds}_fr", "gru_refit"))
    o["fr_gru_refit_change_tied_%"] = rnd(100 * (grf[t].mean() / mean_users(gr)[t].mean() - 1), 1)
    # late fusion
    o["late_gain_conv_%"] = rnd(100 * (late.mean() / s.mean() - 1), 1)
    o["late_gain_share_tied"] = [rnd((late - s)[t].sum() / (late - s).sum()), rnd((lr - sr)[t].sum() / (lr - sr).sum())]
    o["late_refit/pctm_%"] = rnd(100 * (lr.mean() / p.mean() - 1), 1)
    # joint PoE vs late fusion (every seed against the seed-mean late fusion of the same protocol)
    L, LR = late.mean(), lr.mean()
    for tag, ref in (("poe_luo1", L), ("poe_luo0", L), ("poe_luo1_refit", LR), ("poe_luo1_bval", L), ("poe_luo1_fix", L)):
        f = runs(ds, tag, "fused")
        o[f"{tag}_vs_late_%"] = [rnd(100 * (x.mean() / ref - 1), 1) for x in f.values()]
        o[f"{tag}/pctm"] = rnd(np.mean([x.mean() for x in f.values()]) / p.mean())
    a1, a0 = runs(ds, "poe_luo1"), runs(ds, "poe_luo0")
    o["residual_luo_vs_full_%"] = rnd(100 * (np.mean([x.mean() for x in a1.values()]) / np.mean([x.mean() for x in a0.values()]) - 1), 1)
    js = {s_: json.load(open(f"{R}/{ds}/poe_luo1_s{s_}.json")) for s_ in SEEDS}
    o["poe_learned_beta"] = [rnd(j["log"][j["best_epoch"]]["beta"], 2) for j in js.values()]
    jb = [json.load(open(f"{R}/{ds}/poe_luo1_bval_s{s_}.json"))["best_epoch"] for s_ in SEEDS]
    o["bval_best_epoch"] = jb
    tuned = runs(ds, "poe_luo1", "tuned", RELOAD_SEEDS)
    o["tuned_vs_late_%"] = rnd(100 * (np.mean([x.mean() for x in tuned.values()]) / L - 1), 1)
    o["late_beta_val"] = sorted({json.load(open(f"{R}/{ds}/sasrec_s{s_}.json"))["val"]["beta"] for s_ in SEEDS})
    # analyses that reload saved LOO checkpoints
    dw = [json.load(open(f"{R}/{ds}/diag_weights_s{s_}.json"))["std_ratio_catalogue_top100"] for s_ in RELOAD_SEEDS]
    o["std_ratio_top100"] = {k: rnd(np.mean([d[k]["ratio_full_top100"][1] for d in dw]), 2) for k in ("late", "poe")}
    st = json.load(open(f"{R}/{ds}/stacking_test.json"))
    o["stacking_beta_star"] = {k: sorted({st[str(s_)][k]["beta_star"] for s_ in RELOAD_SEEDS}) for k in ("luo", "full")}
    o["stacking_val_beta"] = sorted({st[str(s_)]["val_beta_star"] for s_ in RELOAD_SEEDS})
    ga = [json.load(open(f"{R}/{ds}/gate_s{s_}.json")) for s_ in RELOAD_SEEDS]
    o["gate_minus_late"] = rnd(np.mean([x["test"]["gate"]["ndcg@10"] - x["test"]["late"]["ndcg@10"] for x in ga]), 4)
    lt = [json.load(open(f"{R}/{ds}/ltr_s{s_}.json")) for s_ in RELOAD_SEEDS]
    d_ltr = np.mean([x["test_ndcg10"]["ltr"] - x["test_ndcg10"]["late"] for x in lt])
    d_orc = np.mean([x["test_ndcg10"]["oracle_selector"] - x["test_ndcg10"]["late"] for x in lt])
    o["ltr_gain_%"] = rnd(100 * np.mean([x["test_ndcg10"]["ltr"] / x["test_ndcg10"]["late"] - 1 for x in lt]), 1)
    o["ltr_share_of_oracle_margin"] = rnd(d_ltr / d_orc, 2)
    o["oracle_selector_vs_late_%"] = rnd(100 * np.mean([x["test_ndcg10"]["oracle_selector"] / x["test_ndcg10"]["late"] - 1 for x in lt]), 1)
    e = json.load(open(f"{R}/{ds}/ensemble.json"))
    o["oracle_vs_late_%"] = rnd(100 * (e["oracle_sasrec_vs_pctm"] / L - 1), 1)
    o["oracle_seed_vs_late_%"] = rnd(100 * (e["oracle_seed_vs_seed"] / L - 1), 1)
    o["ens_all_vs_sasrec_%"] = rnd(100 * (e["ens_all"]["alone"] / s.mean() - 1), 1)
    o["ens_all_vs_late_%"] = rnd(100 * (e["ens_all"]["alone"] / L - 1), 1)
    o["jaccard"] = [rnd(e["jaccard_sasrec_vs_pctm"], 2), rnd(e["jaccard_seed_vs_seed"], 2)]
    print(ds, json.dumps(o), flush=True)
json.dump(out, open(f"{R}/text_numbers_loo.json", "w"), indent=1)
