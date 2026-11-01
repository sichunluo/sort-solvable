"""Numbers of the BERT4Rec control (train.py --arch bert, run_bert.sh), seeds 0-4, next to SASRec's: LOO
conventional and refit, fixed targets with random ties, GTS with original and random training ties, each relative
to the training-free baseline that make_tables.py selected on validation for that protocol and dataset; the
fixed-target order control of Table "order" from the cross-evaluation files; the masking chosen on validation.
Writes results/bert_summary.json and, given a tables directory, the paper's Table "bert" (bert.tex).
Usage: python bert_summary.py <results> <processed> [<tables dir>]
"""
import glob
import json
import os
import sys

import numpy as np

R, PROC = sys.argv[1:3]
OUT = sys.argv[3] if len(sys.argv) > 3 else None
DS = ["beauty", "sports", "toys", "ml1m"]
SEEDS = (0, 1, 2, 3, 4)
rng = np.random.default_rng(0)
seed_of = lambda name: int(name.rsplit("_s", 1)[1].split("_")[0].split(".")[0])
nd = lambda r: np.where(r < 10, 1 / np.log2(r + 2.0), 0.0)
rnd = lambda x, k=3: round(float(x), k)


def boot(xs, ys, n=2000):
    """Ratio of means with its two-level 95% interval (seeds, then users), as in make_tables.py."""
    X, Y = (np.atleast_2d(np.asarray(v, dtype=np.float64)) for v in (xs, ys))
    ratio = np.empty(n)
    for i in range(n):
        a = X[rng.integers(0, len(X), len(X))].mean(0)
        b = Y[rng.integers(0, len(Y), len(Y))].mean(0)
        u = rng.integers(0, X.shape[1], X.shape[1])
        ratio[i] = a[u].mean() / b[u].mean()
    return [rnd(X.mean() / Y.mean(), 4), *(rnd(v, 4) for v in np.percentile(ratio, [2.5, 97.5]))]


def runs(ds, tag):
    out = [nd(np.load(f)["alone"]) for f in sorted(glob.glob(f"{R}/{ds}/{tag}_s[0-9]_ranks.npz"))
           if seed_of(f) in SEEDS]
    return out if len(out) == len(SEEDS) else None


def cross(target, train, hist, model):
    f = f"{R}/{target}/cross_{train}__{hist}.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f)
    out = [nd(z[k]) for k in sorted(z.files) if k.startswith(f"{model}_s") and seed_of(k) in SEEDS]
    return out if len(out) == len(SEEDS) else None


NAMES = {"pctm": "\\pctm", "pctm_sym": "\\pctm-sym", "mc": "MC", "seqrules": "SeqRules", "mostpop": "MostPop"}


def experts_of(ds, suf="", kind="run"):
    """Baselines of a protocol, per user: from the run directory, or from the cross-evaluation file."""
    if kind == "cross":
        z = np.load(f"{R}/{ds}{suf}/cross_{ds}{suf}__{ds}{suf}.npz")
        return {NAMES[k]: nd(z[k]) for k in NAMES}
    ds, out = ds + suf, {}
    z = np.load(f"{R}/{ds}/experts_ranks.npz")
    out["\\pctm"], out["MC"] = nd(z["pctm_test"]), nd(z["mc_test"])
    z = np.load(f"{R}/{ds}/baselines_ranks.npz")
    names = {"pctm_sym": "\\pctm-sym", "seqrules": "SeqRules", "recentpop": "RecentPop", "mostpop": "MostPop"}
    out.update({names[k]: nd(z[k]) for k in z.files if k in names})
    return out


def tie(ds):
    g = np.load(f"{PROC}/{ds}_meta.npz")["ho_gap"]
    return g == 0 if ds.startswith("ml1m") else g < 86400


strongest = json.load(open(f"{R}/summary_strongest.json"))
# protocol of the strongest-baseline table -> (result directory suffix, how BERT4Rec and SASRec are read)
PROTOCOLS = {"LOO, conventional": ("", ("run", "bert"), ("run", "sasrec")),
             "LOO, refit": ("", ("run", "bert_refit"), ("run", "sasrec_refit")),
             "LOO, same targets, random ties (refit)": ("_fr", ("cross", "bert"), ("cross", "sasrec")),
             "GTS": ("_t", ("run", "bert"), ("run", "sasrec")),
             "GTS, training ties random": ("_tsx", ("run", "bert"), ("run", "sasrec"))}
CONDS = [("rnd/rnd", "_fr", "_fr"), ("rnd/ID", "_fr", ""), ("ID/rnd", "", "_fr"), ("rev/rev", "_fv", "_fv")]


def model_runs(ds, suf, how):
    kind, tag = how
    return runs(ds + suf, tag) if kind == "run" else cross(ds + suf, ds + suf, ds + suf, tag)


out = {"config": json.load(open(f"{R}/bert_config.json"))} if os.path.exists(f"{R}/bert_config.json") else {}
for ds in DS:
    o = out[ds] = {}
    for label, (suf, hb, hs) in PROTOCOLS.items():
        b, s = model_runs(ds, suf, hb), model_runs(ds, suf, hs)
        if b is None:
            continue
        best = strongest[label][ds]["best"]
        base = experts_of(ds, suf, hb[0])[best]
        o[label] = dict(bert=rnd(np.mean([x.mean() for x in b]), 4), sasrec=rnd(np.mean([x.mean() for x in s]), 4),
                        baseline=best, bert_vs_baseline=boot(b, [base]), bert_vs_sasrec=boot(b, s),
                        sasrec_vs_baseline=strongest[label][ds]["ratio"])
    t, p = tie(ds), experts_of(ds)["\\pctm"]
    for m in ("bert", "sasrec"):
        conv, refit = runs(ds, m), runs(ds, f"{m}_refit")
        if conv and refit:
            o[f"refit_gain/{m}"] = boot(refit, conv)
            o[f"refit/pctm_tied_other/{m}"] = [rnd(np.mean([x[t].mean() for x in refit]) / p[t].mean()),
                                                rnd(np.mean([x[~t].mean() for x in refit]) / p[~t].mean())]
            o[f"conv/pctm_tied_other/{m}"] = [rnd(np.mean([x[t].mean() for x in conv]) / p[t].mean()),
                                               rnd(np.mean([x[~t].mean() for x in conv]) / p[~t].mean())]
    ref = {m: cross(ds, ds, ds, m) for m in ("bert", "sasrec")}
    for cname, tr, hi in CONDS:
        for m in ("bert", "sasrec"):
            x = cross(ds + hi, ds + tr, ds + hi, m)
            if x is None or ref[m] is None:
                continue
            y = ref[m]
            o.setdefault("order", {})[f"{m}:{cname}"] = dict(
                tied_change=rnd(np.mean([v[t].mean() for v in x]) / np.mean([v[t].mean() for v in y]) - 1),
                other_change=rnd(np.mean([v[~t].mean() for v in x]) / np.mean([v[~t].mean() for v in y]) - 1),
                all_ratio=boot(x, y))
    for m in ("bert", "sasrec"):
        a, b = runs(ds + "_tsx", m), runs(ds + "_t", m)
        if a and b:
            o[f"gts_random_training_ties/{m}"] = boot(a, b)
json.dump(out, open(f"{R}/bert_summary.json", "w"), indent=1)

if OUT:
    NAMES = {"beauty": "Beauty", "sports": "Sports", "toys": "Toys", "ml1m": "ML-1M"}
    sig = lambda b: "$^{*}$" if b[1] > 1 or b[2] < 1 else ""
    pm = lambda x: f"{x:+.0f}".replace("-", "$-$")
    rows = [("NDCG@10, conventional", lambda o: f"{o['LOO, conventional']['bert']:.4f}"),
            ("NDCG@10, refit", lambda o: f"{o['LOO, refit']['bert']:.4f}"),
            ("Refit gain (\\%)", lambda o: f"{pm(100 * (o['refit_gain/bert'][0] - 1))}{sig(o['refit_gain/bert'])} "
                                           f"({pm(100 * (o['refit_gain/sasrec'][0] - 1))})"),
            ("Refit/\\pctm, tied", lambda o: f"{o['refit/pctm_tied_other/bert'][0]:.2f} ({o['refit/pctm_tied_other/sasrec'][0]:.2f})"),
            ("Refit/\\pctm, other", lambda o: f"{o['refit/pctm_tied_other/bert'][1]:.2f} ({o['refit/pctm_tied_other/sasrec'][1]:.2f})"),
            ("\\midrule \\multicolumn{5}{l}{Fixed targets: change on tied targets (\\%)} \\\\", None)]
    for cname, lab in (("rnd/rnd", "rnd/rnd"), ("rnd/ID", "rnd/ID (training)"), ("ID/rnd", "ID/rnd (histories)"),
                       ("rev/rev", "rev/rev")):
        rows.append((f"\\quad {lab}", lambda o, c=cname: f"{pm(100 * o['order'][f'bert:{c}']['tied_change'])} "
                                                         f"({pm(100 * o['order'][f'sasrec:{c}']['tied_change'])})"))
    rows += [("\\midrule GTS, random ties (\\%)", lambda o: f"{pm(100 * (o['gts_random_training_ties/bert'][0] - 1))}"
                                                           f"{sig(o['gts_random_training_ties/bert'])} "
                                                           f"({pm(100 * (o['gts_random_training_ties/sasrec'][0] - 1))})"),
             ("\\midrule \\multicolumn{5}{l}{BERT4Rec/SASRec} \\\\", None),
             ("\\quad LOO, refit", lambda o: f"{o['LOO, refit']['bert_vs_sasrec'][0]:.2f}{sig(o['LOO, refit']['bert_vs_sasrec'])}"),
             ("\\quad LOO, fixed targets, rnd.", lambda o: f"{o['LOO, same targets, random ties (refit)']['bert_vs_sasrec'][0]:.2f}"
                                                       f"{sig(o['LOO, same targets, random ties (refit)']['bert_vs_sasrec'])}"),
             ("\\quad GTS", lambda o: f"{o['GTS']['bert_vs_sasrec'][0]:.2f}{sig(o['GTS']['bert_vs_sasrec'])}"),
             ("\\quad GTS, random ties", lambda o: f"{o['GTS, training ties random']['bert_vs_sasrec'][0]:.2f}"
                                                f"{sig(o['GTS, training ties random']['bert_vs_sasrec'])}")]
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{BERT4Rec (bidirectional attention, cloze objective) under the access and order controls, seeds "
             "0--4; SASRec in parentheses (its NDCG@10 is in Table~\\ref{tab:factorial}). Refit/\\pctm on tied and other LOO "
             "targets. Fixed targets: train/test tie order as in Table~\\ref{tab:order}. GTS, random ties: training and "
             "history ties at random, targets fixed. $^{*}$: two-level 95\\% interval of the ratio excludes 1 (BERT4Rec).}",
             "\\label{tab:bert}", "\\setlength{\\tabcolsep}{2pt}", "\\footnotesize",
             "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{lcccc}", "\\toprule", " & " + " & ".join(NAMES[d] for d in DS) + " \\\\", "\\midrule"]
    for lab, f in rows:
        lines.append(lab if f is None else lab + " & " + " & ".join(f(out[d]) for d in DS) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    open(f"{OUT}/bert.tex", "w").write("\n".join(lines) + "\n")
print(json.dumps({ds: {k: v for k, v in out[ds].items() if k != "order"} for ds in DS}, indent=1))
