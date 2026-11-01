"""Build the paper's LaTeX tables from data/seqrec/results. Run locally:
python -I code/seqrec/make_tables.py data/seqrec/results paper_icde/tables"""
import glob
import json
import os
import sys

import numpy as np

R, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
DS = ["beauty", "sports", "toys", "ml1m"]
NAMES = {"beauty": "Beauty", "sports": "Sports", "toys": "Toys", "ml1m": "ML-1M"}
rng = np.random.default_rng(0)
# Every neural condition is averaged over five seeds: 0-4, or 1-5 for the analyses that reload saved LOO
# checkpoints (LTR, re-tuned PoE, seed ensemble), whose seed-0 checkpoints were not kept.
SEEDS, RELOAD_SEEDS = (0, 1, 2, 3, 4), (1, 2, 3, 4, 5)
SPAN, RSPAN = f"{SEEDS[0]}--{SEEDS[-1]}", f"{RELOAD_SEEDS[0]}--{RELOAD_SEEDS[-1]}"
ESEEDS = (32, 33, 34, 35, 36)  # seeds of the eSASRec runs (run_esasrec_queue.sh)
ESPAN = f"{ESEEDS[0]}--{ESEEDS[-1]}"
NSEEDS = {3: "three", 5: "five"}[len(SEEDS)]
seed_of = lambda name: int(name.rsplit("_s", 1)[1].split("_")[0])


def nd(r):
    return np.where(r < 10, 1 / np.log2(r + 2.0), 0.0)


def runs(ds, tag, key, seeds=SEEDS):
    out = []
    for f in sorted(glob.glob(f"{R}/{ds}/{tag}_s[0-9]_ranks.npz")):
        if seed_of(f) not in seeds:
            continue
        z = np.load(f)
        if key in z.files:
            out.append(nd(z[key]))
    return out


def expert(ds, key="pctm_test"):
    return [nd(np.load(f"{R}/{ds}/experts_ranks.npz")[key])]


def boot(xs, ys, n=2000, gen=None):
    """Two-level paired bootstrap: resample the seeds of each method, then the test users (95% intervals).

    xs, ys: per-user NDCG arrays, one per seed (one array for a deterministic method). Returns the difference of
    means and the ratio of means with their intervals, so that seed variance is part of every interval. gen: a
    separate generator, so that intervals added later leave the stream of the existing ones unchanged."""
    g = rng if gen is None else gen
    X, Y = (np.atleast_2d(np.asarray(v, dtype=np.float64)) for v in (xs, ys))
    diff, ratio = np.empty(n), np.empty(n)
    for i in range(n):
        a = X[g.integers(0, len(X), len(X))].mean(0)
        b = Y[g.integers(0, len(Y), len(Y))].mean(0)
        u = g.integers(0, X.shape[1], X.shape[1])
        diff[i], ratio[i] = a[u].mean() - b[u].mean(), a[u].mean() / b[u].mean()
    m, r = X.mean() - Y.mean(), X.mean() / Y.mean()
    return dict(diff=(m, *np.percentile(diff, [2.5, 97.5])), ratio=(r, *np.percentile(ratio, [2.5, 97.5])))


def ci(xs, ys, n=2000):
    """Difference of means with its two-level 95% interval."""
    return boot(xs, ys, n)["diff"]


def cell(xs, ref=None, bold=False):
    """mean (± std over seeds); dagger/asterisk when the paired CI vs ref excludes 0 (worse/better)."""
    if not xs:
        return "--"
    per_seed = [x.mean() for x in xs]
    s = f"{np.mean(per_seed):.4f}"
    if len(xs) > 1:
        s += f"\\,$\\pm${np.std(per_seed):.4f}"
    if ref is not None:
        _, lo, hi = ci(xs, ref)
        s += "$^{\\uparrow}$" if lo > 0 else ("$^{\\downarrow}$" if hi < 0 else "")
    return f"\\textbf{{{s}}}" if bold else s


def ltr_ranks(ds):
    return [nd(np.load(f)["ltr"]) for f in sorted(glob.glob(f"{R}/{ds}/ltr_s[0-9]_ranks.npz"))
            if seed_of(f) in RELOAD_SEEDS]


def oracle(ds):
    p = expert(ds)[0]
    return [np.maximum(s, p) for s in runs(ds, "sasrec", "alone")]


# ---- Table: combining SASRec and PCTM under LOO ----
late = lambda ds: runs(ds, "sasrec", "late")
late_r = lambda ds: runs(ds, "sasrec_refit", "late")
rows = [("\\pctm", lambda ds: expert(ds), None, False),
        ("SASRec", lambda ds: runs(ds, "sasrec", "alone"), None, False),
        ("Late fusion", late, None, True),
        ("Joint PoE (leave-user-out)", lambda ds: runs(ds, "poe_luo1", "fused"), late, False),
        ("\\quad $\\beta$ re-fit on val.\\ each epoch", lambda ds: runs(ds, "poe_luo1_bval", "fused"), late, False),
        ("\\quad $\\beta$ fixed to late-fusion $\\beta^*$", lambda ds: runs(ds, "poe_luo1_fix", "fused"), late, False),
        ("\\quad $\\beta$ re-tuned after training$^\\dagger$", lambda ds: runs(ds, "poe_luo1", "tuned", RELOAD_SEEDS), late, False),
        ("LTR re-ranking (stacked)", ltr_ranks, late, False),
        ("Refit: SASRec", lambda ds: runs(ds, "sasrec_refit", "alone"), None, False),
        ("\\quad late fusion", late_r, None, True),
        ("\\quad joint PoE (leave-user-out)", lambda ds: runs(ds, "poe_luo1_refit", "fused"), late_r, False)]


def short(xs, ref=None, bold=False):
    """mean over seeds with an arrow when the two-level CI vs ref excludes 0."""
    if not xs:
        return "--"
    s = f"{np.mean([x.mean() for x in xs]):.4f}"
    if ref:
        _, lo, hi = ci(xs, ref)
        s += "$^{\\uparrow}$" if lo > 0 else ("$^{\\downarrow}$" if hi < 0 else "")
    return f"\\textbf{{{s}}}" if bold else s


lines = ["\\begin{table}[!ht]", "\\centering",
         f"\\caption{{Combining SASRec and \\pctm under LOO (test NDCG@10, mean over seeds {SPAN}; LTR and "
         f"$^\\dagger$: seeds {RSPAN}, whose checkpoints were kept)."
         " $\\uparrow$/$\\downarrow$: two-level 95\\% interval vs.\\ the late fusion of the same "
         "block excludes zero.}",
         "\\label{tab:fusion}", "\\setlength{\\tabcolsep}{2.5pt}", "\\footnotesize",
         "\\begin{tabular}{l" + "c" * len(DS) + "}", "\\toprule",
         " & " + " & ".join(NAMES[d] for d in DS) + " \\\\", "\\midrule"]
for i, (label, fn, ref_fn, bold) in enumerate(rows):
    if label.startswith(("Joint PoE", "LTR", "Refit:")):
        lines.append("\\midrule")
    cells = [short(fn(ds), ref_fn(ds) if ref_fn is not None else None, bold) for ds in DS]
    lines.append(label + " & " + " & ".join(cells) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(f"{OUT}/fusion.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))


# ---- Table: timestamp ties in LOO ----
PROC = os.environ.get("SEQREC_PROCESSED", "data/seqrec/processed")


def mean_seeds(xs):
    return np.mean(xs, 0) if xs else None


def fmt(x, std=None):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.4f}" + (f"\\,$\\pm${std:.4f}" if std is not None else "")


def seeds_cell(xs):
    if not xs:
        return "--"
    m = [x.mean() for x in xs]
    return fmt(np.mean(m), np.std(m) if len(m) > 1 else None)


def baseline(ds, key):
    f = f"{R}/{ds}/baselines.json"
    return json.load(open(f))[key]["ndcg@10"] if os.path.exists(f) else None


tie_rows = []
for ds in DS:
    gap = np.load(f"{PROC}/{ds}_meta.npz")["ho_gap"]
    tie = gap == 0 if ds == "ml1m" else gap < 86400
    p, s = expert(ds)[0], mean_seeds(runs(ds, "sasrec", "alone"))
    ps = expert(ds + "_shuf")[0] if os.path.exists(f"{R}/{ds}_shuf/experts_ranks.npz") else None
    ss = runs(ds + "_shuf", "sasrec", "alone")
    tie_rows.append([
        f"{100 * tie.mean():.1f}", f"{100 * p[tie].sum() / p.sum():.1f}",
        f"{p[tie].mean():.4f}", f"{s[tie].mean():.4f}", f"{s[tie].mean() / p[tie].mean():.2f}",
        f"{p[~tie].mean():.4f}", f"{s[~tie].mean():.4f}", f"{s[~tie].mean() / p[~tie].mean():.2f}",
        fmt(baseline(ds, "pctm_sym")),
        fmt(ps.mean() if ps is not None else None), seeds_cell(ss), seeds_cell(runs(ds + "_shuf", "sasrec", "late")),
        f"{np.mean([x.mean() for x in ss]) / ps.mean():.2f}" if ss and ps is not None else "--",
        fmt(baseline(ds + "_shuf", "pctm_sym"))])
labels = ["Targets tied with previous event (\\%)", "Share of \\pctm NDCG from tied targets (\\%)",
          "Tied targets: \\pctm", "Tied targets: SASRec", "\\quad SASRec / \\pctm", "Other targets: \\pctm",
          "Other targets: SASRec", "\\quad SASRec / \\pctm", "Direction-free \\pctm ($C+C^\\top$)",
          "Ties shuffled: \\pctm", "\\quad SASRec (no refit)", "\\quad late fusion (no refit)", "\\quad SASRec (no refit) / \\pctm",
          "\\quad direction-free \\pctm"]
lines = ["\\begin{table*}[t]", "\\centering",
         "\\caption{Timestamp ties in the LOO benchmarks. Tied: same day as the previous history event (Amazon) or "
         "same second (ML-1M). Shuffled: ties randomized before splitting, so labels can change. SASRec is conventional "
         f"(no refit), with means $\\pm$ seed standard deviations over {NSEEDS} seeds. NDCG@10.}}",
         "\\label{tab:ties}", "\\setlength{\\tabcolsep}{6pt}", "\\small",
         "\\begin{tabular}{l" + "c" * len(DS) + "}", "\\toprule",
         " & " + " & ".join(NAMES[d] for d in DS) + " \\\\", "\\midrule"]
for i, lab in enumerate(labels):
    if i in (2, 8, 9):
        lines.append("\\midrule")
    lines.append(lab + " & " + " & ".join(r[i] for r in tie_rows) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
open(f"{OUT}/ties.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines[9:-3]))


# ---- Table: global temporal split, controlled leakage ----
def tuned(ds):
    f = f"{R}/{ds}/pctm_tuned.json"
    return json.load(open(f))["test"]["ndcg@10"] if os.path.exists(f) else None


def folds(ds, tag, key):
    """Per-seed NDCG over the union of the two leaky folds (= the GTS test targets)."""
    a = {seed_of(f): nd(np.load(f)[key]) for f in glob.glob(f"{R}/{ds}_tl0/{tag}_s[0-9]_ranks.npz") if seed_of(f) in SEEDS}
    b = {seed_of(f): nd(np.load(f)[key]) for f in glob.glob(f"{R}/{ds}_tl1/{tag}_s[0-9]_ranks.npz") if seed_of(f) in SEEDS}
    return [np.concatenate([a[k], b[k]]) for k in sorted(set(a) & set(b))]


def folds_expert(ds):
    try:
        return np.concatenate([nd(np.load(f"{R}/{ds}_tl{k}/experts_ranks.npz")["pctm_test"]) for k in (0, 1)])
    except FileNotFoundError:
        return None


def cross_sasrec(target, train, hist):
    """Per-user SASRec NDCG of every seed from an eval_cross.py file."""
    z = np.load(f"{R}/{target}/cross_{train}__{hist}.npz")
    return [nd(z[f"sasrec_s{s}"]) for s in SEEDS]


def pct_ci(b):
    """Ratio with interval as percent changes, the interval below the point estimate."""
    return (f"\\shortstack{{{100 * (b[0] - 1):+.1f}\\%\\\\{{\\scriptsize[{100 * (b[1] - 1):+.1f}, "
            f"{100 * (b[2] - 1):+.1f}]}}}}")


# (row, target, training data, test histories): the targets of _t and _tsx are the same rows
XCONDS = [("Random training ties only", "{}_t", "{}_tsx", "{}_t"),
          ("Random history ties only", "{}_tsx", "{}_t", "{}_tsx"),
          ("Random training and history ties", "{}_tsx", "{}_tsx", "{}_tsx")]
xrng = np.random.default_rng(0)
tt, xt = [], []
for ds in DS:
    t = ds + "_t"
    p = expert(t)[0]
    s = runs(t, "sasrec", "alone")
    d, lo, hi = ci(s, [p]) if s else (np.nan,) * 3
    pl, sl = folds_expert(ds), folds(ds, "sasrec", "alone")
    tt.append([fmt(expert(t, "mc_test")[0].mean()), fmt(baseline(t, "mostpop")), fmt(baseline(t, "recentpop")),
               fmt(p.mean()), fmt(baseline(t, "pctm_sym")), fmt(baseline(t, "seqrules")), fmt(tuned(t)), seeds_cell(s), seeds_cell(runs(t, "sasrec_norefit", "alone")),
               seeds_cell(runs(t, "gru", "alone")), seeds_cell(runs(t, "sasrec", "late")),
               f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]".replace("[", "{\\scriptsize[").replace("]", "]}"),
               fmt(pl.mean() if pl is not None else None) + (f" ({100 * (pl.mean() / p.mean() - 1):+.0f}\\%)" if pl is not None else ""),
               # relative change vs. the same seeds (leaky runs use seeds 0..k-1; runs() is sorted by seed)
               seeds_cell(sl) + (f" ({100 * (np.mean([x.mean() for x in sl]) / np.mean([x.mean() for x in s[:len(sl)]]) - 1):+.0f}\\%)" if sl and s else ""),
               str(len(p))])
    # fixed-target order crossing on the GTS checkpoints (eval_cross.py), from its own generator
    ref = cross_sasrec(t, t, t)
    xt.append([pct_ci(boot(cross_sasrec(*(c.format(ds) for c in spec)), ref, gen=xrng)["ratio"]) for _, *spec in XCONDS])
labels = ["MC", "MostPop", "RecentPop (last 30 days)", "\\pctm (LOO configuration)", "\\pctm-sym", "SeqRules",
          "\\pctm (re-tuned on GTS val)",
          "SASRec (refit on all pre-cutoff data)", "SASRec (no refit)", "GRU4Rec (refit)", "Late fusion (SASRec)",
          "SASRec $-$ \\pctm (LOO config.), 95\\% CI", "Leaky GTS: \\pctm", "Leaky GTS: SASRec", "Test targets"]
lines = ["\\begin{table*}[t]", "\\centering",
         "\\caption{Global temporal split (GTS): test NDCG@10 (mean\\,$\\pm$\\,std over " + NSEEDS + " seeds). Leaky GTS evaluates the same targets after adding other users' "
         "post-cutoff events to training (relative change vs.\\ GTS with the same seeds). Crossover cells report SASRec "
         "percent changes with pointwise two-level 95\\% intervals in brackets; targets and outer training events are "
         "fixed.}",
         "\\label{tab:gts}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{l" + "c" * len(DS) + "}", "\\toprule",
         " & " + " & ".join(NAMES[d] for d in DS) + " \\\\", "\\midrule"]
for i, lab in enumerate(labels):
    if i in (3, 7, 11, 12, 14):
        lines.append("\\midrule")
    if i == 14:  # the crossing, before the target counts
        lines.append(f"\\multicolumn{{{1 + len(DS)}}}{{l}}{{\\textbf{{Fixed-target SASRec order control: percent change "
                     "and 95\\% interval}} \\\\")
        lines += [lab_ + " & " + " & ".join(r[k] for r in xt) + " \\\\" for k, (lab_, *_) in enumerate(XCONDS)]
        lines.append("\\midrule")
    lines.append(lab + " & " + " & ".join(r[i] for r in tt) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
open(f"{OUT}/gts.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines[8:-3]))


# ---- Table: 2x2 factorial (tie order x refit) under LOO, with tied/other split ----
def ratio_ci(s_list, p):
    """SASRec/expert ratio of means and paired 95% CI of the difference (seed-averaged per user)."""
    if not s_list or p is None:
        return "--"
    s = mean_seeds(s_list)
    _, lo, hi = ci(s_list, [p])
    star = "$^{*}$" if lo > 0 or hi < 0 else ""
    return f"{s.mean() / p.mean():.2f}{star}"


def tie_mask(ds):
    gap = np.load(f"{PROC}/{ds}_meta.npz")["ho_gap"]
    return gap == 0 if ds.startswith("ml1m") else gap < 86400


fact = []
for ds in DS:
    p = expert(ds)[0]
    tie = tie_mask(ds)
    nr, rf = runs(ds, "sasrec", "alone"), runs(ds, "sasrec_refit", "alone")
    sh = ds + "_shuf"
    ps = expert(sh)[0] if os.path.exists(f"{R}/{sh}/experts_ranks.npz") else None
    snr, srf = runs(sh, "sasrec", "alone"), runs(sh, "sasrec_refit", "alone")
    split = lambda xs, m: [x[m] for x in xs]
    fact.append([fmt(p.mean()), seeds_cell(nr), seeds_cell(rf), ratio_ci(nr, p), ratio_ci(rf, p),
                 ratio_ci(split(rf, tie), p[tie]) if rf else "--", ratio_ci(split(rf, ~tie), p[~tie]) if rf else "--",
                 fmt(ps.mean() if ps is not None else None), seeds_cell(snr), seeds_cell(srf),
                 ratio_ci(snr, ps), ratio_ci(srf, ps)])
# the tied/other refit ratios (columns 5-6) are quoted in the text; computing them here keeps the bootstrap
# stream of the later tables unchanged
print("factorial tied/other refit ratios:", {ds: row[5:7] for ds, row in zip(DS, fact)})
# eSASRec (run_esasrec.py), conventional and refit, over its own seeds
esr = {ds: [np.mean([nd(np.load(f"{R}/{ds}/{tag}_s{s}_ranks.npz")["alone"]).mean() for s in ESEEDS])
            / expert(ds)[0].mean() for tag in ("esasrec", "esasrec_refit")] for ds in DS}
blocks = [("Original tie order", 0), ("Ties randomized before splitting", 7)]
fac_rows = ["\\pctm", "SASRec, conventional", "SASRec, refit", "Ratio, conventional", "Ratio, refit"]
lines = ["\\begin{table*}[t]", "\\centering",
         f"\\caption{{Data-access controls under original and shuffled-before-split LOO (NDCG@10; trained values are "
         f"means $\\pm$ standard deviations over seeds {SPAN}). The original and shuffled blocks have different tied "
         "labels; access comparisons within a block retain targets. Ratio rows give SASRec/\\pctm; $^{*}$ marks a "
         "two-level 95\\% interval excluding equality. The final block repeats the access test with the published "
         f"eSASRec implementation ({NSEEDS} seeds, {ESPAN}); ratios are point summaries.}}",
         "\\label{tab:factorial}", "\\small", "\\setlength{\\tabcolsep}{6pt}",
         "\\begin{tabular}{l" + "c" * len(DS) + "}", "\\toprule",
         "Model / access & " + " & ".join(NAMES[d] for d in DS) + " \\\\", "\\midrule"]
for bi, (title, off) in enumerate(blocks):
    if bi:
        lines.append("\\midrule")
    lines.append(f"\\multicolumn{{{1 + len(DS)}}}{{l}}{{\\textbf{{{title}}}}} \\\\")
    # cells off..off+4 of each row; the original block's tied/other cells (5-6) are not shown
    for c, lab in enumerate(fac_rows):
        lines.append(lab + " & " + " & ".join(row[off + c] for row in fact) + " \\\\")
lines += ["\\midrule", f"\\multicolumn{{{1 + len(DS)}}}{{l}}{{\\textbf{{Published eSASRec code, original order}}}} \\\\",
          "eSASRec/\\pctm, conventional & " + " & ".join(f"{esr[d][0]:.2f}" for d in DS) + " \\\\",
          "eSASRec/\\pctm, refit & " + " & ".join(f"{esr[d][1]:.2f}" for d in DS) + " \\\\"]
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
open(f"{OUT}/factorial.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines[9:-3]))


# ---- Table: SASRec vs the strongest training-free expert under each protocol ----
def experts_of(ds):
    out = {}
    if os.path.exists(f"{R}/{ds}/experts_ranks.npz"):
        z = np.load(f"{R}/{ds}/experts_ranks.npz")
        out["MC"], out["\\pctm"] = nd(z["mc_test"]), nd(z["pctm_test"])
    if os.path.exists(f"{R}/{ds}/baselines_ranks.npz"):
        z = np.load(f"{R}/{ds}/baselines_ranks.npz")
        names = {"pctm_sym": "\\pctm-sym", "seqrules": "SeqRules", "recentpop": "RecentPop", "mostpop": "MostPop"}
        out.update({names[k]: nd(z[k]) for k in z.files if k in names})
    return out


import hashlib


def val_scores(ds):
    """Validation NDCG@10 of the training-free baselines of a result directory (for validation-selected baselines)."""
    out = {}
    if os.path.exists(f"{R}/{ds}/experts.json"):
        e = json.load(open(f"{R}/{ds}/experts.json"))
        out["MC"], out["\\pctm"] = e["mc_val"]["ndcg@10"], e["pctm_val"]["ndcg@10"]
    if os.path.exists(f"{R}/{ds}/baselines.json"):
        b = json.load(open(f"{R}/{ds}/baselines.json"))
        names = {"pctm_sym": "\\pctm-sym", "seqrules": "SeqRules", "recentpop": "RecentPop", "mostpop": "MostPop"}
        out.update({names[k[:-4]]: v["ndcg@10"] for k, v in b.items() if k.endswith("_val") and k[:-4] in names})
    return out


def sha_ranks(catalog):
    """SHA-256 tie-break order of external ids (common.stable_hash_ranks without torch)."""
    dig = [hashlib.sha256(str(i).encode()).digest() for i in catalog]
    order = sorted(range(len(catalog)), key=lambda i: (dig[i], str(catalog[i])))
    r = np.empty(len(catalog), dtype=np.int64)
    r[np.asarray(order)] = np.arange(len(catalog))
    return r


def set_ndcg(ranks, users, k=10):
    """Per-user NDCG@k with every row of the user relevant (single-target rows reduce to plain NDCG)."""
    u, inv, cnt = np.unique(users, return_inverse=True, return_counts=True)
    dcg = np.bincount(inv, weights=nd(ranks))
    ideal = np.cumsum(1 / np.log2(np.arange(k) + 2.0))
    return dcg / ideal[np.minimum(cnt, k) - 1]


def cross(target, train, hist, per_user=False):
    """Methods -> per-user (or per-row) NDCG from eval_cross.py; SASRec as a list over seeds."""
    f = f"{R}/{target}/cross_{train}__{hist}.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f)
    m = (lambda r: set_ndcg(r, z["users"])) if per_user else nd
    out = {k: m(z[k]) for k in ("pctm", "pctm_sym", "mc", "seqrules", "mostpop")}
    out["sasrec"] = {k: m(z[k]) for k in z.files if k.startswith("sasrec_s") and seed_of(k) in SEEDS}
    out["late"] = {k[5:]: m(z[k]) for k in z.files if k.startswith("late_sasrec_s") and seed_of(k) in SEEDS}
    return out


def recentpop_set(ds):
    """RecentPop (30 days before the cutoff) on the GTS set targets, per user."""
    z, t = np.load(f"{PROC}/{ds}_t.npz"), np.load(f"{PROC}/{ds}_tset.npz")
    items, times, users = z["items"], z["times"], z["users"]
    n = len(z["catalog"])
    sc = np.log1p(np.bincount(items[times >= times.max() - 30 * 86400], minlength=n)) \
        + 1e-6 * np.log1p(np.bincount(items, minlength=n))
    pos = np.empty(n, dtype=np.int64)
    pos[np.lexsort((sha_ranks(z["catalog"]), -sc))] = np.arange(n)
    starts = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1]
    hist = dict(zip(users[starts], np.split(pos[items], starts[1:])))
    r = np.asarray([pos[y] - np.sum(hist[u] < pos[y]) for u, y in zip(t["ho_users"], t["ho_items"])])
    return set_ndcg(r, t["ho_users"])


BASE = {"pctm": "\\pctm", "pctm_sym": "\\pctm-sym", "mc": "MC", "seqrules": "SeqRules", "mostpop": "MostPop",
        "recentpop": "RecentPop"}


def protocol(ds, spec):
    """(baselines name -> per-user NDCG, SASRec list over seeds, validation directory) for one protocol."""
    kind = spec[0]
    if kind in ("run", "run_other"):
        _, suf, tag = spec
        if not os.path.exists(f"{R}/{ds}{suf}"):
            return {}, [], None
        ex, s = experts_of(ds + suf), runs(ds + suf, tag, "alone")
        if kind == "run_other":  # LOO without same-day targets
            keep = ~tie_mask(ds)
            ex, s = {k: v[keep] for k, v in ex.items()}, [x[keep] for x in s]
        return ex, s, ds + suf
    _, target, train, hist, per_user = spec
    c = cross(target.format(ds), train.format(ds), hist.format(ds), per_user)
    if c is None:
        return {}, [], None
    ex = {BASE[k]: c[k] for k in BASE if k in c}
    if target.endswith("_tset"):
        ex["RecentPop"] = recentpop_set(ds)
    return ex, [c["sasrec"][k] for k in sorted(c["sasrec"])], train.format(ds)


PROTOCOLS = [("LOO, conventional", ("run", "", "sasrec")),
             ("LOO, refit", ("run", "", "sasrec_refit")),
             ("LOO, ties shuffled (refit)", ("run", "_shuf", "sasrec_refit")),
             ("LOO, same targets, random ties (refit)", ("cross", "{}_fr", "{}_fr", "{}_fr", False)),
             ("LOO, day-level sets, random ties (refit)", ("cross", "{}_day", "{}_day", "{}_day", True)),
             ("GTS", ("run", "_t", "sasrec")),
             ("GTS, set targets", ("cross", "{}_tset", "{}_t", "{}_t", True)),
             ("GTS, training ties random", ("run", "_tsx", "sasrec")),
             ("GTS, training ties random, set targets", ("cross", "{}_tset", "{}_tsx", "{}_tsx", True)),
             ("LOO, refit, non-tied targets only", ("run_other", "", "sasrec_refit")),
             ("GTS, earlier cutoffs", ("run", "_tq", "sasrec")),
             ("GTS, earlier cutoffs, training ties random", ("run", "_tqx", "sasrec")),
             ("GTS, pre-cutoff 5-core (Beauty)", ("run", "_tp", "sasrec")),
             ("GTS, pre-cutoff 5-core, training ties random", ("run", "_tpx", "sasrec"))]
summary = {}
ABBR = {"\\pctm": "P", "\\pctm-sym": "S", "SeqRules": "R", "MC": "M", "MostPop": "Pop", "RecentPop": "RP"}
lines = ["\\begin{table*}[t]", "\\centering",
         "\\caption{SASRec relative to the strongest prespecified training-free baseline (NDCG@10; " + NSEEDS + " seeds). "
         "val.: baseline selected on validation, ratio and two-level 95\\% interval ($^{*}$: excludes 1). test: "
         "test-best baseline and ratio if different (= : same). Codes: MC (M), \\pctm (P), \\pctm-sym (S), SeqRules (R), "
         "MostPop and RecentPop (RP). All LOO rows except conventional use refit SASRec. Randomized GTS ties affect "
         "training and test histories. Transition configurations are frozen; GTS-retuned \\pctm is separate in "
         "Table~\\ref{tab:gts}. Non-tied targets use the full protocol's validation selection. Earlier cutoffs are the "
         "70th and 80th percentiles.}",
         "\\label{tab:strongest}", "\\setlength{\\tabcolsep}{2.2pt}", "\\scriptsize",
         "\\begin{tabular}{l" + "lc" * len(DS) + "}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{NAMES[d]}}}" for d in DS) + " \\\\",
         " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DS))),
         "Protocol & " + " & ".join("val. & test" for _ in DS) + " \\\\", "\\midrule"]
star = lambda b: "$^{*}$" if b[1] > 1 or b[2] < 1 else ""
SHORT = {"LOO, ties shuffled (refit)": "LOO, ties shuffled", "LOO, same targets, random ties (refit)":
         "LOO, fixed targets, random ties", "LOO, day-level sets, random ties (refit)": "LOO, day-level sets",
         "GTS, training ties random": "GTS, randomized ties", "GTS, training ties random, set targets":
         "GTS, randomized ties, set targets", "LOO, refit, non-tied targets only": "LOO, non-tied targets",
         "GTS, earlier cutoffs, training ties random": "GTS, earlier cutoffs, randomized ties",
         "GTS, pre-cutoff 5-core (Beauty)": "GTS, pre-cutoff 5-core",
         "GTS, pre-cutoff 5-core, training ties random": "GTS, pre-cutoff 5-core, randomized ties"}
# the five comparisons of Fig. ratio first, then the supporting checks in PROTOCOLS order (computed in PROTOCOLS
# order, which fixes the bootstrap stream)
MAIN = ["LOO, conventional", "LOO, refit", "LOO, same targets, random ties (refit)", "GTS",
        "GTS, training ties random, set targets"]
rows_of = {}
for label, spec in PROTOCOLS:
    cells, nseeds = [], {}
    for ds in DS:
        ex, s, vdir = protocol(ds, spec)
        if not ex or not s:
            cells += ["--", ""]
            continue
        vs = {k: v for k, v in val_scores(vdir).items() if k in ex}
        best_test = max(ex, key=lambda k: ex[k].mean())
        best = max(vs, key=vs.get) if len(vs) == len(ex) else best_test  # selected on validation when available
        b = boot(s, [ex[best]])["ratio"]
        bt = boot(s, [ex[best_test]])["ratio"] if best_test != best else b
        summary.setdefault(label, {})[ds] = dict(best=best, best_test=best_test, ratio_test=[float(v) for v in bt],
                                                 by_val=len(vs) == len(ex), base=float(ex[best].mean()),
                                                 sasrec=float(np.mean([x.mean() for x in s])), seeds=len(s),
                                                 ratio=[float(v) for v in b])
        dag = "" if len(vs) == len(ex) else "$^{\\dagger}$"  # no validation scores: selected on test
        cells.append(f"{ABBR[best]}{dag} {b[0]:.2f}{star(b)} [{b[1]:.2f}, {b[2]:.2f}]")
        cells.append("=" if best_test == best else f"{ABBR[best_test]} {bt[0]:.2f}{star(bt)}")
        nseeds[ds] = len(s)
    assert set(nseeds.values()) <= {len(SEEDS)}, (label, nseeds)
    rows_of[label] = SHORT.get(label, label) + " & " + " & ".join(cells) + " \\\\"
group = lambda title: f"\\multicolumn{{{1 + 2 * len(DS)}}}{{l}}{{\\textbf{{{title}}}}} \\\\"
lines += [group("Main comparisons (Fig.~\\ref{fig:ratio})")] + [rows_of[l] for l in MAIN]
lines += ["\\midrule", group("Supporting checks (Table~\\ref{tab:protocols})")]
lines += [rows_of[l] for l, _ in PROTOCOLS if l not in MAIN]
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
open(f"{OUT}/strongest.tex", "w").write("\n".join(lines) + "\n")
json.dump(summary, open(f"{R}/summary_strongest.json", "w"), indent=1)
print("\n".join(lines[13:-3]))


# ---- Table: fixed-target order control under LOO (same targets, same events, only the tie order changes) ----
CONDS = [("ID / ID", "", ""), ("rnd / rnd", "_fr", "_fr"), ("rnd / ID", "_fr", ""), ("ID / rnd", "", "_fr"),
         ("rev / rev", "_fv", "_fv")]
METHODS = [("\\pctm", "pctm"), ("MC", "mc"), ("\\pctm-sym", "pctm_sym"), ("SASRec (refit)", "sasrec")]
SEED_KEYS = [f"sasrec_s{s}" for s in SEEDS]  # seeds trained on every order variant
order_rows, order_json = [], {}
for mname, key in METHODS:
    for ci_, (cname, tr, hi) in enumerate(CONDS):
        cells = []
        for ds in DS:
            c, ref = cross(ds + hi, ds + tr, ds + hi), cross(ds, ds, ds)
            if c is None or ref is None:
                cells.append("--")
                continue
            if key == "sasrec":
                x, y = [c["sasrec"][k] for k in SEED_KEYS], [ref["sasrec"][k] for k in SEED_KEYS]
            else:
                x, y = [c[key]], [ref[key]]
            if ci_ == 0:
                cells.append(f"{np.mean([v.mean() for v in x]):.4f}")
                continue
            b = boot(x, y)["ratio"]
            star = "$^{*}$" if b[1] > 1 or b[2] < 1 else ""
            tie = tie_mask(ds)
            t = np.mean([v[tie].mean() for v in x]) / np.mean([v[tie].mean() for v in y]) - 1
            o = np.mean([v[~tie].mean() for v in x]) / np.mean([v[~tie].mean() for v in y]) - 1
            order_json.setdefault(ds, {})[f"{key}:{cname}"] = [float(v) for v in b] + [float(t), float(o)]
            cells.append(f"{100 * t:+.0f} ({100 * (b[0] - 1):+.0f}{star})")
        order_rows.append((mname if ci_ == 0 else "", cname.replace(" ", ""), cells))
lines = ["\\begin{table}[t]", "\\centering",
         "\\caption{Fixed-target order control under LOO: the same test targets and training events, with tied events "
         "in their original file order (ID: ascending ASIN on Amazon; rating-file order on ML-1M), at random (rnd), or "
         "reversed (rev), in the training data / the test histories. ID/ID: NDCG@10; other rows: change in \\% on tied "
         "targets (on all targets; $^{*}$: two-level 95\\% interval excludes 0). "
         f"SASRec refit, seeds {SPAN}.}}",
         "\\label{tab:order}", "\\setlength{\\tabcolsep}{2.5pt}", "\\scriptsize",
         "\\begin{tabular}{ll" + "c" * len(DS) + "}", "\\toprule",
         " & Train/test & " + " & ".join(NAMES[d] for d in DS) + " \\\\"]
for i, (m, cname, cells) in enumerate(order_rows):
    if i % len(CONDS) == 0:
        lines.append("\\midrule")
    lines.append(f"{m.replace(' (refit)', '')} & {cname} & " + " & ".join(cells) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(f"{OUT}/order.tex", "w").write("\n".join(lines) + "\n")
json.dump(order_json, open(f"{R}/summary_order.json", "w"), indent=1)
print("\n".join(lines[9:-3]))


# ---- Table: tie audit of benchmark files and sources ----
if os.path.exists(f"{R}/audit.jsonl"):
    rows = [json.loads(l) for l in open(f"{R}/audit.jsonl")]
    src = [r for r in rows if "name" in r]
    repos = [r for r in rows if "repo" in r]
    pct = lambda v: "--" if v is None else f"{100 * v:.1f}"
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Tie audit of raw files (audit\\_ties.py), before any split filtering; entries are percentages. Tied pairs: consecutive events of a user with identical "
             "timestamps; tied LOO targets: last event tied with the previous one; file sorted: adjacent rows with "
             "non-decreasing item id; ID order: tied pairs in ascending item-id order after a stable sort (50 is a chance "
             "reference).}",
             "\\label{tab:audit}", "\\setlength{\\tabcolsep}{2pt}", "\\footnotesize",
             "\\begin{tabular}{lrrrr}", "\\toprule",
             "Source (time unit) & Tied pairs & Tied tgt. & Sorted & ID order \\\\", "\\midrule"]
    for r in src:
        res = "d" if r["resolution_s"] == 86400 or r["name"].startswith("Amazon 201") else "s"
        name = (r["name"].replace(" (by day)", "").replace("_and_Outdoors", "").replace("_and_Games", "").replace("Amazon 20", "Amazon'").replace(" via SASRec preprocessing", ", SASRec").replace("Luxury ", "Lux. ").replace("MovieLens-", "ML-"))
        lines.append(f"{name} ({res}) & {pct(r['tie_pairs'])} & {pct(r['tied_targets'])} & "
                     f"{pct(r['file_sorted'])} & {pct(r['tied_asc'])} \\\\")
    same = sum(r["identical_to_s3rec"] for r in repos)
    lines += ["\\midrule", f"\\multicolumn{{5}}{{l}}{{Byte-identical copies of the S3-Rec files: {same} of {len(repos)} "
              f"files in {len(set(r['repo'] for r in repos))} code bases}} \\\\",
              "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(f"{OUT}/audit.tex", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[9:-3]))


# ---- Table: conventional (non-refit) fixed-target control ----
oc = []
for ds in DS:
    cells = []
    for suf in ("", "_fr"):
        p, s = expert(ds + suf)[0].mean(), np.mean([x.mean() for x in runs(ds + suf, "sasrec", "alone")])
        cells += [f"{p:.4f}", f"{s:.4f}", f"{s / p:.2f}"]
    oc.append(NAMES[ds] + " & " + " & ".join(cells) + " \\\\")
lines = ["\\begin{table*}[t]", "\\centering",
         "\\caption{The conventional (non-refit) fixed-target control. Test NDCG@10 on identical outer LOO targets and "
         f"outer training-event multisets; SASRec means use seeds {SPAN}. Random ties change the training sequences and "
         "their test histories; inner validation is recomputed. Ratios are SASRec/\\pctm point estimates. This is "
         "distinct from the refit SASRec order effects in Table~\\ref{tab:order}.}",
         "\\label{tab:order_conventional}", "\\small", "\\setlength{\\tabcolsep}{6pt}",
         "\\begin{tabular}{lrrrrrr}", "\\toprule",
         " & \\multicolumn{3}{c}{Original order} & \\multicolumn{3}{c}{Random ties, fixed targets} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "Dataset & \\pctm & SASRec & Ratio & \\pctm & SASRec & Ratio \\\\", "\\midrule"] + oc
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
open(f"{OUT}/order_conventional.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(oc))


# ---- Table: Amazon mechanism diagnostics (point summaries) ----
# refit strata from analysis_extra.py, the access-matched expert from expert_access.py, weights from the training
# logs (train.py), score dispersion from diag_weights.py, selector AUC from ltr_fit.py
AMZ = DS[:3]
ax = json.load(open(f"{R}/analysis_extra.json"))
jl = lambda ds, f: json.load(open(f"{R}/{ds}/{f}"))
pc = lambda v: f"{100 * v:+.1f}\\%"
diag = [("\\multicolumn{4}{l}{\\textbf{Access and refit (seeds " + SPAN + ")}} \\\\", None),
        ("Refit gain, target-item withheld share below 10\\%", lambda ds: pc(ax["refit"][ds]["<10%"]["refit_gain"])),
        ("Refit gain, target-item withheld share at least 50\\%", lambda ds: pc(ax["refit"][ds][">=50%"]["refit_gain"])),
        ("Refit gain, lowest future-exposure quintile", lambda ds: pc(ax["refit_by_leak"][ds][0]["refit_gain"])),
        ("Refit gain, highest future-exposure quintile", lambda ds: pc(ax["refit_by_leak"][ds][-1]["refit_gain"])),
        ("Conventional SASRec/access-matched \\pctm", lambda ds: f"""{np.mean([x.mean() for x in runs(ds, "sasrec", "alone")])
                                                                  / jl(ds, "expert_access.json")["inner"]["pctm"]["ndcg@10"]:.3f}"""),
        ("\\multicolumn{4}{l}{\\textbf{Combination weights (seeds " + SPAN + ")}} \\\\", None),
        ("Joint PoE learned $\\beta$, seed mean",
         lambda ds: f"{np.mean([(j := jl(ds, f'poe_luo1_s{s}.json'))['log'][j['best_epoch']]['beta'] for s in SEEDS]):.2f}"),
        ("Late-fusion $\\beta$ selected on validation",
         lambda ds: " or ".join(f"{b:g}" for b in sorted({jl(ds, f"sasrec_s{s}.json")["val"]["beta"] for s in SEEDS}))),
        ("\\multicolumn{4}{l}{\\textbf{Score dispersion and selection (seeds " + RSPAN + ")}} \\\\", None),
        ("Score-dispersion ratio, joint PoE", lambda ds: f"""{np.mean([jl(ds, f"diag_weights_s{s}.json")[
            "std_ratio_catalogue_top100"]["poe"]["ratio_full_top100"][1] for s in RELOAD_SEEDS]):.2f}"""),
        ("Score-dispersion ratio, late fusion", lambda ds: f"""{np.mean([jl(ds, f"diag_weights_s{s}.json")[
            "std_ratio_catalogue_top100"]["late"]["ratio_full_top100"][1] for s in RELOAD_SEEDS]):.2f}"""),
        ("Mean AUC for choosing which model hits", lambda ds: f"""{np.mean([jl(ds, f"ltr_s{s}.json")[
            "predictability_auc"]["sasrec_only_vs_pctm_only"]["auc"] for s in RELOAD_SEEDS]):.3f}""")]
lines = ["\\begin{table*}[t]", "\\centering",
         "\\caption{Amazon mechanism diagnostics under original-order LOO. Refit gains are relative changes in NDCG@10; "
         "item-share and exposure strata are defined in Section~\\ref{sec:access}. The access-matched expert excludes "
         "the same validation events as conventional SASRec. Score dispersion is $\\mathrm{std}(\\beta S)/\\mathrm{std}(z)$ "
         "over each model's fused top-100 candidates, summarized by the mean of seed-wise user medians. Selector AUC "
         "evaluates users hit by exactly one component. All entries are descriptive point summaries.}",
         "\\label{tab:diagnostics}", "\\small", "\\setlength{\\tabcolsep}{7pt}",
         "\\begin{tabular}{p{0.53\\textwidth}rrr}", "\\toprule",
         "Diagnostic & " + " & ".join(NAMES[d] for d in AMZ) + " \\\\", "\\midrule"]
for i, (lab, f) in enumerate(diag):
    if f is None and i:
        lines.append("\\midrule")
    lines.append(lab if f is None else lab + " & " + " & ".join(f(ds) for ds in AMZ) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
open(f"{OUT}/diagnostics.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines[9:-3]))
