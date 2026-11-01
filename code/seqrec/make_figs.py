"""Figures for the paper. Run locally: python -I code/seqrec/make_figs.py data/seqrec/results data/seqrec/processed paper_icde/figs"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R, P, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
DS = ["beauty", "sports", "toys", "ml1m"]
NAMES = {"beauty": "Beauty", "sports": "Sports", "toys": "Toys", "ml1m": "ML-1M"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6, "pdf.fonttype": 42})
SEEDS = (0, 1, 2, 3, 4)  # every neural condition is averaged over the same five seeds
seed_of = lambda name: int(name.rsplit("_s", 1)[1].split("_")[0])
COLORS = {"pctm": "#4c72b0", "sasrec": "#dd8452", "late": "#55a868", "pop": "#8c8c8c"}


def nd(r):
    return np.where(r < 10, 1 / np.log2(r + 2.0), 0.0)


def runs(ds, tag, key="alone"):
    return [nd(np.load(f)[key]) for f in sorted(glob.glob(f"{R}/{ds}/{tag}_s[0-9]_ranks.npz"))
            if seed_of(f) in SEEDS and key in np.load(f).files]


def pctm(ds):
    return nd(np.load(f"{R}/{ds}/experts_ranks.npz")["pctm_test"])


# ---- Fig. 1: SASRec / strongest training-free baseline per protocol (from make_tables.py) ----
rng = np.random.default_rng(0)


def ratio_boot(s, p, n=500):
    idx = rng.integers(0, len(s), (n, len(s)))
    r = s[idx].mean(1) / p[idx].mean(1)
    return s.mean() / p.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)


summ = json.load(open(f"{R}/summary_strongest.json"))
# the five main comparisons of the paper (the other protocols stay in the strongest-baseline table)
conds = [("LOO, conventional", "LOO\nconventional"), ("LOO, refit", "LOO\nrefit"),
         ("LOO, same targets, random ties (refit)", "LOO, same targets\nrandom ties (refit)"),
         ("GTS", "GTS\nfirst eligible target"), ("GTS, training ties random, set targets", "GTS, random ties\nset targets")]
fig, ax = plt.subplots(figsize=(7.16, 2.35))
markers = {"beauty": "o", "sports": "s", "toys": "^", "ml1m": "D"}
for i, ds in enumerate(DS):
    for j, (c, _) in enumerate(conds):
        m, lo, hi = summ[c][ds]["ratio"]
        x = j + (i - 1.5) * 0.13
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt=markers[ds], ms=4, color=f"C{i}", lw=0.85,
                    capsize=1.8, label=NAMES[ds] if j == 0 else None)
        if summ[c][ds]["best_test"] != summ[c][ds]["best"]:  # best baseline on test, where it differs
            ax.plot(x, summ[c][ds]["ratio_test"][0], markers[ds], ms=4, mfc="none", mec=f"C{i}", mew=0.8)
ax.axhline(1.0, color="black", lw=0.65, ls="--")
ax.axvline(2.5, color="#999999", lw=0.65)
ax.set_xticks(range(len(conds)))
ax.set_xticklabels([lab for _, lab in conds], fontsize=7)
ax.set_ylabel("SASRec / strongest\nbaseline (NDCG@10)", fontsize=8)
ax.tick_params(axis="y", labelsize=7)
ax.set_xlim(-0.45, len(conds) - 0.55)
ax.legend(fontsize=7, ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.25), handletextpad=0.3,
          columnspacing=1.2)
for x, text in ((1, "Fixed outer LOO targets"), (3.5, "Global temporal tasks")):
    ax.text(x, 0.99, text, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.7, color="#555555")
fig.tight_layout(pad=0.25)
fig.savefig(f"{OUT}/ratio.pdf"); fig.savefig(f"{OUT}/ratio_preview.png", dpi=220)
print("ratio figure written")

# ---- Fig. 2: rolling evaluation by gap, SASRec/PCTM ratio with ID-ordered vs random ties ----
def strata_masks(gaps, amazon):
    if amazon:
        return {"0 d": gaps == 0, "1-7 d": (gaps > 0) & (gaps <= 7 * 86400),
                "8-30 d": (gaps > 7 * 86400) & (gaps <= 30 * 86400), ">30 d": gaps > 30 * 86400}
    return {"0 s": gaps == 0, "<1 h": (gaps > 0) & (gaps <= 3600), ">1 h": gaps > 3600}


fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.8))
for ax, ds in zip(axes, DS):
    for off, (suf, lab, col) in zip((-0.18, 0.18), (("_t", "original file order", "#c44e52"),
                                                   ("_ts", "ties shuffled", "#4c72b0"))):
        z = np.load(f"{R}/{ds}{suf}/rolling_ranks.npz")
        keep = np.load(f"{R}/{ds}{suf}/rolling_keep.npy")  # without the validation targets (analysis_extra.py)
        sas = np.mean([nd(z[k]) for k in z.files if k.startswith("sasrec_s") and seed_of(k) in SEEDS], 0)[keep]
        pc = nd(z["pctm"])[keep]
        masks = strata_masks(z["gaps"][keep], ds != "ml1m")
        for x, (k, m) in enumerate(masks.items()):
            r, lo, hi = ratio_boot(sas[m], pc[m])
            ax.bar(x + off, r, 0.34, color=col, alpha=0.85, linewidth=0, label=lab if x == 0 else None)
            ax.errorbar(x + off, r, yerr=[[r - lo], [hi - r]], fmt="none", ecolor="k", lw=0.6, capsize=1.2)
        if suf == "_t":  # label strata with the shares of the original (ID-ordered) split
            share = [f"{k}\n({m.mean():.0%})" for k, m in masks.items()]
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    ax.set_xticks(range(len(share)))
    ax.set_xticklabels(share, fontsize=5.5)
    ax.set_title(NAMES[ds], fontsize=8, pad=2)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_ylim(0.5, 2.0)
axes[0].set_ylabel("SASRec / PCTM", fontsize=7)
axes[0].legend(fontsize=5.5, frameon=False, loc="upper left", handlelength=1.0)
fig.tight_layout(pad=0.2, w_pad=0.6)
fig.savefig(f"{OUT}/rolling.pdf"); fig.savefig(f"{OUT}/rolling_preview.png", dpi=220)
print("rolling figure written")
