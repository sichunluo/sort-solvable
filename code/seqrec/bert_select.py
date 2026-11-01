"""Pick the BERT4Rec cloze masking and architecture per dataset on validation NDCG@10 (seed 0 tuning runs of
run_bert.sh) and write the five-seed job list. Usage: python bert_select.py <results> <jobs out>"""
import json
import os
import sys
import time

R, OUT = sys.argv[1], sys.argv[2]
GRID = [(p, q, a) for p in ("0.2", "0.4", "0.6") for q in ("0", "0.2") for a in ("sas", "orig")]
ARCH = {"sas": [], "orig": ["--blocks", "2", "--heads", "2", "--dropout", "0.1"]}  # SASRec's / Sun et al.'s
SPLITS = [("", []), ("", ["--refit", "1"]), ("_fr", []), ("_fr", ["--refit", "1"]), ("_fv", ["--refit", "1"]),
          ("_t", ["--refit", "1"]), ("_tsx", ["--refit", "1"])]


def tuning_running():
    """True while a tuning run (tag bert_m*) of any worker, including run_bert_extra.sh's, is still training."""
    for p in filter(str.isdigit, os.listdir("/proc")):
        try:
            c = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if b"train.py" in c and any(x.startswith(b"bert_m") for x in c):
            return True
    return False


while tuning_running():
    time.sleep(60)
chosen, lines = {}, []
for ds in ("beauty", "sports", "toys", "ml1m"):
    val = {}
    for p, q, a in GRID:
        try:
            r = json.load(open(f"{R}/{ds}/bert_m{p}_l{q}_{a}_s0.json"))
        except FileNotFoundError:
            print(f"missing {ds} {p} {q} {a}", file=sys.stderr)
            continue
        val[f"{p}/{q}/{a}"] = dict(ndcg10=r["val"]["alone"]["ndcg@10"], best_epoch=r["best_epoch"])
    best = max(val, key=lambda k: val[k]["ndcg10"])
    p, q, a = best.split("/")
    chosen[ds] = dict(mask_prob=float(p), mask_last=float(q), arch=a, grid=val)
    print(ds, best, val[best], flush=True)
    for seed in range(5):
        for suf, extra in SPLITS:
            lines.append(" ".join([ds + suf, "--arch", "bert", "--mask_prob", p, "--mask_last", q, *ARCH[a],
                                   "--epochs", "1000", "--patience", "40", *extra, "--seed", str(seed)]))
json.dump(chosen, open(f"{R}/bert_config.json", "w"), indent=1)
open(OUT, "w").write("\n".join(lines) + "\n")
