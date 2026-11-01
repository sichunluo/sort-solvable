"""Candidate features for LTR re-ranking over SASRec + PCTM, plus per-user features for the predictability probe.

Candidates per user = union of the SASRec top-50 and PCTM top-50 (unseen items). Val features use the inner-trained
SASRec with inner PCTM counts and histories; test features use full-train PCTM and histories (same as late fusion).
Usage: python ltr_dump.py <dataset> <sasrec seed>
"""
import json
import sys

import numpy as np
import torch

from common import CONFIGS, PCTM, RESULTS, Data
from model import SASRec, right_aligned

ds, seed = sys.argv[1], int(sys.argv[2])
K_CAND = 50
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
cfg, d = CONFIGS[ds], Data(ds)
model = SASRec(d.n_items, cfg["max_len"], **cfg["sas"]).to(dev)
model.load_state_dict(torch.load(f"{RESULTS}/{ds}/sasrec_s{seed}.pt", map_location=dev))
model.eval()
beta = json.load(open(f"{RESULTS}/{ds}/sasrec_s{seed}.json"))["val"]["beta"]
CAND_FEATURES = ["z", "z_gap", "z_logp", "z_logrank", "s", "s_gap", "s_logp", "s_logrank", "late_gap", "late_logrank",
                 "log_pop", "e_last", "log_c1_last", "e_max_last5"]
USER_FEATURES = ["log_len", "z_margin", "s_margin", "top10_overlap", "log_outdeg_last", "log_pop_last",
                 "z_entropy", "s_entropy", "z_max_logp", "s_max_logp"]


def full_rank(x):
    """0-based rank of every item under descending x (rows with -inf for seen items)."""
    order = x.argsort(1, descending=True)
    r = torch.empty_like(order)
    r.scatter_(1, order, torch.arange(x.shape[1], device=dev).expand_as(order))
    return r


@torch.no_grad()
def dump(split, pctm, mc, bs=256):
    users, ys, hists = d.split(split)
    outdeg = (mc.C > 0).sum(1).float()
    logpop = torch.log1p(pctm.freq).float()
    out = {k: [] for k in ("X", "label", "row", "cand", "U", "rank_z", "rank_s", "rank_late")}
    for s in range(0, len(users), bs):
        hs = [hists[u] for u in users[s:s + bs]]
        y = torch.from_numpy(ys[s:s + bs]).to(dev)
        B = len(hs)
        seen = d.exclude(hs, dev)
        z = model.scores(model.encode(right_aligned(hs, model.max_len))[:, -1]).masked_fill(seen, -float("inf"))
        S = pctm.score_histories(hs, dtype=torch.float32).masked_fill(seen, -float("inf"))
        late = z + beta * S
        rz, rs, rl = full_rank(z), full_rank(S), full_rank(late)
        zlp, slp = z - z.logsumexp(1, keepdim=True), S - S.logsumexp(1, keepdim=True)
        cm = torch.zeros_like(seen)
        cm.scatter_(1, z.topk(K_CAND, 1).indices, True)
        cm.scatter_(1, S.topk(K_CAND, 1).indices, True)
        rows, cols = cm.nonzero(as_tuple=True)
        last = torch.tensor([h[-1] for h in hs], device=dev)
        last5 = torch.from_numpy(np.stack([np.pad(h[-5:], (5 - len(h[-5:]), 0), mode="edge") for h in hs])).to(dev)
        emax5 = pctm.E[last5].amax(1)
        zmax, smax, lmax = z.amax(1), S.amax(1), late.amax(1)
        X = torch.stack([z[rows, cols], z[rows, cols] - zmax[rows], zlp[rows, cols], torch.log1p(rz[rows, cols].float()),
                         S[rows, cols], S[rows, cols] - smax[rows], slp[rows, cols], torch.log1p(rs[rows, cols].float()),
                         late[rows, cols] - lmax[rows], torch.log1p(rl[rows, cols].float()),
                         logpop[cols], pctm.E[last[rows], cols], torch.log1p(mc.C[last[rows], cols].float()),
                         emax5[rows, cols]], 1)
        tz, ts = z.topk(10, 1), S.topk(10, 1)
        pz, ps = zlp.exp(), slp.exp()
        U = torch.stack([torch.log1p(torch.tensor([float(len(h)) for h in hs], device=dev)),
                         tz.values[:, 0] - tz.values[:, 9], ts.values[:, 0] - ts.values[:, 9],
                         (tz.indices[:, :, None] == ts.indices[:, None, :]).any(2).float().mean(1),
                         torch.log1p(outdeg[last]), logpop[last],
                         torch.special.entr(pz).sum(1), torch.special.entr(ps).sum(1),
                         zlp.amax(1), slp.amax(1)], 1)
        out["X"].append(X.cpu().numpy().astype(np.float32))
        out["label"].append((cols == y[rows]).cpu().numpy())
        out["row"].append(rows.cpu().numpy() + s)
        out["cand"].append(cols.cpu().numpy().astype(np.int32))
        out["U"].append(U.cpu().numpy().astype(np.float32))
        for k, r in (("rank_z", rz), ("rank_s", rs), ("rank_late", rl)):
            out[k].append(r.gather(1, y[:, None])[:, 0].cpu().numpy())
    res = {k: np.concatenate(v) for k, v in out.items()}
    np.savez(f"{RESULTS}/{ds}/ltr_{split}_s{seed}.npz", **res, cand_features=np.array(CAND_FEATURES),
             user_features=np.array(USER_FEATURES), beta=beta)
    inc = np.zeros(len(users), bool)
    inc[res["row"][res["label"]]] = True
    print(f"{ds} s{seed} {split}: users {len(users)} rows {len(res['label'])} target-in-candidates {inc.mean():.3f}",
          flush=True)


for split, seqs in (("val", d.inner), ("test", d.seqs)):
    pctm = PCTM(seqs, d.n_items, cfg["pctm"], cfg["max_len"], dev)
    mc = PCTM(seqs, d.n_items, dict(cfg["pctm"], counts="exp:1:1"), 1, dev)
    dump(split, pctm, mc)
    del pctm, mc
    torch.cuda.empty_cache()
