"""SASRec with full-softmax CE, either alone (+ late fusion with PCTM tuned on val) or trained as a
product of experts with PCTM: logits = z_theta + beta * S_pctm, with leave-user-out expert counts (--luo 1).
--arch gru / bert swap the encoder for GRU4Rec / BERT4Rec (cloze objective, see cloze())."""
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import CONFIGS, PCTM, RESULTS, Data, TrainExpert, metrics, save_json, target_ranks
from model import ARCHS, right_aligned

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--method", choices=("sasrec", "poe"), default="sasrec")
ap.add_argument("--luo", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--epochs", type=int)
ap.add_argument("--patience", type=int)
ap.add_argument("--beta_init", type=float, default=1.0)
ap.add_argument("--fix_beta", type=int, default=0)
ap.add_argument("--beta_val", type=int, default=0)  # poe: set beta to the val-optimal late-fusion weight each epoch
ap.add_argument("--refit", type=int, default=0)
ap.add_argument("--refit_scale", type=float, default=1.0)  # refit epochs = scale x early-stopped epochs (sensitivity)
ap.add_argument("--arch", choices=("sasrec", "gru", "bert"), default="sasrec")
ap.add_argument("--mask_prob", type=float, default=0.2)  # bert: cloze masking probability
ap.add_argument("--mask_last", type=float, default=0.0)  # bert: share of sequences that mask only the last item
for k, t in (("blocks", int), ("heads", int), ("dropout", float)):  # override the dataset's SASRec architecture
    ap.add_argument(f"--{k}", type=t)
ap.add_argument("--tag")
args = ap.parse_args()
assert not (args.arch == "bert" and args.method == "poe")
cfg = CONFIGS[args.dataset]
cfg = dict(cfg, sas=dict(cfg["sas"], **{k: getattr(args, k) for k in ("blocks", "heads", "dropout") if getattr(args, k) is not None}))
epochs, patience = args.epochs or cfg["epochs"], args.patience or cfg["patience"]
tag = args.tag or ((args.arch if args.method == "sasrec" else f"poe_luo{args.luo}" + ("_bval" if args.beta_val else "")) + ("_refit" if args.refit and "_t" not in args.dataset else "")
                    + (f"_rs{args.refit_scale:g}" if args.refit_scale != 1.0 else ""))
if os.path.exists(f"{RESULTS}/{args.dataset}/{tag}_s{args.seed}.json"):
    sys.exit(f"done already: {args.dataset} {tag} s{args.seed}")
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True  # training only; evaluate() switches it off for exact ranking
dev = "cuda"
LATE_BETAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]


@torch.no_grad()
def evaluate(model, d, split, pctm, betas, bs=512):
    """Ranks of the target under model + beta * expert for each beta (beta 0 = model alone)."""
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    users, ys, hists = d.split(split)
    sha = torch.from_numpy(d.sha_rank).to(dev)
    out = {b: [] for b in betas}
    for s in range(0, len(users), bs):
        hs = [hists[u] for u in users[s:s + bs]]
        y = torch.from_numpy(ys[s:s + bs]).to(dev)
        logits = model.scores(model.last(right_aligned(hs, model.max_len)))
        S = pctm.score_histories(hs, dtype=torch.float32) if pctm is not None and any(betas) else None
        seen = d.exclude(hs, dev)
        for b in betas:
            sc = logits + b * S if b else logits.clone()
            sc[seen] = -float("inf")
            out[b].append(target_ranks(sc, y, sha).cpu().numpy())
    model.train()
    torch.backends.cuda.matmul.allow_tf32 = True
    return {b: np.concatenate(v) for b, v in out.items()}


d = Data(args.dataset)
poe = args.method == "poe"
pctm_inner = PCTM(d.inner, d.n_items, cfg["pctm"], cfg["max_len"], dev)
expert = TrainExpert(pctm_inner, cfg["max_len"], leave_user_out=bool(args.luo))


def new_model():
    m = ARCHS[args.arch](d.n_items, cfg["max_len"], **cfg["sas"]).to(dev)
    with torch.no_grad():
        m.beta.fill_(args.beta_init)
    params = [p for n, p in m.named_parameters() if n != "beta" or (poe and not (args.fix_beta or args.beta_val))]
    return m, torch.optim.Adam(params, lr=args.lr)


def cloze(model, X, Y):
    """BERT4Rec cloze batch over the user's last min(T, n) training items: each item is masked with probability
    mask_prob, a share mask_last of the sequences masks only the last item (Sun et al. 2019), and every sequence
    masks at least one item. Returns the logits and targets (+1) of the masked positions."""
    Z = torch.cat([X, Y[:, -1:]], 1)[:, -model.max_len:]
    m = (torch.rand(Z.shape, device=Z.device) < args.mask_prob) & (Z > 0)
    only_last = torch.rand(len(Z), device=Z.device) < args.mask_last
    m[only_last] = False
    m[:, -1] |= ~m.any(1)
    return model.scores(model.encode(Z.masked_fill(m, model.mask_id))[m]), Z[m]


def run_epoch(model, opt, seqs, users, expert):
    tot, nb = 0.0, 0
    order = rng.permutation(users)
    for s in range(0, len(order), args.batch):
        X, Y, S = expert.batch([seqs[u] for u in order[s:s + args.batch]], scores=poe)
        if args.arch == "bert":
            logits, y = cloze(model, X, Y)
        else:
            valid = Y > 0
            logits, y = model.scores(model.encode(X)[valid]), Y[valid]
            if poe:
                logits = logits + model.beta * S[valid]
        loss = F.cross_entropy(logits, y - 1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot, nb = tot + loss.item(), nb + 1
    return tot / nb


model, opt = new_model()
train_users = np.asarray([u for u, s in enumerate(d.inner) if len(s) >= 2])
print(f"{args.dataset} {tag} seed={args.seed} items={d.n_items} train_users={len(train_users)} "
      f"val={len(d.val_u)} test={len(d.test_u)}", flush=True)

best, best_state, best_ep, bad, log = -1.0, None, -1, 0, []
for ep in range(epochs):
    t0 = time.time()
    loss = run_epoch(model, opt, d.inner, train_users, expert)
    if poe and args.beta_val:
        # Out-of-sample combination weight: the next epoch trains with the beta that is best on validation users.
        rs = evaluate(model, d, "val", pctm_inner, LATE_BETAS)
        bval = max(rs, key=lambda b: metrics(rs[b])["ndcg@10"])
        with torch.no_grad():
            model.beta.fill_(bval)
        r = rs[bval]
    else:
        bval = model.beta.item() if poe else 0.0
        r = evaluate(model, d, "val", pctm_inner if poe else None, [bval])[bval]
    m = metrics(r)
    log.append(dict(epoch=ep, loss=loss, beta=model.beta.item(), sec=time.time() - t0, **m))
    print(f"ep {ep} loss {loss:.4f} beta {model.beta.item():.3f} val ndcg {m['ndcg@10']:.4f} "
          f"rec {m['recall@10']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    if m["ndcg@10"] > best:
        best, best_ep, bad, best_state = m["ndcg@10"], ep, 0, copy.deepcopy(model.state_dict())
    else:
        bad += 1
        if bad >= patience:
            break

model.load_state_dict(best_state)
res = dict(args=vars(args), tag=tag, best_epoch=best_ep, log=log)
torch.save(best_state, f"{RESULTS}/{args.dataset}/{tag}_s{args.seed}.pt")
if poe:
    beta = model.beta.item()
    val = evaluate(model, d, "val", pctm_inner, sorted(set(LATE_BETAS) | {beta}))
    grid = {b: metrics(v)["ndcg@10"] for b, v in val.items()}
    btuned = max(grid, key=grid.get)
    res["val"] = {"fused": metrics(val[beta]), "alone": metrics(val[0.0]), "beta": beta,
                  "tuned": metrics(val[btuned]), "beta_tuned": btuned, "grid": grid}
else:
    val = evaluate(model, d, "val", pctm_inner, LATE_BETAS)
    grid = {b: metrics(v)["ndcg@10"] for b, v in val.items()}
    bstar = max(grid, key=grid.get)
    res["val"] = {"alone": metrics(val[0.0]), "late": metrics(val[bstar]), "beta": bstar, "grid": grid}
del expert, pctm_inner
torch.cuda.empty_cache()
pctm_full = PCTM(d.seqs, d.n_items, cfg["pctm"], cfg["max_len"], dev)
if args.refit:
    # Retrain from scratch on the full train split for the early-stopped epoch count (val choices kept).
    torch.manual_seed(args.seed)
    model, opt = new_model()
    expert_full = TrainExpert(pctm_full, cfg["max_len"], leave_user_out=bool(args.luo))
    full_users = np.asarray([u for u, s in enumerate(d.seqs) if len(s) >= 2])
    n_refit = max(1, round(args.refit_scale * (best_ep + 1)))
    for ep in range(n_refit):
        print(f"refit ep {ep} loss {run_epoch(model, opt, d.seqs, full_users, expert_full):.4f}", flush=True)
    del expert_full
    res["refit_epochs"] = n_refit
    torch.save(model.state_dict(), f"{RESULTS}/{args.dataset}/{tag}_s{args.seed}_refit.pt")
if poe:
    beta = model.beta.item()
    test = evaluate(model, d, "test", pctm_full, sorted({0.0, beta, btuned}))
    res["test"] = {"fused": metrics(test[beta]), "alone": metrics(test[0.0]), "beta": beta,
                   "tuned": metrics(test[btuned]), "beta_tuned": btuned}
    ranks = {"fused": test[beta], "alone": test[0.0], "tuned": test[btuned]}
else:
    test = evaluate(model, d, "test", pctm_full, sorted({0.0, bstar}))
    res["test"] = {"alone": metrics(test[0.0]), "late": metrics(test[bstar]), "beta": bstar}
    ranks = {"alone": test[0.0], "late": test[bstar]}
print("RESULT", args.dataset, tag, args.seed, {k: v for k, v in res["test"].items()}, flush=True)
save_json(f"{RESULTS}/{args.dataset}/{tag}_s{args.seed}.json", res)
np.savez(f"{RESULTS}/{args.dataset}/{tag}_s{args.seed}_ranks.npz", **ranks)
