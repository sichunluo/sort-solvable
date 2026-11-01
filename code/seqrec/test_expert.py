"""Check TrainExpert against brute force: leave-user-out scores must equal PCTM rebuilt without that user."""
import sys

import numpy as np
import torch

from common import CONFIGS, PCTM, Data, TrainExpert

name = sys.argv[1] if len(sys.argv) > 1 else "beauty"
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
d, cfg = Data(name), CONFIGS[name]
full = PCTM(d.inner, d.n_items, cfg["pctm"], cfg["max_len"], dev)
rng = np.random.default_rng(1)
lens = np.array([len(s) for s in d.inner])
users = list(rng.choice(np.flatnonzero(lens >= 2), 3, replace=False)) + [int(np.argmax(lens))]
rep = [u for u, s in enumerate(d.inner) if len(s) >= 3 and len(np.unique(s)) < len(s)]
users += rep[:2]
for luo in (True, False):
    ex = TrainExpert(full, cfg["max_len"], leave_user_out=luo)
    _, Y, S = ex.batch([d.inner[u] for u in users])
    worst = 0.0
    for b, u in enumerate(users):
        s = d.inner[u]
        ref_model = full
        if luo:
            ref_model = PCTM([x for v, x in enumerate(d.inner) if v != u], d.n_items, cfg["pctm"], cfg["max_len"], dev)
        T_n = int((Y[b] > 0).sum())
        hists = [s[:k] for k in range(len(s) - T_n, len(s))]
        ref = ref_model.score_histories(hists).float()
        got = S[b, -T_n:]
        worst = max(worst, (ref - got).abs().max().item())
        if luo:
            del ref_model
    print(f"{name} luo={luo} users={len(users)} (repeats: {len(rep[:2])}) max |diff| = {worst:.2e}")
