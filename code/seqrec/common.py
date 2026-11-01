"""Shared data, PCTM expert and evaluation code for the shortcut-delegation pilot."""
import hashlib
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch

DATA = os.environ.get("SEQREC_DATA", os.path.expanduser("~/seqrec/data"))
RESULTS = os.environ.get("SEQREC_RESULTS", os.path.expanduser("~/seqrec/results"))

# Frozen per-dataset configs from capacity-probes configs/table1.json (PCTM + SASRec+ architecture).
CONFIGS = {
    "beauty": dict(max_len=50, pctm=dict(counts="inv:10:0.5", tau=11250.0, kernel="exp:0.6", pop_boost=-0.04),
                   sas=dict(d=64, blocks=1, heads=1, dropout=0.2), epochs=200, patience=20),
    "sports": dict(max_len=50, pctm=dict(counts="inv:10:0.5", tau=15000.0, kernel="tail:10:0.8:0.7", pop_boost=0.0),
                   sas=dict(d=64, blocks=1, heads=1, dropout=0.2), epochs=200, patience=20),
    "toys": dict(max_len=50, pctm=dict(counts="exp:20:0.7", tau=15000.0, kernel="tail:7:0.8:0.6", pop_boost=0.0),
                 sas=dict(d=64, blocks=1, heads=1, dropout=0.2), epochs=200, patience=20),
    "ml1m": dict(max_len=200, pctm=dict(counts="inv:20:1.0", tau=150.0, kernel="tail:7:0.8:0.6", pop_boost=-0.4),
                 sas=dict(d=64, blocks=2, heads=1, dropout=0.1), epochs=100, patience=20),
    "ml20m": dict(max_len=200, pctm=dict(counts="exp:10:0.5", tau=225.0, kernel="tail:7:0.8:0.6", pop_boost=-0.5),
                  sas=dict(d=256, blocks=4, heads=8, dropout=0.2), epochs=100, patience=10),
}
# Temporal (_t), leaky-fold (_tl0/_tl1), tie-shuffled LOO (_shuf) and tie-shuffled temporal (_ts) variants reuse the LOO configs,
# as do the order-controlled variants of prep_fixed.py (_fr, _fv, _day, _tset).
CONFIGS.update({f"{k}{v}": c for k, c in list(CONFIGS.items()) for v in ("_t", "_tl0", "_tl1", "_shuf", "_ts", "_tsx", "_shuf1", "_shuf2", "_shuf3", "_shuf4",
                                                                    "_fr", "_fv", "_day", "_tset", "_fr1", "_fr2",
                                                                    "_tq", "_tqx", "_tp", "_tpx")})
REPORTED_NDCG10 = {  # capacity-probes Table 1
    "beauty": dict(mc=.0492, sasrec_plus=.0537, esasrec=.0524, seqrules=.0605, pctm=.0635),
    "sports": dict(mc=.0251, sasrec_plus=.0315, esasrec=.0324, seqrules=.0371, pctm=.0368),
    "toys": dict(mc=.0548, sasrec_plus=.0575, esasrec=.0533, seqrules=.0730, pctm=.0738),
    "ml1m": dict(mc=.1240, sasrec_plus=.1662, esasrec=.1739, seqrules=.1505, pctm=.1815),
    "ml20m": dict(mc=.1050, sasrec_plus=.1806, esasrec=.1969, seqrules=.1115, pctm=.1431),
}


class Data:
    """Train sequences, the final holdout, and an inner leave-one-out validation split of train."""

    def __init__(self, name):
        z = np.load(f"{DATA}/{name}.npz")
        self.name, self.catalog = name, z["catalog"]
        self.n_items = len(self.catalog)
        users, items = z["users"], z["items"]
        self.user_ids, starts, counts = np.unique(users, return_index=True, return_counts=True)
        self.seqs = [items[s:s + c] for s, c in zip(starts, counts)]
        self.test_u = np.searchsorted(self.user_ids, z["ho_users"])
        assert np.all(self.user_ids[self.test_u] == z["ho_users"])
        self.test_y = z["ho_items"].astype(np.int64)
        if "inner_mask" in z.files:
            # Temporal split (prep_time.py): inner = events before the validation cutoff, val targets given.
            mask = z["inner_mask"]
            self.inner = [items[s:s + c][mask[s:s + c]] for s, c in zip(starts, counts)]
            self.val_u = np.searchsorted(self.user_ids, z["val_users"])
            self.val_y = z["val_items"].astype(np.int64)
        else:
            # Leave-one-out: each user's last train event is the validation target, filtered like the holdout.
            self.inner = [s[:-1] for s in self.seqs]
            in_inner = np.zeros(self.n_items, bool)
            in_inner[np.concatenate(self.inner)] = True
            val_u = [u for u, s in enumerate(self.seqs)
                     if len(s) >= 2 and in_inner[s[-1]] and not np.any(s[:-1] == s[-1])]
            self.val_u = np.asarray(val_u)
            self.val_y = np.asarray([self.seqs[u][-1] for u in val_u], dtype=np.int64)
        self.sha_rank = stable_hash_ranks(self.catalog)
        # Optional: event times per train event, and a restricted candidate catalogue (leaky folds).
        times = z["times"] if "times" in z.files else None  # load once: each z[...] access re-reads the array
        self.times = [times[s:s + c] for s, c in zip(starts, counts)] if times is not None else None
        self.blocked = None
        if "cand_catalog" in z.files:
            self.blocked = ~np.isin(self.catalog, z["cand_catalog"])

    def exclude(self, hists, device):
        """Items a user may not be recommended: their history, plus items outside the candidate catalogue."""
        m = seen_mask(hists, self.n_items, device)
        if self.blocked is not None:
            m |= torch.from_numpy(self.blocked).to(device)[None]
        return m

    def split(self, which):
        if which == "val":
            return self.val_u, self.val_y, self.inner
        return self.test_u, self.test_y, self.seqs


def stable_hash_ranks(catalog):
    """PCTM tie-break: ascending SHA-256 of the UTF-8 external id (as in capacity-probes ranking.py)."""
    digests = [hashlib.sha256(str(i).encode()).digest() for i in catalog]
    order = sorted(range(len(catalog)), key=lambda i: (digests[i], str(catalog[i])))
    ranks = np.empty(len(catalog), dtype=np.int64)
    ranks[np.asarray(order)] = np.arange(len(catalog))
    return ranks


def count_weights(spec):
    kind, *parts = spec.split(":")
    offsets = np.arange(int(parts[0]), dtype=np.float64)
    if kind == "exp":
        return np.power(float(parts[1]), offsets)
    if kind == "inv":
        return 1.0 / np.power(offsets + 1.0, float(parts[1]))
    raise ValueError(spec)


def kernel_weights(spec, length):
    kind, *parts = spec.split(":")
    ages = np.arange(length, dtype=np.float64)
    if kind == "exp":
        w = np.power(float(parts[0]), ages)
    elif kind == "tail":
        head = min(int(parts[0]), length)
        recent_mass, beta = float(parts[1]), float(parts[2])
        w = np.zeros(length)
        recent = np.power(beta, np.arange(head, dtype=np.float64))
        w[:head] = recent_mass * recent / recent.sum()
        if length > head:
            w[head:] = (1.0 - recent_mass) / (length - head)
    else:
        raise ValueError(spec)
    return w / w.sum()


class PCTM:
    """Dense GPU PCTM (Petrov et al.): S(a|h) = sum_j w_j log1p(C[h_j,a] |I|/tau) + boost * log(f_a + 1).

    Per-history constants of log P(a|b) are dropped (they cancel in ranking and in a softmax).
    """

    def __init__(self, seqs, n_items, cfg, max_history, device):
        self.n, self.L, self.device = n_items, max_history, device
        self.cw = count_weights(cfg["counts"])
        self.scale = n_items / cfg["tau"]
        self.boost = cfg["pop_boost"]
        self.kspec = cfg["kernel"]
        self.kern = {l: kernel_weights(self.kspec, l) for l in range(1, max_history + 1)}
        self.dtype = torch.float64 if n_items < 20000 else torch.float32
        users = np.repeat(np.arange(len(seqs)), [len(s) for s in seqs])
        items = np.concatenate(seqs).astype(np.int64)
        C = torch.zeros(n_items * n_items, dtype=self.dtype, device=device)
        for d, w in enumerate(self.cw, 1):
            same = users[:-d] == users[d:]
            idx = torch.from_numpy(items[:-d][same] * n_items + items[d:][same]).to(device)
            C.index_add_(0, idx, torch.full(idx.shape, w, dtype=self.dtype, device=device))
        self.C = C.view(n_items, n_items)
        self._E = None
        freq = torch.from_numpy(np.bincount(items, minlength=n_items).astype(np.float64)).to(device)
        self.freq = freq
        self.pop = (self.boost * torch.log(freq + 1.0)).float()

    @property
    def E(self):
        """float32 evidence log1p(C |I|/tau), built once (training and float32 evaluation)."""
        if self._E is None:
            self._E = torch.log1p(self.C * self.scale).float()
        return self._E

    def _rows_budget(self, width):
        return max(1, int(1.2e8 // (width * self.n)))

    def score_histories(self, hists, dtype=None):
        """Full-catalogue expert scores for complete histories (evaluation). Returns (B, n).

        dtype float32 uses the cached evidence; the default (float64 for small catalogues) recomputes it exactly.
        """
        dtype = dtype or self.dtype
        hs = [h[-self.L:][::-1] for h in hists]
        Lb = max(len(h) for h in hs)
        idx = np.zeros((len(hs), Lb), dtype=np.int64)
        w = np.zeros((len(hs), Lb))
        for b, h in enumerate(hs):
            idx[b, :len(h)] = h
            w[b, :len(h)] = self.kern[len(h)]
        idx = torch.from_numpy(idx).to(self.device)
        w = torch.from_numpy(w).to(self.device, dtype)
        out = torch.empty(len(hs), self.n, dtype=dtype, device=self.device)
        step = self._rows_budget(Lb)
        for s in range(0, len(hs), step):
            if dtype == torch.float32:
                E = self.E[idx[s:s + step]]
            else:
                E = torch.log1p(self.C[idx[s:s + step]].to(dtype) * self.scale)
            out[s:s + step] = torch.bmm(w[s:s + step, None, :], E)[:, 0]
        return out + self.pop.to(dtype)


class TrainExpert:
    """PCTM scores for every training position of a batch, optionally leave-user-out.

    A training window predicts targets s[n-T_n .. n-1] of a user's inner sequence s. The expert for target k uses
    the newest min(k, L) items before k, which all lie in z = s[o:n-1]. Leave-user-out removes the user's own
    transition counts (and own popularity counts) so that, as at test time, the expert never counts the target pair.
    """

    def __init__(self, pctm, max_len, leave_user_out):
        self.p, self.T, self.luo = pctm, max_len, leave_user_out
        self.Z = max_len + pctm.L - 1
        self._W = {}

    def window(self, n):
        T_n = min(self.T, n - 1)
        o = max(0, n - T_n - self.p.L)
        return T_n, o, n - 1 - o

    def W(self, n):
        key = min(n, self.T + self.p.L + 1)
        if key not in self._W:
            T_n, o, Z_n = self.window(key)
            W = np.zeros((self.T, self.Z), dtype=np.float32)
            for t in range(T_n):
                k = key - T_n + t
                l = min(k, self.p.L)
                W[self.T - T_n + t, self.Z - Z_n + (k - 1 - np.arange(l)) - o] = self.p.kern[l]
            self._W[key] = W
        return self._W[key]

    def own_entries(self, s, o, Z_n):
        """(j, a, m): the user's own contribution m to C[z_j, a] for z = s[o:o+Z_n]."""
        n = len(s)
        z = s[o:o + Z_n]
        if len(np.unique(s)) == n:
            js = np.arange(Z_n)
            P = o + js
        else:
            pos = defaultdict(list)
            for p, it in enumerate(s):
                pos[it].append(p)
            js = np.asarray([j for j, it in enumerate(z) for _ in pos[it]], dtype=np.int64)
            P = np.asarray([p for it in z for p in pos[it]], dtype=np.int64)
        D = np.arange(1, len(self.p.cw) + 1)
        Q = P[:, None] + D[None, :]
        ok = Q < n
        return (np.broadcast_to(js[:, None], Q.shape)[ok], s[Q[ok]],
                np.broadcast_to(self.p.cw[None, :], Q.shape)[ok])

    def batch(self, seqs, scores=True):
        """Inputs, targets (both +1, 0 = pad, right-aligned) and the expert scores (B, T_b, n) or None."""
        B = len(seqs)
        wins = [self.window(len(s)) for s in seqs]
        T_b = max(w[0] for w in wins)
        Z_b = max(w[2] for w in wins)
        X = np.zeros((B, T_b), dtype=np.int64)
        Y = np.zeros((B, T_b), dtype=np.int64)
        for b, (s, (T_n, _, _)) in enumerate(zip(seqs, wins)):
            X[b, T_b - T_n:] = s[len(s) - 1 - T_n:len(s) - 1] + 1
            Y[b, T_b - T_n:] = s[len(s) - T_n:] + 1
        dev = self.p.device
        if not scores:
            return torch.from_numpy(X).to(dev), torch.from_numpy(Y).to(dev), None
        Zi = np.zeros((B, Z_b), dtype=np.int64)
        W = np.zeros((B, T_b, Z_b), dtype=np.float32)
        own_b, own_j, own_a, own_m = [], [], [], []
        pop_b, pop_a, pop_c = [], [], []
        for b, (s, (T_n, o, Z_n)) in enumerate(zip(seqs, wins)):
            n = len(s)
            Zi[b, Z_b - Z_n:] = s[o:n - 1]
            W[b] = self.W(n)[self.T - T_b:, self.Z - Z_b:]
            if self.luo:
                j, a, m = self.own_entries(s, o, Z_n)
                own_b.append(np.full(len(j), b))
                own_j.append(j + Z_b - Z_n)
                own_a.append(a)
                own_m.append(m)
                items, cnt = np.unique(s, return_counts=True)
                pop_b.append(np.full(len(items), b))
                pop_a.append(items)
                pop_c.append(cnt)
        p = self.p
        Zi_t = torch.from_numpy(Zi).to(dev)
        W_t = torch.from_numpy(W).to(dev)
        S = torch.empty(B, T_b, p.n, dtype=torch.float32, device=dev)
        pop = p.pop.expand(B, p.n).clone()
        if self.luo:
            # Aggregate the user's own counts per (b, j, a), then turn them into exact evidence corrections.
            key = (np.concatenate(own_b) * Z_b + np.concatenate(own_j)) * p.n + np.concatenate(own_a)
            key, inv = np.unique(key, return_inverse=True)
            m = np.bincount(inv, weights=np.concatenate(own_m))
            key = torch.from_numpy(key).to(dev)
            ob, oj, oa = key // (Z_b * p.n), (key // p.n) % Z_b, key % p.n
            c = p.C[Zi_t[ob, oj], oa]
            delta = (torch.log1p((c - torch.from_numpy(m).to(dev, p.dtype)).clamp_min(0) * p.scale)
                     - torch.log1p(c * p.scale)).float()
            pb, pa = (torch.from_numpy(np.concatenate(v)).to(dev) for v in (pop_b, pop_a))
            pc = torch.from_numpy(np.concatenate(pop_c).astype(np.float64)).to(dev)
            pop[pb, pa] = (p.boost * torch.log(p.freq[pa] + 1.0 - pc)).float()
        step = p._rows_budget(Z_b)
        for s0 in range(0, B, step):
            s1 = min(B, s0 + step)
            rows = p.E[Zi_t[s0:s1]]
            if self.luo:
                sel = (ob >= s0) & (ob < s1)
                rows.index_put_((ob[sel] - s0, oj[sel], oa[sel]), delta[sel], accumulate=True)
            S[s0:s1] = torch.bmm(W_t[s0:s1], rows)
        S += pop[:, None, :]
        return torch.from_numpy(X).to(dev), torch.from_numpy(Y).to(dev), S


def target_ranks(scores, y, tie):
    """0-based rank of y under (score desc, tie asc); scores already have seen items set to -inf."""
    st = scores.gather(1, y[:, None])
    gt = (scores > st).sum(1)
    eq = ((scores == st) & (tie[None, :] < tie[y][:, None])).sum(1)
    return gt + eq


def seen_mask(hists, n, device):
    b = np.concatenate([np.full(len(h), i) for i, h in enumerate(hists)])
    a = np.concatenate(hists)
    m = torch.zeros(len(hists), n, dtype=torch.bool, device=device)
    m[torch.from_numpy(b).to(device), torch.from_numpy(a.astype(np.int64)).to(device)] = True
    return m


def metrics(ranks, k=10):
    r = np.asarray(ranks)
    hit = r < k
    return {f"ndcg@{k}": float(np.mean(np.where(hit, 1.0 / np.log2(r + 2.0), 0.0))),
            f"recall@{k}": float(np.mean(hit)), "n": int(len(r))}


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
