"""SASRec (pre-LN, tied output embeddings), GRU4Rec and BERT4Rec encoders used by train.py and the analyses."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.drop, self.heads, self.p = nn.Dropout(dropout), heads, dropout

    def forward(self, h, mask):
        B, T, d = h.shape
        q, k, v = self.qkv(self.ln1(h)).view(B, T, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.p if self.training else 0.0)
        h = h + self.drop(self.out(a.transpose(1, 2).reshape(B, T, d)))
        return h + self.drop(self.ff(self.ln2(h)))


class SASRec(nn.Module):
    causal = True

    def __init__(self, n_items, max_len, d, blocks, heads, dropout):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        nn.init.xavier_normal_(self.item.weight)
        nn.init.xavier_normal_(self.pos.weight)
        self.blocks = nn.ModuleList(Block(d, heads, dropout) for _ in range(blocks))
        self.drop, self.ln, self.d, self.max_len = nn.Dropout(dropout), nn.LayerNorm(d), d, max_len
        self.beta = nn.Parameter(torch.tensor(1.0))  # expert weight, used only by the PoE objective

    def encode(self, X):
        """X: (B, T) item ids + 1, right-aligned, 0 = pad. Returns (B, T, d)."""
        B, T = X.shape
        valid = X > 0
        h = self.item(X) * math.sqrt(self.d) + self.pos.weight[self.max_len - T:]
        h = self.drop(h) * valid[..., None]
        eye = torch.eye(T, dtype=torch.bool, device=X.device)
        mask = valid[:, None, :] | eye[None]  # pads only see themselves
        if self.causal:
            mask = mask & (eye.cumsum(0) > 0)[None]
        for blk in self.blocks:
            h = blk(h, mask[:, None])
        return self.ln(h)

    def last(self, X):
        """Representation that scores the next item after the histories X."""
        return self.encode(X)[:, -1]

    def scores(self, h):
        return h @ self.item.weight[1:].T


class GRURec(nn.Module):
    """GRU4Rec-style encoder with the same interface as SASRec (full-softmax CE, tied output embeddings)."""

    def __init__(self, n_items, max_len, d, blocks, heads, dropout):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, d, padding_idx=0)
        nn.init.xavier_normal_(self.item.weight)
        self.gru = nn.GRU(d, d, num_layers=1, batch_first=True)
        self.drop, self.ln, self.max_len = nn.Dropout(dropout), nn.LayerNorm(d), max_len
        self.beta = nn.Parameter(torch.tensor(1.0))

    def encode(self, X):
        """X: (B, T) right-aligned with left padding; the GRU runs over the valid suffix only."""
        B, T = X.shape
        lens = (X > 0).sum(1)
        t = torch.arange(T, device=X.device)
        left = (t[None, :] + (T - lens)[:, None]) % T  # left-aligned position -> right-aligned index
        h = self.drop(self.item(X.gather(1, left)))
        packed = nn.utils.rnn.pack_padded_sequence(h, lens.clamp_min(1).cpu(), batch_first=True,
                                                   enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
        back = (t[None, :] - (T - lens)[:, None]) % T  # right-aligned index -> left-aligned position
        return self.ln(self.drop(out.gather(1, back[..., None].expand(-1, -1, out.shape[-1]))))

    def last(self, X):
        return self.encode(X)[:, -1]

    def scores(self, h):
        return h @ self.item.weight[1:].T


class BERT4Rec(SASRec):
    """BERT4Rec (Sun et al. 2019) as SASRec with bidirectional attention and a [mask] token (id n_items + 1),
    trained with the cloze objective in train.py; at test time [mask] is appended to the history and its
    representation scores the next item. Everything else (blocks, sizes, tied output embeddings) is SASRec's."""
    causal = False

    def __init__(self, n_items, max_len, d, blocks, heads, dropout):
        super().__init__(n_items + 1, max_len, d, blocks, heads, dropout)
        self.mask_id = n_items + 1

    def last(self, X):
        X = torch.cat([X, torch.full_like(X[:, :1], self.mask_id)], 1)[:, -self.max_len:]
        return self.encode(X)[:, -1]

    def scores(self, h):
        return h @ self.item.weight[1:-1].T


ARCHS = {"sasrec": SASRec, "gru": GRURec, "bert": BERT4Rec}


def right_aligned(hists, max_len, dev="cuda"):
    X = np.zeros((len(hists), min(max_len, max(len(h) for h in hists))), dtype=np.int64)
    for b, h in enumerate(hists):
        h = h[-X.shape[1]:]
        X[b, X.shape[1] - len(h):] = h + 1
    return torch.from_numpy(X).to(dev)
