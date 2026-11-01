"""flan-t5-xl item embeddings for TGH, computed as GraphRec's src/generate_embeddings.py does (T5EncoderModel in
float32, texts tokenized to max_length 128 with padding and truncation, last hidden state mean-pooled over the
attention mask), from the item texts of the GRID items split (read with tfrecord_lite instead of TensorFlow).
Writes <grid dir>/<ds>/t5xl.npy, one row per item id. t5_npy2pkl.py turns it into the t5xl.pkl that tgh.py reads.
Usage: python t5_embed.py <model dir> <grid dir> <ds> [device] [batch size]
"""
import glob
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tfrecord_lite as T  # noqa: E402

MODEL, GRID, DS = sys.argv[1:4]
DEVICE = sys.argv[4] if len(sys.argv) > 4 else "cpu"
BATCH = int(sys.argv[5]) if len(sys.argv) > 5 else 8
MAX_LENGTH = 128

items = sorted((ex["id"][0], ex["text"][0].decode("utf-8", errors="replace"))
               for f in sorted(glob.glob(f"{GRID}/{DS}/items/*.tfrecord.gz")) for ex in T.read(f))
assert [i for i, _ in items] == list(range(len(items)))
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = T5EncoderModel.from_pretrained(MODEL, torch_dtype=torch.float32).eval().to(DEVICE)
out = np.zeros((len(items), model.config.d_model), dtype=np.float32)
t0 = time.time()
for b in range(0, len(items), BATCH):
    texts = [t for _, t in items[b:b + BATCH]]
    enc = tokenizer(texts, max_length=MAX_LENGTH, padding="max_length", truncation=True, add_special_tokens=True,
                    return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        h = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state
    mask = enc.attention_mask.unsqueeze(-1).float()
    out[b:b + BATCH] = ((h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)).cpu().float().numpy()
    if b // BATCH % 100 == 0:
        print(f"{DS}: {b + len(texts)}/{len(items)} items, {time.time() - t0:.0f}s", flush=True)
np.save(f"{GRID}/{DS}/t5xl.npy", out)
print(f"{DS}: wrote {out.shape} in {time.time() - t0:.0f}s", flush=True)
