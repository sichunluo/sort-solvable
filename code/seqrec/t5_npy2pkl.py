"""Write the t5xl.pkl / t5xl.pt that GraphRec's tgh.py reads (the format of its generate_embeddings.py: a list of
{"item_id", "embedding"} records and the [n_items, d] tensor) from the t5xl.npy of t5_embed.py.
Usage: python t5_npy2pkl.py <grid dir>/<ds>
"""
import os
import pickle
import sys

import numpy as np
import torch

d = sys.argv[1]
emb = np.load(os.path.join(d, "t5xl.npy"))
torch.save(torch.from_numpy(emb), os.path.join(d, "t5xl.pt"))
with open(os.path.join(d, "t5xl.pkl"), "wb") as f:
    pickle.dump([{"item_id": i, "embedding": emb[i].tolist()} for i in range(len(emb))], f)
print("wrote", d, emb.shape)
