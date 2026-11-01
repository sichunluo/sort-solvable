"""eSASRec with the pinned upstream code of the probe study (blondered/transformer_benchmark @2b03992, RecTools 0.13.0)
and its frozen per-dataset configuration (configs/table1.json of spotify-research/sequential-capacity-probes), on our
original and order-controlled LOO splits.

conv:  the upstream protocol: each user's last training event is the validation target (get_val_mask_func_all),
       early stopping on Recall@10 with patience 50, best checkpoint evaluated.
refit: retrain from scratch on the complete training split (no validation mask) for the number of optimizer steps
       at which the conv run of the same split and seed stopped (read from <out>/<ds>/esasrec_s<seed>.json).

Splits are read from our <processed>/<ds>.npz (the eSASRec LOO splits; datetime = position in the sequence, as the
upstream S3-Rec builder does), the model is built with RecTools' model_from_params exactly as upstream
validate_model_on_holdout does, and the per-row rank of every holdout target (10 = not in the top 10) is saved in
our results format (<out>/<ds>/esasrec[_refit]_s<seed>_ranks.npz, key "alone").
Usage: python run_esasrec.py <upstream dir> <processed dir> <out dir> <ds> <conv|refit> [seed]
"""
import json
import os
import sys
import time

UP, PROC, OUT, DS, MODE = sys.argv[1:6]
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 32  # upstream default seed
OUT, PROC, UP = os.path.abspath(OUT), os.path.abspath(PROC), os.path.abspath(UP)
os.chdir(UP)  # upstream resolves src/val_schemes and its logs relative to the checkout
sys.path.insert(0, UP)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sys.modules["cupy"] = None  # the shared env's cupy fails on import; RecTools only needs it for PureSVD
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from rectools import Columns  # noqa: E402
from rectools.dataset import Dataset  # noqa: E402
from rectools.metrics import calc_metrics  # noqa: E402
from rectools.models import model_from_params  # noqa: E402

import src.models.transformers.trainer as upstream_trainer  # noqa: E402
from pytorch_lightning.loggers import CSVLogger  # noqa: E402
from src.evaluation.common import get_val_scheme_arguments  # noqa: E402
from src.utils import setup_deterministic  # noqa: E402

BASE = DS.split("_")[0]
# frozen choices of the probe study (configs/table1.json) on top of the upstream configs/paper/loo/*.yaml entries
FROZEN = {"beauty": dict(n_factors=64, session_max_len=50, n_blocks=1, n_heads=1, dropout_rate=0.2, ff=4),
          "sports": dict(n_factors=64, session_max_len=50, n_blocks=1, n_heads=1, dropout_rate=0.2, ff=2),
          "toys": dict(n_factors=64, session_max_len=50, n_blocks=1, n_heads=1, dropout_rate=0.2, ff=1),
          "ml1m": dict(n_factors=64, session_max_len=200, n_blocks=2, n_heads=1, dropout_rate=0.1, ff=None)}[BASE]
TRAINER = "get_trainer" if BASE == "ml1m" else "get_trainer_200_epochs"


def params():
    p = {"cls": "SASRecModel", "get_trainer_func": f"src.models.transformers.trainer.{TRAINER}", "verbose": 1,
         "deterministic": True, "get_val_mask_func": "src.models.transformers.trainer.get_val_mask_func_all",
         "transformer_layers_type": "src.models.transformers.transformer_layers.ligr.LiGRLayers",
         "use_key_padding_mask": True, "loss": "sampled_softmax", "n_negatives": 256, "lr": 0.001, "batch_size": 128,
         **{k: v for k, v in FROZEN.items() if k != "ff"}}
    if FROZEN["ff"] is not None:  # the ML-1M eSASRec entry keeps the upstream default multiplier
        p["transformer_layers_kwargs.ff_factors_multiplier"] = FROZEN["ff"]
    if MODE == "refit":
        conv = json.load(open(f"{OUT}/{DS}/esasrec_s{SEED}.json"))
        os.environ["ESASREC_REFIT_STEPS"] = str(conv["best_global_step"])
        os.environ["ESASREC_RUN"] = f"{DS}_refit_s{SEED}"
        p["get_val_mask_func"] = None
        p["get_trainer_func"] = "esasrec_refit.get_trainer_fixed_steps"
    return p


def splits():
    z = np.load(f"{PROC}/{DS}.npz")
    cat, users = z["catalog"], z["users"]
    starts = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1]
    pos = np.arange(len(users)) - np.repeat(starts, np.diff(np.r_[starts, len(users)])) + 1
    train = pd.DataFrame({Columns.User: users, Columns.Item: cat[z["items"]], Columns.Weight: 1,
                          Columns.Datetime: pos})
    holdout = pd.DataFrame({Columns.User: z["ho_users"], Columns.Item: cat[z["ho_items"]], Columns.Weight: 1})
    return train, holdout


# one log/checkpoint directory per run: the upstream module-level logger would let parallel runs share a version dir
upstream_trainer.LOGGER = CSVLogger("rectools_logs", name=f"{DS}_{MODE}_s{SEED}")
setup_deterministic(SEED)
train, holdout = splits()
model = model_from_params(params())
dataset = Dataset.construct(train)
t0 = time.time()
model.fit(dataset)
fit_time = time.time() - t0
reco = model.recommend(users=holdout[Columns.User].unique(), dataset=dataset, k=10, filter_viewed=True,
                       on_unsupported_targets="warn")
metrics, k = (lambda a: (a["METRICS"], a["K"]))(get_val_scheme_arguments("leave_one_out", ["METRICS", "K"]))
official = {m: float(v) for m, v in calc_metrics(metrics=metrics, reco=reco, interactions=holdout,
                                                 prev_interactions=train,
                                                 catalog=train[Columns.Item].unique()).items()}
rank = dict(zip(zip(reco[Columns.User], reco[Columns.Item]), reco[Columns.Rank] - 1))
ranks = np.asarray([rank.get((u, i), 10) for u, i in zip(holdout[Columns.User], holdout[Columns.Item])])
tag = "esasrec" if MODE == "conv" else "esasrec_refit"
os.makedirs(f"{OUT}/{DS}", exist_ok=True)
np.savez(f"{OUT}/{DS}/{tag}_s{SEED}_ranks.npz", alone=ranks)
info = dict(ds=DS, mode=MODE, seed=SEED, params=model.get_params(simple_types=True), official=official,
            fit_time=round(fit_time), torch=torch.__version__)
if MODE == "conv":  # where early stopping selected the model, for the refit
    ckpt = next(c for c in model.fit_trainer.callbacks if type(c).__name__ == "ModelCheckpoint")
    state = torch.load(ckpt.best_model_path, map_location="cpu", weights_only=False)
    info.update(best_epoch=int(state["epoch"]), best_global_step=int(state["global_step"]),
                best_recall=float(ckpt.best_model_score))
json.dump(info, open(f"{OUT}/{DS}/{tag}_s{SEED}.json", "w"), indent=1, default=str)
print("RESULT", DS, tag, SEED, official, {k: info.get(k) for k in ("best_epoch", "best_global_step")}, flush=True)
