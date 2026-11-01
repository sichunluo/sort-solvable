"""Trainer for the eSASRec refit (run_esasrec.py): no validation, no early stopping, a fixed number of optimizer
steps (ESASREC_REFIT_STEPS, the step at which the early-stopped run of the same split and seed was selected)."""
import os

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger


def get_trainer_fixed_steps() -> Trainer:
    return Trainer(max_steps=int(os.environ["ESASREC_REFIT_STEPS"]), max_epochs=-1, deterministic=True,
                   enable_progress_bar=False, enable_checkpointing=False,
                   logger=CSVLogger("rectools_logs", name=os.environ.get("ESASREC_RUN", "refit")),
                   accelerator="gpu", devices=[0], callbacks=[])
