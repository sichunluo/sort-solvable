"""Run one model of the BSARec code base (yehjin-shin/BSARec, src/ unchanged) and keep its test ranking.

The repository logs only aggregate metrics; this wrapper additionally saves, for the final test evaluation, each
user's target and top-20 list (the repository's own full-ranking predictions, history masked) to
<output_dir>/<train_name>_test.npz, so that tied and non-tied targets can be scored separately. All arguments are
passed to the repository's main.py.
Usage: python run_bsarec.py <BSARec src dir> [main.py arguments]
"""
import os
import runpy
import sys

import numpy as np

SRC = os.path.abspath(sys.argv[1])
sys.argv = [os.path.join(SRC, "main.py")] + sys.argv[2:]
sys.path.insert(0, SRC)
os.chdir(SRC)

import trainers  # noqa: E402

_score = trainers.Trainer.get_full_sort_score


def get_full_sort_score(self, epoch, answers, pred_list):
    if self.args.train_matrix is self.args.test_rating_matrix:  # the test call (valid uses valid_rating_matrix)
        np.savez(os.path.join(self.args.output_dir, self.args.train_name + "_test.npz"),
                 answers=np.asarray(answers), top20=np.asarray(pred_list))
    return _score(self, epoch, answers, pred_list)


trainers.Trainer.get_full_sort_score = get_full_sort_score
runpy.run_path(sys.argv[0], run_name="__main__")
