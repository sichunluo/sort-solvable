# Shortcut-Solvable or Sort-Solvable? Timestamp Ties and Unequal Data Access in Sequential-Recommendation Benchmarks

Code and data for "Shortcut-Solvable or Sort-Solvable? Timestamp Ties and Unequal Data Access in Sequential-Recommendation Benchmarks".

## Structure

| Path | Contents |
|---|---|
| `code/seqrec/` | All scripts: data preparation, baselines, SASRec/GRU4Rec/PoE training, evaluation, analyses, tables, figures |
| `data/seqrec/timestamps/<Name>_timestamps.tsv.gz` | Reconstructed timestamps of the S3-Rec sequences, one row per event in file order |
| `data/seqrec/processed/` | Every Amazon split used in the paper (`.npz`, format below) |
| `data/seqrec/bsarec_data/<Name>_shuf.txt` | The S3-Rec files with same-day ties in random order, in the S3-Rec format |
| `data/seqrec/results/` | `audit.jsonl` (tie audit of the sources and copies) and the JSON summaries behind every number of the paper |
| `paper_icde/tables/` | The paper's tables, against which `artifact/verify_tables.sh` checks regenerated ones |
| `artifact/sha256_inputs.txt` | SHA-256 of the pinned raw inputs |

Not included: the raw data (download them, see below), the MovieLens splits (the MovieLens license does not allow
redistribution; `prep*.py` rebuild them from `ml-1m.zip` / `ml-20m.zip` in minutes), model checkpoints, and the
per-user rank files of the trained models (about 530 MB), which the training scripts below regenerate.

## Reconstructed timestamps

`export_timestamps.py <raw dir> <out dir>` rebuilds the three S3-Rec sequence files from the Amazon 2014 5-core review
files (user k = k-th reviewer in file order, items numbered by first appearance, stable sort on the day-resolution
review time), asserts that all 77,373 published sequences are reproduced, and writes

```
user  position  item  asin  unixReviewTime
```

for every event, in the order of `<Name>.txt`. Same-day events share a timestamp; in all 271,100 consecutive
same-day pairs the later item has the larger ASIN.

## Split format

Every split is a NumPy `.npz` archive (no pickled objects; load with `allow_pickle=False`):

| Key | Meaning |
|---|---|
| `users`, `items` | training events, grouped by user in sequence order (item ids as in the S3-Rec files) |
| `ho_users`, `ho_items` | test targets (one row per target; set-valued protocols have several rows per user) |
| `catalog` | items seen in training |
| `val_users`, `val_items`, `inner_mask`, `times`, `cutoffs` | GTS only: validation targets, events before the validation cutoff, timestamps, the two cutoffs |
| `cand_catalog` | leaky GTS folds only: the candidate items (those seen before the test cutoff) |
| `ho_gap` (`*_meta.npz`) | per LOO test user, seconds between the target and the previous history event (0 = same day) |

| Suffix | Protocol |
|---|---|
| (none) | leave-one-out (LOO), the eSASRec benchmark split; row counts match the published ones |
| `_shuf`, `_shuf1`-`_shuf4` | LOO with ties broken at random before splitting (five permutations) |
| `_fr`, `_fr1`, `_fr2` | LOO, original targets, tied training events at random (three permutations) |
| `_fv` | LOO, original targets, tied training events in reverse order |
| `_day` | LOO with each user's whole last day as an unordered target set, earlier ties at random |
| `_t` | global temporal split (GTS), cutoffs at the 80th/90th percentile of event times |
| `_ts`, `_tsx` | GTS with all ties / only the training ties at random (test targets unchanged in `_tsx`) |
| `_tset` | GTS with every eligible event of the target's day as a target |
| `_tl0`, `_tl1` | leaky GTS: the two test folds, each trained with the other fold's post-cutoff events |
| `_tq`, `_tqx` | GTS with cutoffs at the 70th/80th percentile, original / random training ties |
| `_tp`, `_tpx` | Beauty only: GTS with 5-core filtering on pre-cutoff events, original / random training ties |

The de-tied drop-in replacements for the ID-ordered LOO benchmarks are `<ds>_day.npz`, `<ds>_fr.npz` and
`bsarec_data/<Name>_shuf.txt`.

## Inputs

| Input | Source | Used for |
|---|---|---|
| S3-Rec Beauty / Sports / Toys sequences | github.com/RUCAIBox/CIKM2020-S3Rec (`data/*.txt`, commit 2a81540) | LOO benchmarks |
| Amazon 2014 5-core review files | snap.stanford.edu/data/amazon/productGraph (`reviews_<cat>_5.json.gz`) | timestamps, GTS |
| Amazon 2014 full Beauty reviews | same (`reviews_Beauty.json.gz`) | SASRec-script audit, pre-cutoff 5-core GTS |
| MovieLens-1M / 20M | grouplens.org (`ml-1m.zip`, `ml-20m.zip`) | LOO, GTS |
| Amazon 2018 / 2023, code-base copies | URLs in `code/seqrec/run_audit.sh` | audit only |

Place them as `data/seqrec/raw/{s3rec,amazon5,ml}/` and compare against `artifact/sha256_inputs.txt`.

Third-party code is not redistributed; the runs used these commits:

| Code base | Repository | Commit |
|---|---|---|
| eSASRec benchmark (RecTools 0.13.0) | github.com/blondered/transformer_benchmark | 2b039927ceb4c7654131d7ac7c43ea124b49d240 |
| Frozen probe configurations | github.com/spotify-research/sequential-capacity-probes | 705710f78abd613cd8ee139f7435d813eef2f8ab |
| BSARec code base (BSARec, DuoRec, SASRec, FMLP-Rec, GRU4Rec) | github.com/yehjin-shin/BSARec | c80bdc0f1a889b78fe9a27e2058389cc84caea84 |
| TGH | github.com/haoyuhan1/GraphRec | 2db502a1c7b6eb31452ace1cdf40953cd2a0a05e |

TGH also needs the `google/flan-t5-xl` checkpoint for its item embeddings.

## Environment

Python 3.9, PyTorch 2.x (CUDA for training; the training-free baselines and all analyses also run on CPU), numpy,
pandas, scipy, matplotlib, and LightGBM for the LTR re-ranker. One 48 GB GPU suffices: the LOO and GTS SASRec runs
take 2-15 minutes each, ML-20M about 7 hours per stage. The Python scripts read `SEQREC_DATA` and `SEQREC_RESULTS`
(default `~/seqrec/data` and `~/seqrec/results`); the shell scripts assume `~/seqrec/{code,data,results,logs}`.

## Pipeline

| Step | Script | Output |
|---|---|---|
| LOO splits (eSASRec protocol, row counts checked) | `prep.py <raw> <processed>` | `<ds>.npz` |
| Timestamp reconstruction, tie meta, shuffled-tie LOO | `prep_ties.py`, `export_timestamps.py` | `<ds>_meta.npz`, `<ds>_shuf[1-4].npz`, `timestamps/` |
| GTS, leaky folds, shuffled variants, earlier cutoffs | `prep_time.py` | `<ds>_t`, `_tl0/1`, `_ts`, `_tsx`, `_tq`, `_tqx` |
| Fixed-target, day-level and set-target variants | `prep_fixed.py` | `<ds>_fr[1,2]`, `_fv`, `_day`, `_tset`, `ml20m_shuf` |
| Pre-cutoff 5-core GTS (Beauty) | `prep_prospective.py` | `beauty_tp`, `beauty_tpx` |
| PCTM, MC (validation and test) | `run_pctm.py <ds>` | `results/<ds>/experts*` |
| PCTM and MC without the validation events (access-matched to conventional SASRec) | `expert_access.py <ds>` | `results/<ds>/expert_access.json` |
| PCTM-sym, SeqRules, MostPop, RecentPop | `run_baselines.py <ds>` | `results/<ds>/baselines*` |
| SASRec / GRU4Rec / BERT4Rec / joint PoE | `train.py <ds> [--refit 1] [--arch gru\|bert] [--method poe ...]` | `results/<ds>/<tag>_s<seed>*` |
| BERT4Rec: masking and architecture chosen on validation, then five seeds and the order cross-evaluation | `run_bert.sh` (`jobs_bert_tune.txt`, `bert_select.py`, `jobs_bert_main.txt`, `run_bert_extra.sh`), `bert_summary.py <results> <processed> [<tables>]` | `results/bert_config.json`, `results/bert_summary.json`, Table `bert` |
| Order and target cross-evaluation | `eval_cross.py <train_ds> <hist_ds> [<target_ds>]` | `results/<target>/cross_*` |
| Rolling evaluation | `eval_rolling.py <ds>_t` | `results/<ds>_t/rolling*` |
| Refit mechanism, direction, GTS selection, rolling masks | `analysis_extra.py <raw> <processed> <results>` | `results/analysis_extra.json` |
| Tie audit of raw files and copies | `run_audit.sh` / `audit_ties.py` | `results/audit.jsonl` |
| Stacking, seed ensembles, LTR, gate, weight diagnostics | `stacking_test.py`, `ensemble.py`, `ltr_*.py`, `gate.py`, `diag_weights.py` | `results/<ds>/*` |
| eSASRec, conventional and refit | `run_esasrec.py <upstream> <processed> <results> <ds> conv\|refit [seed]`, `run_esasrec_queue.sh` | `results/<ds>/esasrec[_refit]_s<seed>*` |
| BSARec code base on original and shuffled-tie files | `prep_bsarec.py`, `run_bsarec.py`, `run_bsarec_queue.sh`, `bsarec_tied.py` | `results/bsarec.json` |
| TGH on its GRID/P5 splits and de-tied copies | `prep_grid.py`, `tfrecord_lite.py`, `t5_embed.py`, `t5_npy2pkl.py`, `run_tgh.sh`, `tgh_npz.py`, `tgh_tied.py` | `results/tgh.json` |
| Numbers quoted only in the text | `text_numbers.py`, `text_numbers_loo.py`, `text_numbers_gts.py` (`<results> <processed>`), `esasrec_summary.py <results>` | `results/text_numbers*.json`, `results/esasrec_summary.json` |

The exact job lists are `code/seqrec/jobs_*.txt`, executed by `run_queue.sh`; `run_r*.sh` chain the rounds. A
training job is skipped if its result exists, so the lists can be re-run. Every neural condition uses five seeds:
0-4 for our models (1-5 for the analyses that reload saved LOO checkpoints), 32-36 for eSASRec and 42-46 for the
BSARec code base; TGH is deterministic.

## Tables and figures

```
python3 -I code/seqrec/make_tables.py data/seqrec/results paper_icde/tables
python3 -I code/seqrec/bert_summary.py data/seqrec/results data/seqrec/processed paper_icde/tables
python3 -I code/seqrec/make_figs.py data/seqrec/results data/seqrec/processed paper_icde/figs
bash artifact/verify_tables.sh   # regenerates all tables and checks them against paper_icde/tables
```

`make_tables.py` writes every table except Table `bert` (`bert_summary.py`, which reads the
`summary_strongest.json` that `make_tables.py` writes) and the protocol map `protocols.tex`, which contains no
numbers and is written by hand. The scripts read the per-user rank files written by the pipeline (0-based rank of
each test target; NDCG@10 = 1/log2(rank+2) for ranks below 10), the split meta files and the JSON outputs of the
pipeline (training logs, `expert_access.json`, `diag_weights_s*.json`, `ltr_s*.json`, `analysis_extra.json`), and
use seeded bootstraps, so the output is identical. `verify_tables.sh` fails if a regenerated table differs from
the paper's or if a table of the paper is neither generated nor hand-written.

## License

The code is released under the MIT License (`LICENSE`). The splits and timestamps in `data/` are derived from the
Amazon review data of McAuley et al. and the S3-Rec sequence files, and remain subject to the terms of those
releases (research use).
