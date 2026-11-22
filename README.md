# Slow drift distorts brain-wide neural encoding maps in both directions

Code for the paper by Arjun Bhupatiraju. It tests whether brain-wide encoding maps are distorted by slow within-session drift. The analyses compare a standard test (interleaved cross-validation, trial-shuffle null) with a drift-valid test (drift removal, contiguous cross-validation, linear-shift null), using drift twins and planted codes of known size.

All data are public and are **not** stored here. The scripts download or stream them.

| Dataset | Source |
|---|---|
| IBL Brain-Wide Map | ONE / openalyx; session list in `manifests/bwm_2023_12_release.csv` |
| MAP (Chen et al. 2024; Wang et al.) | DANDI 000363, version 0.231012.2129 |
| Stringer et al. 2019 calcium imaging | figshare 6163622 |
| Steinmetz et al. 2019 | figshare 9598406 |
| Zatka-Haas et al. 2021 inactivation | figshare 13008038 |

## Install

```
python3 -m pip install -r requirements.txt
```

Tested with Python 3.13 on macOS (8 GB RAM; use 1 to 4 workers).

## What produces what

| Result in the paper | Script | Pre-registration |
|---|---|---|
| IBL drift twins, standard vs valid test, share lost (D1, D2) | `analysis/part6/decompose.py` | `decompose` |
| Other-mouse null and power (R1, R2) | `analysis/part7/robustness.py` | `robustness` |
| MAP reanalysis: false positives, share lost, area ranking, smoothness, Kilosort drift (MA1 to MA3) | `analysis/part7/map_fetch.py`, `analysis/part7/map_decompose.py` | `map-reanalysis` |
| MAP richer movement features (RF1, RF2) | `analysis/part7/map_rich.py` | `map-rich` |
| Calcium imaging (CA1, CA2) | `analysis/part7/calcium_fetch.py`, `analysis/part7/calcium_decompose.py` | `calcium` |
| Calibration, drift law, bias-corrected share (C1 to C4) | `analysis/part7/calibrate.py` | `calibrate`, `calibrate-c4` |
| Block-prior map and prevalence (G1, H1 to H4) | `analysis/part6/prior_map.py` | `prior-map` |
| Prior after movement removal (M1) | `analysis/part6/prior_movement.py` | `prior-movement` |
| Behaviour link (B1, P1, P2) | `analysis/part7/prior_choice.py`, `analysis/part7/prior_choice_pooled.py` and `analysis/part6/prior_followup.py behavior` | `prior-behavior`, `robustness`, `prior-choice-pooled` |
| Map reliability across 11 task variables | `analysis/part7/mapquality.py` | `mapquality` |
| Replication prediction | `analysis/part6/atlas.py`, `analysis/part6/replication.py` | `replication`, `replication-confirm`, `twin-v2` |
| Shared prior state (T1 to T6) | `analysis/part7/shared_prior.py` | `shared-prior` |
| Validation against inactivation (A1, A2) | `analysis/part8/causal.py` | `causal-validation` |
| Figures 1 to 5 | `analysis/paper_figs/final_figures.py` | none (figures) |

Each script has usage notes in its header. Most have a `mock` or `test` command that runs on synthetic data first.

## Pre-registration

Every confirmatory analysis was written down before it was run, with a SHA-256 fingerprint of its code and dated amendments. The documents are in `preregistration/`, and `preregistration/INDEX.md` lists them with the date each was written.

## Citation

Bhupatiraju, A. Slow drift distorts brain-wide neural encoding maps in both directions (2026).

## Licence

MIT (see `LICENSE`).
