# Pre-registration: FEATURE RICHNESS test on the MAP data (answers "you tested the paper's weaker model")

Registered 2026-11-08, before any rich features were extracted from real data. Code: `part7/map_rich.py`, SHA-256 prefix 46dcfca4f2205bed. It imports the frozen step 3 code unchanged: map_fetch.py (93508cb79ebdfa9d), map_decompose.py (ccd49093527abca7) and part6/decompose.py. It has been tested only on synthetic NWB files.

## Why
The paper's per-trial video embeddings are not public. The step 3 reanalysis used 7 window-averaged DeepLabCut features. The objection: richer features would explain more real variance, so less movement encoding would fail to survive the valid analysis. This test asks whether the step 3 conclusion holds when movement is described much more richly.

## Rich features (fixed)
Per trial and epoch, the epoch window is cut into 10 equal sub-windows:
- jaw x, jaw y, nose x and nose y: mean position in each sub-window;
- jaw speed and nose speed: mean in each sub-window;
- tongue-visible fraction in each sub-window;
- left and right lick counts in each sub-window.

That gives up to 90 features. Inclusion and transforms:
- A feature is kept if at least 80% of trials have a value.
- Positions are clipped at the session 1st and 99th percentiles.
- Speeds, fractions and counts are log1p-transformed and then clipped the same way.

Everything else is identical to step 3: trials, units, insertions, epochs, task predictors, drift basis, CV, nulls and twins. The registered 7 marker features are recomputed on the same populations, so the comparison is paired.

## Hypotheses
**RF1 (confirmatory).** With rich features, the share of standard movement encoding that does not survive the valid analysis, 1 − Σ valid / Σ standard unique movement R², must have an insertion-bootstrap lower bound above 0.10.

**RF2 (confirmatory, validity with rich features).**
- RF2a: valid-test twin false positives ≤ 0.07, with session-bootstrap upper bound ≤ 0.08.
- RF2b: standard-test lower bound ≥ 0.20.

**Descriptive.**
- The difference in share, rich minus markers, with its CI. No directional prediction.
- By epoch and by area.
- RF0: the gain in standard and valid unique movement R² from rich features (descriptive only).

## Reading
- RF1 PASS: the overstatement persists with a time-resolved, roughly 13 times richer movement description.
- RF1 FAIL: richer movement descriptions remove the overstatement. Report that, and state that the step 3 conclusion is specific to coarse features.

## Mock check (before registration; not evidence)
3 synthetic sessions with no real movement link: RF0 standard gain +0.007; rich share 0.85 vs markers 0.99; valid twin false positives 0%; standard 79%.

## Will not do
- Change the sub-window count, features, transforms or criteria after seeing results.
