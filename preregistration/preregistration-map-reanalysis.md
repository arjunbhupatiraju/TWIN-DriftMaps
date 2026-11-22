# Pre-registration: STEP 3, reanalysis of the MAP data (Wang et al.; Chen et al.) with drift-valid tests

Registered 2026-11-08, before any MAP neural data were analysed. Only one session's file structure has been inspected (no analysis). Code:
- `part7/map_fetch.py`, SHA-256 prefix 2f37aa8ed6be59cf (after Amendment 1: 93508cb79ebdfa9d);
- `part7/map_decompose.py`, SHA-256 prefix 9dc950e53fe92e81 (after Amendment 1: ccd49093527abca7).

These reuse `part6/decompose.py` unchanged. They have been tested only on synthetic NWB files built in the MAP layout.

## Target
Wang, Kurgyis, ... Svoboda, Li, Druckmann, Nature Neuroscience (doi 10.1038/s41593-025-02114-x). Their data: DANDI:000363 (174 NWB files, 28 mice, CC0). Their methods describe an 80/20 trial split, no drift control, and no session-level null for the encoding models.

## Data (fixed)
- **Files:** every file in DANDI:000363 version 0.231012.2129, streamed.
- **Trials:** task "audio delay"; no photostimulation, no auto or free water, outcome hit or miss, no early lick. Sessions need at least 200 clean trials.
- **Units:** unit_quality "good" and presence_ratio ≥ 0.9. The population is one probe insertion, labelled by its target area with the hemisphere dropped. Insertions need at least 10 units, capped at 150 (random subsample).
- **Epochs:** pre, 0.5 s before sample onset; sample, sample onset to offset; delay, the last 1.0 s before the go cue; response, go cue to +0.6 s.
- **Neural data:** square-root spike counts, z-scored.
- **Movement:** the paper's own DeepLabCut tracking (side camera; jaw, nose and tongue; likelihood ≥ 0.9).
  - Features: jaw speed, jaw height, nose speed, tongue-visible fraction, tongue speed (0 when not visible), left licks, right licks.
  - A feature is used if at least 80% of trials have a value; missing values are filled with the median.
  - Transform: log1p of speeds, fractions and counts, then clipping at the session 1st and 99th percentiles. This is the step 2 robust transform, fixed in advance.
- **Task predictors:** instruction (±1), outcome (hit ±1), previous instruction, previous outcome.
- **Drift:** a cosine basis over trials with periods of at least 200 trials.

## Methods
Identical to step 2 (decompose.tests): the standard test (interleaved 5-fold CV, drift left in, 100 trial shuffles) and the valid test (contiguous CV on drift-removed data, linear-shift null with shifts up to 50 trials). There are 3 drift twins per insertion-epoch.

Per neuron, unique movement explained variance (EV) is computed three ways: standard EV; valid EV; standard EV on one drift twin, which keeps spatially correlated electrode drift but has no movement link.

## Gate
At least 30 sessions and at least 3 target areas.

## Hypotheses
**MA1 (confirmatory; replicates D1 on their data).**
- MA1a: valid-test twin false positives ≤ 0.07, with session-bootstrap upper bound ≤ 0.08.
- MA1b: standard-test lower bound ≥ 0.20.

**MA2 (confirmatory; replicates the robust D2 on their data).** The share of standard movement encoding that does not survive the valid analysis, 1 − Σ valid / Σ standard unique movement R², must have an insertion-bootstrap lower bound above 0.10. It is also reported by epoch and by area.

**MA3 (confirmatory).** Does slow drift alone make movement-encoding maps look spatially smooth along the probe?
- Smoothness of a per-neuron map = Spearman(value, mean of the 5 nearest units by depth) minus the same statistic with values shuffled across positions (200 shuffles).
- PASS if the mean smoothness of the drift-twin standard EV maps has a session-bootstrap lower bound above 0.

**Descriptive.** Smoothness of the real standard and real valid EV maps, and the twin/real ratio; mean per-neuron EV by area, standard vs valid, and whether the ranking of areas is preserved; the within-probe Spearman between Kilosort drift_metric or cumulative_drift and the EV lost under the valid analysis; the fraction of neurons with positive EV under each analysis.

## Limits, stated in advance
- We use their DeepLabCut markers, not their autoencoder embeddings or end-to-end CNN.
- We use trial epochs, not 20 ms bins.
- Our test asks how much trial-to-trial movement encoding is valid. It does not ask how much within-trial, time-locked variance the video explains.

## Mock check (before registration; not evidence)
On 4 synthetic NWB sessions with spatially structured drift: valid twin false positives 3.1%; drift-twin map smoothness +0.070 (CI +0.026 to +0.114), while real maps with no movement link were near 0. A neighbour-mean correlation is biased negative in small samples, so the smoothness statistic was changed to subtract a position-shuffled baseline. This was done before registration, on mock data only.

## Will not do
- Change trial filters, unit filters, epochs, features, transforms, CV, nulls or criteria after seeing MAP results.
- Drop sessions or areas based on results.

## Amendment 1 (2026-11-08, during the download, before any analysis; speed only)
map_fetch.py can now run as several parallel copies (`--part i --of n`). Each session is written to a temporary file and renamed when complete. map_decompose.py ignores temporary files. Extraction and analysis are unchanged.
