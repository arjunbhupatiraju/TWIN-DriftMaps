# Pre-registration: STEP 4, do the movement-encoding results hold for calcium imaging?

Registered 2026-11-08, before any calcium data were downloaded or analysed. Code:
- `part7/calcium_fetch.py`, SHA-256 prefix 8dd89778a4276f82;
- `part7/calcium_decompose.py`, SHA-256 prefix b3e1f9985fabad7e (after Amendment 1: 61ca2f6d39a149a6).

These reuse `part6/decompose.py` (cv_r2, detrend) and `part6/prior_map.py` (_p) unchanged. They have been tested only on mock data.

## Target
Stringer, Pachitariu et al., "Spontaneous behaviors drive multidimensional, brainwide activity", Science 2019. Two-photon recordings of about 10,000 neurons in mouse visual cortex during spontaneous behaviour, with face-video motion SVDs, running speed and pupil area (figshare 6163622, public). That paper reported that facial behaviour predicts a large share of spontaneous activity. We ask how much of the behaviour-explained variance survives the drift-valid analysis, and whether the standard test is valid on imaging data.

## Data (fixed)
- **Recordings:** every `spont_*.mat` file in figshare 6163622.
- **Neural data:** deconvolved activity (Fsp), summed in bins of 4 frames, then square root, then z-scored.
- **Populations:** neurons are ordered by imaging plane, then x, then y, and cut into consecutive groups of 150 neighbouring neurons. Up to 40 groups per recording are chosen at random (seeded).
- **Behaviour:** the first 20 face motion SVD components, running speed and pupil area (if present). Each is bin-averaged and clipped at the recording's 1st and 99th percentiles. A feature needs at least 80% finite values; missing values are filled with the median.
- **Drift:** a cosine basis over the recording with periods of at least 200 bins.
- **No task predictors** (the behaviour is spontaneous).

## Methods
- **Standard test:** interleaved 5-fold CV, drift left in. The statistic is population R² from behaviour. The null is 100 shuffles of the behaviour time points.
- **Valid test:** contiguous 5-fold CV on drift-removed data. The null is linear shifts of behaviour relative to the neurons: every 16th shift from −800 to +800 bins, at most T/6, and never closer than 16 bins. It is scored on the central bins.
- **Drift twins:** 3 phase-randomized twins per population (no real link to behaviour).
- **Ridge/GCV:** as in step 2.

## Gate
At least 5 recordings.

## Hypotheses
**CA1 (confirmatory).**
- CA1a: valid-test twin false positives ≤ 0.07, with a recording-bootstrap upper bound ≤ 0.08. This is a gate for CA2.
- CA1b: standard-test lower bound ≥ 0.20.

**CA2 (confirmatory).** The share of the behaviour-explained variance found by the standard analysis that does not survive the valid analysis, 1 − Σ R²valid / Σ R²standard, must have a recording-bootstrap lower bound above 0.10.

**Descriptive.**
- R² under the standard analysis, under contiguous CV without drift removal, and under the valid analysis.
- The slow-drift share of activity.
- The fraction of populations with significant behaviour encoding under each test.
- Per-recording values.

## Design choices made on mock data before registration (not evidence)
The first draft had a 400-bin shift range and drift periods of at least 1,000 bins. On mock recordings with very slow behaviour, the valid test then gave 11.8% false positives on twins.

Four settings were compared on the mock:

| Setting | Twin false positives (valid test) |
|---|---|
| 400 bins, step 8, periods ≥ 1,000 | 11.8% |
| 400 bins, step 8, periods ≥ 200 | 4.9% |
| 800 bins, step 16, periods ≥ 200 | 2.8% |
| 800 bins, step 16, periods ≥ 100 | 6.2% |

800 / 16 / 200 was chosen: it gave 2.8% false positives, detected every population in all 3 mock recordings with a real link, and flagged none in the 3 without one.

The full mock run with these settings gave 5.6% valid false positives (upper bound 0.10, from only 108 twins, so too few to pass the criterion; the real run has about 10 times more) and 100% standard false positives. The mock behaviour is unrealistically slow.

## Will not do
- Change bins, populations, features, transforms, drift basis, shift range, CV, nulls or criteria after seeing real results.
- Drop recordings or populations based on results.

## Amendment 1 (2026-11-08, before any results; memory only)
The first run froze an 8 GB laptop, because each worker loaded a full recording in double precision. The deconvolved activity is now loaded in single precision, binned (sums kept in double precision), and the raw frames are freed before testing. The run uses `--workers 1` on 8 GB of RAM. The analysis is unchanged.
