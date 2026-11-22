# Pre-registration: DECOMPOSE (step 2): task vs movement vs slow drift, brain-wide, with valid tests

Registered 2026-11-07, before any real data were analysed with this pipeline. Code: `part6/decompose.py`, SHA-256 prefix 2fb6fdb71ba94a44 (after Amendment 1: 2d4c3078d273b4f3). It reuses the step 1 feature code (`part6/prior_movement.py`) and the Fisher/BH helpers from `part6/prior_map.py`. It has been tested only on mock data. Any later change is recorded below as a dated amendment, and both versions are reported.

## Why
Brain-wide studies report that body movement explains a large share of neural activity in nearly every area (Musall 2019; Stringer 2019; Wang ... Druckmann, Nature Neuroscience 2025). Movement and neurons both drift slowly within a session, through engagement, arousal, satiety and electrode motion, and two slowly drifting signals look related even when they are not. TWIN showed that this breaks the standard test for task encoding (92.5% false positives). Here we ask the same question for movement encoding, and we split neural variance into task, movement and drift across the IBL Brain-Wide Map.

## Data
- All cached IBL Brain-Wide Map sessions, with spike counts per region in four windows:
  - pre-stimulus: stimOn −600 to −100 ms;
  - stimulus: stimOn 0 to 100 ms;
  - choice: first movement −100 to 0 ms;
  - feedback: feedback 0 to 200 ms.
- Trials: those with a choice and a valid event time for the window. A session-window needs at least 300 trials.
- Units: at least 10 with variance; regions with more are randomly subsampled to 150. Counts are square-root transformed and z-scored.
- Movement: the same 9 features as step 1, measured in each window: signed wheel velocity, wheel speed, signed wheel displacement, licks, face motion energy, pupil, and left paw, right paw and nose speed (Lightning Pose, falling back to DLC).
  - A feature is used for a session-window if at least 80% of trials have a value; missing values are filled with the median.
  - Speeds, counts and motion energy are log1p-transformed.
- Task predictors: signed contrast, absolute contrast, choice, feedback, the Bayes block prior, previous choice and previous feedback.
- Drift: a cosine basis over the session with periods of at least 200 trials, which is slower than the block structure (mean block about 60 trials).

## Methods (fixed)
**Encoding model.** Ridge regression from predictors to all neurons. Ridge strength is chosen per neuron by GCV inside each training fold. The score is population held-out R² (neurons weighted equally).

**Unique variance**, in the valid analysis:
- Slow drift is removed from the neurons and from every predictor by whole-session projection, so no slow regressor is extrapolated into held-out blocks.
- Unique task R² = R²(task + movement) − R²(movement).
- Unique movement R² = R²(task + movement) − R²(task).
- These use contiguous 5-fold CV.
- The drift share is the fraction of neural variance in the slow components.

**Standard test (for comparison).** Interleaved 5-fold CV with drift left in. Unique movement R² is compared against 100 trial shuffles of the movement predictors.

**Valid test.** Contiguous CV on drift-removed data, against a linear-shift null (Harris 2020):
- shifts of the movement predictors relative to the neurons, every second shift from −S to +S, with S = 50 trials (or less in short sessions);
- scored on the central trials;
- the same test is run for the task predictors.

**Drift twins.** 3 phase-randomized twins per session-region-window. They keep each neuron's autocorrelation and the cross-neuron correlations, but have no real link to movement or task. Both tests are run on each twin.

## Gate F2 (coverage)
Features must exist for at least 70% of cached sessions, and at least 70% of analysed sessions must have a video feature. If this fails, D2 is reported but not interpreted.

## Hypotheses and criteria
**D1 (validity on drift twins; confirmatory).**
- D1a: the valid test must report movement encoding on at most 7% of twins, with a session-bootstrap upper 95% bound of at most 8%. D1a is a gate for D2.
- D1b: the standard test reports it on at least 20% of twins (lower 95% bound ≥ 0.20). This claims the standard movement-encoding test is invalid under drift.

Results are also reported per window.

**D2 (main; confirmatory).** The share of the movement encoding found by the standard analysis that does not survive the valid analysis, 1 − Σ(valid unique movement R²) / Σ(standard unique movement R²), must have a region-bootstrap 95% CI lower bound above 0.10. We also report this per window, and report the inflation factor (standard / valid) and the part due to drift alone (same contiguous CV, drift left in vs removed).

**D3 (descriptive).** For each window, we report the fraction of regions with significant encoding (Fisher across sessions, BH q = 0.05): movement under the standard test; movement under the valid test; task under the valid test. We also report the mean unique task R², unique movement R² and drift share by brain system.

**D4 (descriptive).** Across regions, we report the Spearman correlation between pre-stimulus valid movement encoding and step 1's posterior probability that a region carries the prior after movement removal.

## What each outcome means
| Outcome | Reading |
|---|---|
| D1b PASS, D2 PASS | Brain-wide movement encoding, as standardly measured, is substantially inflated by slow drift. |
| D1b PASS, D2 FAIL | The standard test is invalid on drift twins, but on real data the valid analysis keeps nearly all movement encoding. |
| D1b FAIL | Movement encoding is robust to drift even with the standard test. |
| D1a FAIL or Gate F2 FAIL | D2 is reported but not interpreted. |

## Will not do
- Change features, windows, transforms, predictors, drift basis, CV scheme, nulls, thresholds or criteria after seeing real results.
- Add video PCs or nonlinear movement terms to change D2. That would be a separate, later, pre-registered step.
- Drop sessions or regions based on their results.
- Claim anything about a specific published paper's pipeline.

## Mock checks (before registration; not evidence)
**Development fix (before registration).** The first version put the drift basis inside the CV model as regressors. Contiguous CV then extrapolated them into held-out blocks, and that version gave nonsense on mock data. It was replaced by whole-session drift removal before any registration or real data.

**Null mock** (20 sessions; movement drifting slowly, no link to neurons): false positives on twins, valid 5.4%, standard 99%; regions with movement encoding, valid 0%, standard 100%.

**Positive-control mock** (10 sessions): the valid test detected the linked region-windows 100% of the time, and the unlinked ones 4%.

## Amendment 1 (2026-11-07, before any decompose features were used; a bug fix only)
The same wheel fix as step 1 Amendment 2: the wheel position is interpolated onto a 2 ms grid in each window (displacement = end − start; velocity = displacement / window length; speed = summed absolute change / window length). Nothing else changes. New SHA-256 prefix: 2d4c3078d273b4f3.

## Amendment 2 (2026-11-08, AFTER seeing the registered results; exploratory; all runs reported)
The registered D2 share was 1.514 (95% CI 1.31–1.75), passing the rule but out of range, because the summed valid unique movement R² was negative. A diagnostic traced this to rare extreme movement trials (valid R² −0.166 in the quarter of session-windows with the largest movement |z|). A fast effect-size path reproduced the registered number (1.525). With robust predictors (signed log for wheel velocity and displacement, then clipping at the session 1st and 99th percentiles; label-free), the share is 0.449 (95% CI 0.423–0.479). Code SHA-256 prefix 7d697e4756d72deb. The registered run remains the confirmatory result; the robust share is post hoc and is reported as such. The same robust predictors were then fixed in advance for every later dataset (MAP, calcium).
