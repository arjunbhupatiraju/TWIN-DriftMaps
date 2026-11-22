# Pre-registration: PRIOR MAP step 1, is the hidden block prior just movement?

Registered 2026-11-07, before any real data were analysed with this pipeline. Code: `part6/prior_movement.py`, SHA-256 prefix 78e48b3418095477 (after Amendment 1: c204d0287010746c; after Amendment 2: 5852f5d9e5095f93). It imports the decoder, nulls, grid and estimator unchanged from `part6/prior_map.py` (SHA-256 prefix 67b0a273f7d81406). It has been tested only on mock data. Any later change is recorded below as a dated amendment, and both versions are reported.

## Why
PRIOR MAP estimated that about half of brain regions carry the block prior (52%, 95% CI 36 to 68), while only 7% are detected. Mice move in anticipation of the side they expect, and movement drives neurons across the whole brain (Musall 2019; Stringer 2019; Wang ... Druckmann, Nature Neuroscience 2025). The hidden prior could therefore be movement. This is the first objection a reviewer will raise, and it has to be answered before the result can be claimed.

## Data
- The same IBL Brain-Wide Map cache, trials, window (stimOn −600 to −100 ms), unit limits and 90-trial burn-in as PRIOR MAP.
- Movement features per trial, as the mean over the same window. All come from public IBL datasets:
  - signed wheel velocity, wheel speed, and signed wheel displacement;
  - lick count;
  - left-camera face motion energy and body-camera motion energy;
  - pupil diameter;
  - left-camera pose tracking (Lightning Pose, or DLC where Lightning Pose is absent): left paw, right paw, nose and tongue speed.
- A feature is used for a session only if at least 80% of its trials have a value. Missing values are filled with the session median.
- Speeds, counts and motion energies are log1p-transformed. Every feature is z-scored, and an intercept is added.

## Methods (fixed)
**Movement removal.** The least-squares projection on the movement features (with intercept) is removed from every neuron. The same removal is applied to every target:
- the real Bayes prior;
- every pseudo-session prior;
- every planted code.

Pseudo-sessions are independent of both the neurons and the movement, so the valid null stays exact.

**Everything else is as in PRIOR MAP Amendment 1:**
- the ridge decoder, with 5 contiguous folds and GCV;
- 1,000 pseudo-session nulls;
- 48 planted codes per grid value, in 20% of neurons; codes are planted before movement removal;
- the same estimator, using the region-shared code size, a kernel bandwidth of 0.35, EM, and a 500-resample region bootstrap.

**Drift twins.** There are 5 phase-randomized twins per session-region, and the same movement removal is applied to each.

**Interpretation, fixed in advance.** The removal is linear and applies to both sides. What survives is the part of the prior code that movement in this window cannot explain linearly. If the brain's prior causes the anticipatory movement, removing movement also removes some genuine prior signal. The surviving estimate is therefore a lower bound on non-movement prior coding, not an estimate of the total.

## Validity checks (must pass before M1 is interpreted)
**Gate F (coverage).** Two conditions must both hold:
- movement features exist for at least 70% of PRIOR MAP sessions;
- at least 70% of the analysed sessions have at least one video feature.

**G1 (estimator).** The same check as in PRIOR MAP, run on the movement-removed data:
- with no planted code, the estimated fraction with r ≥ 0.04 must be at most 0.10;
- with a code planted in 30% of regions, the estimate must fall between 0.20 and 0.40.

**V1 (test validity).** The false-positive rate of the movement-removed valid test on drift twins must be at most 0.07, and the upper bound of its session-bootstrap 95% CI must be at most 0.08.

## Hypotheses and criteria
**M1 (confirmatory).** After movement removal, the power-corrected fraction of regions carrying the prior (r ≥ 0.04) must have a 95% CI lower bound above 0.071, the fraction detected in the registered PRIOR MAP run.
- PASS means the prior is still hidden in more regions than were detected, even after movement is removed. The hidden prior is not explained by movement.
- FAIL means the detection gap cannot be separated from movement.

The fractions at r ≥ 0.02 and r ≥ 0.1, and the H3-style verdict (BRAIN-WIDE, LOCALIZED or INCONCLUSIVE), are reported alongside.

**Retained (descriptive).** On the same session-regions, using the PRIOR MAP Amendment 1 rows, we report:
- the power-corrected fraction before and after movement removal;
- their ratio, with a region-bootstrap 95% CI.

**M2 (descriptive).** On the same trials and nulls, we report:
- the single-session and region-level detection before and after removal;
- how many regions detected before are still detected after.

**M3 (descriptive).** We report:
- how often movement alone predicts the prior (same decoder, same pseudo-session null);
- the Spearman correlation across sessions between movement-only prior information and the drop in neural prior information after removal.

## What each outcome means
| Outcome | Reading |
|---|---|
| M1 PASS, retained near 1 | The hidden prior is internal, not movement. This is the strongest version of the PRIOR MAP claim. |
| M1 PASS, retained well below 1 | Movement explains part of the hidden prior, but a non-movement prior remains in more regions than are detected. Report both parts. |
| M1 FAIL | The hidden prior cannot be separated from movement with these data. PRIOR MAP is reframed as "prior-or-movement", and step 2 (task vs movement vs drift decomposition) becomes the main analysis. |
| Gate F, G1 or V1 FAIL | M1 is reported but not interpreted. |

## Will not do
- Change the features, window, transforms, decoder, nulls, grid or criteria after seeing real results.
- Add nonlinear or lagged movement terms, or video PCs, to rescue or sink M1. Any such analysis is a separate, later, pre-registered step.
- Drop sessions or regions based on their results.

## Commands
- `python3 part6/prior_movement.py features`
- `python3 part6/prior_movement.py wheel` (Amendment 2)
- `python3 part6/prior_movement.py run`
- `python3 part6/prior_movement.py test`

## Mock check (before registration; not evidence)
On mock data, the pipeline behaved as expected in both directions:
- movement built to correlate with the prior (r about 0.6) removed most of the planted block code (retained 0.06);
- movement unrelated to the prior left it intact (retained 1.11; detection 1/8 regions before and after).

## Amendment 1 (2026-11-07, before any neural analysis; a bug fix only)
A 3-session test of the feature download (no neural data analysed, no run or test command executed) showed that pose-tracking speeds were never extracted. There were two causes. First, pandas returned read-only arrays, so masking low-confidence frames failed silently. Second, camera times and pose rows can differ in length by a few frames. Both are fixed: the arrays are copied and truncated to the shared length.

The pose source order is now Lightning Pose first, with DLC as the fallback. This is because Lightning Pose tracks the left paw more reliably (97% vs 72% of frames above likelihood 0.9 in the test session). Features, transforms, criteria and the analysis are unchanged. The body-camera files do not exist for the test session, so body_motion will likely be absent for most sessions, which the coverage gate already allows.

New SHA-256 prefix: c204d0287010746c. The 3 test feature files are deleted and re-extracted with the fixed code.

## Amendment 2 (2026-11-07, after the feature download, before any neural analysis; a bug fix only)
The full feature download (459 sessions) showed that signed wheel velocity and wheel speed were missing for some sessions; the last session printed only wheel_displacement among the wheel features. The cause: the IBL wheel trace is sampled only when the wheel moves. A trial where the mouse held the wheel still has no samples in the window, so it was recorded as missing rather than as zero, and the feature then failed the 80% rule.

The fix interpolates the wheel position onto a 2 ms grid over the window, as ibllib does:
- displacement = end − start;
- velocity = displacement / 0.5 s;
- speed = summed absolute change / 0.5 s.

A new `wheel` command re-downloads only the small wheel files and recomputes these three features for every session. Nothing else changes: the other features, the transforms, the criteria and the analysis. No neural data have been analysed.

New SHA-256 prefix: 5852f5d9e5095f93.

## Amendment 3 (reporting only)
On the first `test` attempt, the coverage count opened all 459 feature files without closing them and crashed ("Too many open files") after the header line and before any result was printed. Each file is now closed after reading. The analysis is unchanged. New SHA-256 prefix: 5b824982511d7c55.
