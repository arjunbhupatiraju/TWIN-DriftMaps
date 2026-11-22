# Pre-registration: SHARED PRIOR. Is the block prior carried by one shared, slow, brain-wide state?

Registered 2026-11-10, before any real data were analysed with this script. Code: `part7/shared_prior.py`, SHA-256 prefix e06c152a832aa612 (Amendment 1: b9d612e28b1fb63c; Amendment 2: 31fe654f619fd477). It has been tested only on synthetic data. Background: Findling et al. 2025 detect the prior in 24–30% of regions with pseudo-session decoding, call this an underestimate, and do not test for a shared state.

## Hypothesis
The prior is implemented largely as a single shared, slow state across the brain, which standard analyses treat as drift.

## Data and preprocessing (identical to step 1, part6/prior_movement.py)
- **Sessions:** IBL sessions with step 1 movement features.
- **Window:** the pre-stimulus window (−600 to −100 ms).
- **Trials:** the first 90 trials are burned, and at least 150 valid trials are required.
- **Units:** regions with at least 10 units, at most 150 kept (random), square-root counts.
- **Movement removal:** 11 movement features, including pupil, are regressed out of the neurons, the Bayes prior and every one of 1,000 pseudo-session priors, so the null stays exact.
- **Inclusion:** sessions need at least 3 such regions.

## Shared state S (unsupervised)
- Take the top 3 principal components of each region's movement-removed, z-scored activity.
- S is the first principal component of all regions' components together.
- The prior is never used to build S.

## Tests

| Test | Statistic | Null | PASS |
|---|---|---|---|
| T1 | mean over sessions of \|corr(S, prior)\| | the mean of the j-th pseudo-session statistic over sessions (1,000) | one-sided p < 0.05 |
| T2 | share of each region's prior information that is S: 1 − Σ info after projecting S out / Σ info before; info = contiguous-CV ridge decoding r (as in PRIOR MAP) minus the mean of 200 pseudo-session decodings, summed over all region-sessions with no selection | projecting out a random direction of the same shared space (10 per session) | session-bootstrap (500) lower bound of share(S) − share(random) > 0. Claim "mostly" only if the lower bound of share(S) > 0.5 |
| T3 | T1 restricted to sessions where pupil was among the removed features | as T1 | p < 0.05 |
| T4 | partial correlation of S (signed by its correlation with the prior) with choice on trials with \|contrast\| ≤ 0.0625 (at least 40 per session), controlling for signed contrast, Bayes prior, choice lags 1–2 and rewarded-choice lags 1–2; pooled as T1 | S signed by its correlation with each pseudo prior | one-sided p < 0.05 |

## Split (fixed now)
- **Assignment:** labs are sorted and permuted with seed 20261110; the first half (floor) is EXPLORE and the rest is CONFIRM.
- **Order:** EXPLORE is run and reported as exploratory. CONFIRM is run once.
- **Verdicts:** confirmatory verdicts come only from CONFIRM.
- **Frozen:** no setting may change between the two halves.

## Interpretation fixed in advance
- **T1 and T2 pass in CONFIRM:** the prior is substantially carried by a shared brain-wide state. "Mostly" is said only if share(S) has a lower bound above 0.5.
- **Adding T3:** the shared state is not pupil-linked arousal or measured movement.
- **Adding T4:** the shared state carries behaviourally relevant prior information beyond the ideal observer and recent history.
- **T1 fails:** no evidence for a shared prior state. Report it: the prior is regionally specific, or the shared component is not prior-related. The current paper stands.
- **Additional caveat:** the shared state may also reflect unmeasured slow behaviour (engagement) that covaries with blocks. Pseudo-sessions control for drift, but not for block-locked behaviour beyond the 11 measured features.

## Synthetic check (not evidence)
16 sessions, 5 regions, a shared drift in both conditions.

| Condition | T1 | T2 | T3 | T4 | Verdict |
|---|---|---|---|---|---|
| No prior state | p = 0.28 | difference lower bound −1.86 | p = 0.28 | p = 0.11 | all FAIL, as intended |
| Shared prior state | p = 0.001 | share 0.96 [0.88, 1.03], random 0.16 | p = 0.001 | partial r +0.14, p = 0.001 | all PASS |

The T2 share is unstable when total prior information is near zero (the no-state mock), which is why the criterion uses the difference against random directions.

## Will not do
- Change the number of PCs, the window, the filters, the null counts, the split or the criteria after seeing EXPLORE.

## EXPLORE result (2026-11-11, exploratory)
151 sessions, 54 mice, 6 labs (churchland, dan, hausser, hofer, mainen, mrsicflogel), 784 region-sessions.
- **Shared state:** variance share 0.23, slow share 0.71.
- **T1:** 0.116 vs null 0.155, p = 1.00, FAIL.
- **T2:** share −0.49 [−1.05, −0.21], random −0.10, difference lower bound −0.78, FAIL.
- **T3:** FAIL, p = 1.00.
- **T4:** partial r −0.004, p = 0.66, FAIL.

The original hypothesis fails in EXPLORE. Unexpected observation: projecting S out *increases* null-corrected prior information.

## Amendment 1 (registered 2026-11-11, after EXPLORE and BEFORE CONFIRM was run)
This amendment adds a new hypothesis, generated from the EXPLORE observation above. T1–T4 and every setting are unchanged.

**T5 hypothesis (masking):** the shared slow state hides the prior from regional decoders, rather than carrying it.
- **Statistic:** share(S) − share(random), the same quantities as T2.
- **PASS:** the session-bootstrap (500) upper 95% bound of the difference is < 0 in CONFIRM.
- **Also reported:** the prior information gained by removing S (−share(S)), with its CI.

**Code change:** one block of evaluate() reports T5. The synthetic check gives T5 FAIL in both mock conditions, so it gives no false positive. The new hash is b9d612e28b1fb63c.

**Interpretation fixed in advance:**
- **T5 passes in CONFIRM:** a brain-wide slow state masks the block prior in individual regions. It raises the pseudo-session null and lowers the null-corrected information, which is one mechanism for why detection undercounts the prior. This is reported as a confirmatory test from an exploration-generated hypothesis.
- **T5 fails:** it is reported as such, and the EXPLORE observation stays exploratory.
- **Confound to discuss either way:** removing any slow component lowers pseudo-session decoding. The random-direction control is drawn from the same shared space, but it is not matched for slowness.

## Amendment 2 (registered 2026-11-11, BEFORE CONFIRM was run; T6 was not computed on EXPLORE before registration)
**T6 hypothesis (hiding law):** the strength of the shared slow state predicts how much prior information it hides. This is the counterpart of the drift law for overstatement, so one quantity would predict errors in both directions.
- **Per session, the drift strength** = var_S × slow_share_S, meaning the variance share of S times its slow share.
- **Per session, the excess gain** = Σ over regions (info_minus_S − info_full) − Σ (info_minus_rand − info_full).
- **Statistic:** the Spearman correlation across CONFIRM sessions between drift strength and excess gain.
- **Null:** one-sided permutation of drift strength (2,000).
- **PASS:** p < 0.05.

**Code change:** one block of evaluate(). The synthetic check gives T6 FAIL in both mock conditions (p = 0.84 and 0.97), so it gives no false positive. The new hash is 31fe654f619fd477.

**Interpretation fixed in advance:**
- **T5 and T6 pass:** the shared slow state hides the prior, and in proportion to its strength. With the drift law (C3), this gives one quantity that predicts both inflation (movement) and hiding (prior). It becomes the paper's unifying claim.
- **Only T5 passes:** masking is reported, without a dose–response.
- **T6 passes but T5 fails:** this is reported as exploratory only.
- **Neither passes:** the paper stands as is.

## CONFIRM result (2026-11-11)
T1 to T6 all FAIL in CONFIRM. No shared-prior or masking claim is made. Full numbers: shared-prior results record.
