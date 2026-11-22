# Pre-registration: gap fixes R1, R2 (robustness of the movement tests) and P1 (hidden prior predicts choice)

Registered 2026-11-08, before any real data were analysed with these scripts. Code:
- `part7/robustness.py`, SHA-256 prefix 9fc492cf02f39bc8;
- `part7/prior_choice.py`, SHA-256 prefix 6fb22372cd696541.

Both have been tested only on mock data.

## R1: an assumption-free null for movement encoding (confirmatory)
**Why.** The drift twins (phase randomization) assume stationary drift, and a reviewer can say real drift has jumps. Here nothing is simulated.

**Design.**
- The real neurons of one session are paired with the real movement of another mouse's session, in the same trial window, using its first T trials.
- There are 2 partner sessions per session, drawn at random from other mice.
- We use the step 2 pipeline with the robust movement predictors (signed log, clipped at the 1st and 99th percentiles), on every 4th session (about 110 sessions), all regions, all 4 windows.
- Both step 2 tests are applied: the standard test (interleaved CV, 100 trial shuffles) and the valid test (contiguous CV on drift-removed data, linear-shift null).

**Criteria.**
- R1a: the valid test has a false-positive rate of at most 0.07, with a session-bootstrap upper 95% bound of at most 0.08.
- R1b: the standard test's lower 95% bound is at least 0.20.

## R2: power of the valid test (descriptive)
- Real movement (the first principal component of the session's drift-removed movement features) is planted into drift twins of the real neurons. It goes into 20% of neurons, at per-neuron r = 0.05, 0.1 and 0.2, with 2 plants each.
- We report the detection rate of the valid test, overall and per window.
- There is no pass/fail. The result answers "is the valid test too strict?"

## P1: does the movement-free neural prior predict choice, trial by trial? (confirmatory)
**Why.** B1 (across sessions) failed narrowly, at p = 0.066. P1 is the within-session, trial-level test, which is more powerful.

**Design.**
- Data: step 1 sessions, regions, pre-stimulus window, burn-in and unit limits, with movement removed from the neurons and from every target exactly as in step 1.
- The Bayes prior and 200 pseudo-session priors are decoded with the PRIOR MAP decoder (contiguous 5-fold, GCV ridge). This gives held-out predictions for each trial.
- Trials: |contrast| ≤ 0.0625, with at least 40 per session.
- Statistic: the partial correlation between the held-out neural prior estimate and choice. It controls for signed contrast, the Bayes prior, choice at lags 1 and 2, and rewarded choice at lags 1 and 2.
- Null: the same statistic for each pseudo-session decoder (same neurons, drift and arousal; a prior that does not exist).
- Pooled test: the mean partial r over all session-regions is compared with the distribution of the mean of the j-th pseudo statistic over all session-regions (200 draws).

**Criterion.** P1 PASS if the mean partial r is above 0 and the one-sided p is below 0.05.

**Descriptive.** The fraction of sessions with a positive mean; per-region Fisher and BH (q = 0.05); the strongest regions; the Spearman correlation across regions with step 1's P(carries the prior after movement removal).

**Reading.**
- PASS: trial-by-trial fluctuations in the brain's movement-free prior representation predict the mouse's choice beyond the ideal-observer prior and its own recent history.
- FAIL: no such link is detectable at the trial level. B1 and P1 are reported together.

## Mock checks (before registration; not evidence)
- **P1, null mock** (choices unrelated to neurons): mean partial r +0.003, null 95th percentile +0.010, p = 0.27, FAIL as expected; 0 of 8 regions significant.
- **P1, positive-control mock** (choices driven by the prior estimate in one region, ACA): mean r +0.042, p = 0.005, PASS. Only ACA was significant (r +0.39).
- **R1/R2 mock** (4 sessions; too small for the criteria): valid false positives 6% vs standard 100%. Power rose with r (0.09, 0.06 and 0.44 at r = 0.05, 0.1 and 0.2).

## Will not do
- Change windows, predictors, trial filters, covariates, nulls or criteria after seeing results.
- Drop sessions or regions based on results.
