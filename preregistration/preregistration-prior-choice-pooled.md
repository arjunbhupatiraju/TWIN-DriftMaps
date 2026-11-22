# Pre-registration: P2, does the pooled, movement-free prior predict choice? (follow-up to P1)

Registered 2026-11-08, AFTER P1 failed and before any real data were analysed with this script. Code: `part7/prior_choice_pooled.py`, SHA-256 prefix 6e9e05ae6b659e84. It imports the decoder and the partial-correlation code unchanged from `part7/prior_choice.py` (P1, SHA-256 prefix 6fb22372cd696541, unchanged).

## Status, stated up front
This is a new hypothesis motivated by the P1 failure. It is not a re-test of P1. P1 stays FAIL. P1 and P2 are reported together, and this follow-up is the only one. If P2 fails, there are no further behavioural variants on these data.

## Rationale
The per-region prior code is weak (mostly r = 0.04–0.1). A single region's held-out estimate on a single trial is therefore mostly noise. If behaviour uses the prior, it most plausibly pools it across regions.

## Design
P2 is identical to P1 in everything except one thing: the neurons of every eligible region in a session are combined into one decoder.

Unchanged from P1:
- the step 1 movement removal, applied to the neurons and to every target;
- the pre-stimulus window, the 90-trial burn-in, and the per-region eligibility (≥ 10 units, capped at 150 per region);
- the 200 pseudo-session priors;
- the PRIOR MAP decoder (contiguous 5-fold, GCV ridge), giving held-out predictions;
- low-contrast trials only (|contrast| ≤ 0.0625, at least 40 per session);
- the partial correlation with choice, controlling for signed contrast, the Bayes prior, choice at lags 1 and 2, and rewarded choice at lags 1 and 2.

The pooled test is one value per session. The mean over sessions is compared with the distribution of the mean of the j-th pseudo statistic over sessions.

## Criterion
P2 PASS if the mean partial r is above 0 and the one-sided p is below 0.05.

## Descriptive
- the fraction of sessions with a positive partial r;
- the fraction individually significant (5% expected by chance);
- the Spearman correlation between the number of pooled neurons and partial r.

## Mock checks (before registration; not evidence)
- **Null mock** (56 sessions): +0.015, p = 0.080, FAIL.
- **Positive-control mock** (choices driven by one region's prior estimate): +0.094, p = 0.005, PASS.

## Will not do
- Change trials, covariates, pooling, window or criterion after seeing the result.
- Run further behavioural variants if P2 fails.
