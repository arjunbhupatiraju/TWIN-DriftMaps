# Pre-registration: REPLICATION (Nov 3, 2026). Part B of the Reliability Atlas

Written before any real repeated-site data were analysed. Code: `part6/replication.py` (PART6_ATLAS.zip, Nov 3; amendment 1 fix included).

## Question
Can TWIN predict, from the recordings alone, whether a neural finding from one lab replicates in another lab?

## Data
- IBL repeated-site recordings in the GATEKEEPER cache: 43 sessions, 10 labs, about 40 mice.
- Regions: CA1, DG, LP, PO, VISa, VISam.
- Valid trials after the first 90.
- 10 to 60 units per region.

## Variables and valid tests
| Variable | Window | Valid test |
|---|---|---|
| block (probabilityLeft) | prior_a | pseudo-sessions, 100 |
| stimulus side | stim_a | pseudo-sessions |
| choice | choice_a | Harris linear shift |
| feedback | feedback_a | Harris linear shift |

## Claims and replication
- **Claim:** (lab, region, variable); the lab's sessions' valid-test p-values, Fisher-combined; significant if p < 0.05.
- **Replication attempt:** the same region and variable in each other lab. Replicated if its combined p < 0.05.
- **PRIMARY set:** attempts to replicate significant claims.
- **SECONDARY set:** all attempts.

## Predictors
1. **TWIN:** empirical-Bayes posterior over code size from the original lab (new codes planted in its own neurons), times the replicating lab's power at each size (new codes planted in its real neurons).
2. **naive:** textbook independent-trials power.
3. **p_orig:** leave-one-lab-out logistic on −log10 of the original p.
4. **const.**

## Tests (PRIMARY)
- **H1:** TWIN AUC CI (cluster bootstrap over original labs) excludes 0.5.
- **H2:** TWIN Brier better than both naive and p_orig (CIs exclude 0).
- **H3:** descriptive only.

## Interpretation (fixed)
| Outcome | Reading |
|---|---|
| H1 + H2 | Replication is predictable from the recordings, better than from p-values or textbook power. |
| H1 only | Weak. |
| H1 fails | Null. |

## Checks before running (mock)
- v1 (raw effect estimate): failed (winner's curse); replaced before real data.
- v2 (empirical Bayes): H1 and H2 PASS on the mock.

## RUN 1 (Nov 3, 22:01): INVALID
- Implementation bug: for choice and feedback, the planted label was the real label, so the lab's own result leaked into its prediction. AUC 0.992.
- Same bug in `atlas.py` (not yet run).

## AMENDMENT 1 (before re-running): bug fix only
- For the linear-shift variables, the planted label is now the real label circularly shifted by a random 25–75% of the session.
- Tests and thresholds unchanged.
- Added a descriptive per-variable AUC.
- Mock re-check: primary AUC 0.856; H1 PASS; H2 FAIL against original p in the mock.

## RUN 2 RESULT (Nov 3, 22:05): the pre-registered result
- 42 sessions, 10 labs, 40 mice; 188 claims, 79 significant.
- Empirical-Bayes prior over r: 0 → 0.16, 0.06 → 0.31, 0.10 → 0.21, 0.13 → 0.10, 0.16 → 0.10, 0.25 → 0.05.

### PRIMARY (566 attempts, replication rate 0.77)
| Predictor | AUC [95% CI] | Brier |
|---|---|---|
| TWIN | **0.771 [0.713, 0.834]** | **0.152** |
| naive | 0.711 [0.646, 0.782] | 0.191 |
| p_orig | 0.645 [0.572, 0.719] | 0.167 |
| const | 0.402 | 0.176 |

- Brier improvement of TWIN over naive: +0.039 [+0.022, +0.057].
- Brier improvement of TWIN over p_orig: +0.015 [+0.003, +0.028].
- **H1 PASS. H2 PASS.**
- **TWIN AUC by variable (descriptive):** choice 0.71 (n = 192); feedback **0.44** (n = 324); stimulus 0.59 (n = 50); block: no significant original claims.
- **Calibration:** good at the top (predicted 0.93, observed 0.90); under-predicts low and middle bins (0.13 → 0.30, 0.31 → 0.48, 0.51 → 0.63).
- **H3 (descriptive):** pseudo-R² TWIN alone 0.165, + lab identity 0.242.

### SECONDARY (1,328 attempts, rate 0.43)
- TWIN AUC 0.908 [0.891, 0.921] vs naive 0.870 and p_orig 0.875.
- Brier gains over naive +0.043, over p_orig +0.020 (both CIs exclude 0).
- By variable: choice 0.65, feedback 0.43, stimulus 0.70.
- H3: 0.425 → 0.462.

### Verdict
The pre-registered tests PASS. Replication across labs is predictable from the recordings, better than from p-values or textbook power.

### Honest caveats (must accompany the claim)
1. The pooled AUC (0.77) is higher than every within-variable AUC. Much of the discrimination is BETWEEN types of claim (which variable / region replicates).
2. Within feedback, TWIN is below chance (0.44). Feedback claims mostly replicate, and the failures are not predicted.
3. The margin over the original p-value is small (Brier +0.015).
4. Lab identity adds explanatory power beyond TWIN (pseudo-R² +0.08 primary). Part of non-replication is lab-specific, not purely statistical.
5. Attempts share labs and claims. The bootstrap is over original labs only.
