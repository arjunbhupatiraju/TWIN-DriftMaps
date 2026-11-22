# Pre-registration: CALIBRATION fix, TWIN v3 (Nov 5, 2026)

## Problem
TWIN v2 is miscalibrated at the extremes. When it predicts about 0.10, about 0.25–0.38 replicate; at the top it predicts about 0.95 and 0.85–0.90 replicate.

## Diagnosis (hypothesis)
v2 assumes every lab measuring a region × variable has the SAME code size. Real labs differ, so ignoring between-lab spread makes predictions too extreme.

## TWIN v3 (zero-shot; claim p-values only)
- Each lab's code size varies around a shared size: log-normal spread tau on (r + 0.01).
- Per variable, tau and the prior over the shared size are fitted by maximum marginal likelihood (EM) over all labs' claims.
- Prediction = posterior over the shared size, then the replicating lab's own spread, times its power.

## Protocol (order fixed)
1. Fresh data (set 3).
2. Exploratory check on already-seen sets.
3. Confirmatory test on set 3.

## Confirmatory criteria
- **Q1:** v3 ECE lower than v2.
- **Q2:** v3 calibration-slope CI overlaps [0.8, 1.25].
- **Q3:** v3 AUC not worse than v2 by more than 0.01.

## Interpretation
| Outcome | Reading |
|---|---|
| Q1 + Q2 + Q3 | Calibration fixed. |
| Q1 only | Better, still imperfect. |
| Q1 fails | Limitation. |

## STEP 2 RESULT: exploratory (already-seen sets)
| Set (PRIMARY) | ECE v2 → v3 | Slope v2 → v3 | AUC v2 → v3 |
|---|---|---|---|
| BWM original | 0.083 → 0.033 | 0.45 → 0.80 | 0.836 → 0.843 |
| BWM new | 0.103 → 0.033 | 0.41 → 0.80 | 0.774 → 0.779 |
| Pooled cross-dataset | 0.118 → 0.064 | 0.32 → 0.63 | 0.813 → 0.826 |

Decision: v3 improves, so it was not revised.

## STEP 3 RESULT: confirmatory (Nov 5, 20:28)
Fresh variables: choice_2back, feedback_2back, prev_stimulus. 425 sessions, 12 labs, 180 regions; 2,787 claims.

### PRIMARY (4,348 attempts, replication rate 0.35)
| Metric | v2 | v3 | v3 improvement [95% CI] |
|---|---|---|---|
| ECE | 0.099 | 0.081 | **+0.018 [+0.001, +0.028]** |
| Calibration slope | 0.46 | **0.56 [0.43, 0.69]** | |
| AUC | 0.714 | 0.713 | −0.001 [−0.009, +0.008] |
| Brier | 0.207 | 0.202 | +0.005 [+0.001, +0.008] |

Bins (predicted → observed):
- v2: 0.11→0.20, 0.29→0.27, 0.49→0.38, 0.69→0.51, 0.91→0.74;
- v3: 0.11→0.20, 0.30→0.28, 0.49→0.42, 0.70→0.61, 0.88→0.71.

### SECONDARY (15,804 attempts, rate 0.28)
ECE 0.095 → 0.089; slope 0.45 → 0.50; AUC 0.682 → 0.683.

### Verdict
**Q1 PASS, Q2 FAIL, Q3 PASS.** Pre-registered reading: "better, still imperfect."
- v3 reliably reduces calibration error without losing discrimination.
- But predictions remain too extreme (slope 0.56).
- On these weak 2-back history variables, TWIN over-predicts replication at the top (0.88 → 0.71).

### Diagnosis (after seeing the result; not grounds to change the verdict)
- The calibration slope falls as effects get weaker and the replication rate gets lower: development sets 0.80 at rates 0.68–0.78; set 3 0.56 at rate 0.35.
- A plausible cause: TWIN's planted-code power assumes the code is the same KIND of signal as planted (20% of neurons, linear). Weak real effects near threshold are noisier than that model.

### Legitimate next options
1. Report as a limitation.
2. A NEW pre-registered test on NEW data of TWIN v3 + a fixed recalibration layer, fitted ONCE on all previously seen outcome sets and frozen (no longer strictly zero-shot; reported as such).
