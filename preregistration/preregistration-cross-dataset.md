# Pre-registration: CROSS-DATASET replication test, IBL ↔ Steinmetz 2019 (Nov 3, 2026)

Written before the Steinmetz rows were computed. TWIN v2 and the design rule are frozen as in `preregistration-twin-v2.md`. Code: `part6/steinmetz_rows.py` and `part6/replication.py confirm3`.

## Why
- Reviewers' main attack: "all IBL data."
- Steinmetz et al. 2019 differs in era, rig, task version (no blocks), spike sorting and pipeline.

## Data
- **Steinmetz:** all sessions with at least 150 valid trials. Regions with at least 10 units (cap 60).
- **Variables:** stimulus side, choice, feedback, previous choice. Linear-shift valid test.
- **IBL:** atlas rows excluding repeated-site sessions; regions also recorded in Steinmetz.
- **Steinmetz = one "lab".** Only cross-dataset attempts are scored.
- **PRIMARY excludes IBL cortexlab.**

## Tests (PRIMARY; cluster bootstrap over original labs)
- **X1:** TWIN v2 AUC CI > 0.5.
- **X2:** TWIN v2 Brier beats naive AND original p (both CIs > 0).
- **X3:** TWIN v2 Brier beats the frozen design rule.

## Interpretation (fixed)
| Outcome | Reading |
|---|---|
| X1 + X2 (+ X3) | Transfers to an independent dataset (and beats the design rule). |
| X1 + X2, X3 fails | Transfers, but no better than the design rule. |
| X1 fails | Does not transfer. |

## RESULT (Nov 4, 17:43, Mac)
21 Steinmetz sessions (8 mice); 344 IBL sessions; 61 shared regions; 2,478 cross-dataset attempts (2,732 including cortexlab). Per-variable v2 priors (mean r): choice 0.135, feedback 0.206, prev_choice 0.061, stimulus 0.039.

### PRIMARY (1,567 attempts, replication rate 0.75)
| Predictor | AUC [95% CI] | Brier |
|---|---|---|
| **TWIN v2** | **0.789 [0.719, 0.807]** | **0.167** |
| design rule | 0.763 | 0.189 |
| TWIN v1 | 0.710 | 0.176 |
| naive | 0.623 | 0.227 |
| original p | 0.522 | 0.192 |
| const | 0.383 | 0.199 |

v2 Brier gains: over design **+0.022 [+0.016, +0.041]**; over naive +0.060 [−0.043, +0.104]; over original p +0.025 [−0.034, +0.051].

By direction: IBL claim → Steinmetz: n = 699, rate 0.84, v2 AUC 0.741 (design 0.734); Steinmetz claim → IBL: n = 868, rate 0.68, v2 AUC 0.817 (design 0.784).

### SECONDARY (2,478 attempts, rate 0.63)
v2 AUC 0.818, Brier 0.185; design 0.779 / 0.213; naive 0.704 / 0.264; original p 0.675 / 0.215. Gains: over design +0.027 [+0.008, +0.066]; over naive +0.079 [+0.009, +0.113]; over original p +0.029 [−0.017, +0.053].

### Verdict
**X1 PASS, X2 FAIL, X3 PASS.** Point estimates favour TWIN v2 over naive and original p, but the cluster-bootstrap CIs include 0. Diagnosis (after the result; not grounds to change the verdict): Steinmetz is a single cluster holding about 55% of PRIMARY attempts, so intervals are very wide. The pre-registered verdict stands; the remedy is more independent datasets in a new pre-registered test.
