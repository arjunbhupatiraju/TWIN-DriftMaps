# Pre-registration: REPLICATION CONFIRMATION on the Brain-Wide Map (Nov 3, 2026)

Written after the repeated-site result (`preregistration-replication.md`, run 2: H1 and H2 PASS) and BEFORE the confirmation data were computed. The TWIN model is frozen as in run 2. Amendment 1 (C4) was added before any confirmation data existed. Code: `part6/atlas.py run` and `part6/replication.py confirm`.

## Data
- All IBL Brain-Wide Map sessions in the cache EXCEPT repeated-site recordings.
- Regions recorded by at least 2 labs.
- The same four variables, windows and valid tests.
- Planted grid r = 0.02 to 0.3, 24 per size.

## Tests (PRIMARY = attempts to replicate significant claims)
- **C1:** TWIN AUC CI excludes 0.5.
- **C2:** TWIN Brier beats naive power AND the original p.
- **C3:** SECONDARY set: within-variable TWIN AUC CI excludes 0.5 in at least half of the variables.
- **C4:** TWIN vs a frozen DESIGN model (logistic on variable type + replicating lab's sessions, log N, log T; trained on the repeated-site attempts).

## Interpretation (fixed)
| Outcome | Reading |
|---|---|
| C1–C3 + C4 = TWIN BETTER | Headline: TWIN's simulation adds value beyond design rules. |
| C1–C3 + C4 = NO DIFFERENCE / DESIGN BETTER | Replication across labs is predictable, mostly from design and claim type. TWIN is a zero-shot route, not a better one. |
| C1 fails | Not confirmed. |

## RESULT (Nov 3, 22:26, Mac)
Data: 425 sessions, 12 labs, 135 mice, 180 regions (repeated-site sessions excluded); 3,705 lab-level claims, 1,615 significant. Empirical-Bayes prior over r: broad, mass 0.04–0.2.

### PRIMARY (9,414 attempts, replication rate 0.78)
| Predictor | AUC [95% CI] | Brier |
|---|---|---|
| TWIN | 0.748 [0.732, 0.765] | 0.158 |
| naive | 0.639 | 0.206 |
| original p | 0.644 | 0.167 |
| const | 0.434 | 0.175 |
| **Design model (frozen, trained on 42 repeated-site sessions)** | **0.831** | **0.148** |

- TWIN Brier gains: over naive +0.048 [+0.030, +0.063]; over original p +0.009 [+0.002, +0.016]; over the design model −0.011 [−0.018, −0.003].
- Within-variable TWIN AUC: choice 0.64 [0.63, 0.66], n = 3,562; feedback 0.71 [0.68, 0.74], n = 4,777; stimulus 0.59 [0.53, 0.63], n = 1,002; block: n = 73, not testable.
- Calibration (predicted → observed): 0.13→0.38, 0.31→0.50, 0.51→0.60, 0.71→0.70, 0.93→0.88.

### SECONDARY (20,952 attempts, rate 0.45)
- TWIN AUC 0.887 [0.880, 0.894], Brier 0.136; naive 0.820; original p 0.853.
- **Design model: AUC 0.922, Brier 0.115.**
- Within-variable TWIN AUC: block 0.68, choice 0.64, feedback 0.74, stimulus 0.64 (all CIs > 0.5).

### Verdict
- C1 PASS, C2 PASS, C3 PASS (4 of 4), **C4 DESIGN MODEL BETTER**.
- Pre-registered reading: replication across labs is predictable, mostly from design and claim type. TWIN is a zero-shot route, not a better one.
