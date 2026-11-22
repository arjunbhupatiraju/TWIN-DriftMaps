# Pre-registration: TWIN v2 vs the design rule on NEW variables (Nov 3, 2026)

Written before the new-variable data were computed. Code: `part6/atlas.py run --new-variables` and `part6/replication.py confirm2` (PART6_ATLAS.zip, Nov 3, about 22:35).

## Context
- In the Brain-Wide Map confirmation, a 4-feature design rule (variable type + replicating lab's sessions, neurons and trials, trained on the repeated-site outcomes) beat TWIN v1 (C4).
- TWIN is zero-shot (it needs no replication outcomes), so the decisive test is claims of kinds NEITHER model was built with.

**Disclosure:** v2's two changes were motivated by what we saw in the v1 Brain-Wide Map results (miscalibration by variable). The NEW-variable data had not been computed when v2 was frozen.

## TWIN v2 (zero-shot; claim p-values only)
1. Per-variable empirical-Bayes prior over code size.
2. Evidence pooled across labs: the original lab's likelihood plus the other labs' likelihoods (excluding the replicating lab), tempered by W_OTHER = 0.5, fixed a priori.
3. Prediction = posterior · power of the replicating lab's sessions.

Mock mechanism check (PRIMARY): v1 AUC 0.83 / Brier 0.156 → v2 AUC 0.96 / Brier 0.070.

## Data (confirmatory)
- NEW variables: previous trial's choice, previous trial's reward, absolute contrast, log RT. All use the linear-shift valid test.
- Brain-Wide Map minus repeated-site sessions; regions recorded by at least 2 labs.

## Opponent
The frozen design rule (C4). For unseen variables it uses the average of its predictions over known variable codings.

## Tests (PRIMARY)
- **N1:** TWIN v2 Brier better than the design rule (CI > 0).
- **N2:** v2 AUC CI > 0.5 and v2 beats naive and original p.

## Interpretation (fixed)
| Outcome | Reading |
|---|---|
| N1 + N2 | TWIN predicts replication for new kinds of claims better than a rule learned from replication data, and better than p-values or textbook power. |
| N2 only | Works zero-shot, but not better than the design rule. |
| N2 fails | Does not generalize. |

## RESULT (Nov 3, 22:43, Mac)
425 sessions, 12 labs, 135 mice, 180 regions. NEW variables: 3,712 claims, 1,970 significant. Per-variable v2 priors (mean r): contrast 0.039, prev_choice 0.051, prev_feedback 0.138, rt 0.153.

### PRIMARY (11,707 attempts, replication rate 0.68)
| Predictor | AUC [95% CI] | Brier |
|---|---|---|
| **TWIN v2** | **0.774 [0.763, 0.782]** | **0.185** |
| design rule | 0.663 [0.650, 0.677] | 0.217 |
| TWIN v1 | 0.684 | 0.215 |
| naive | 0.636 | 0.278 |
| original p | 0.615 | 0.209 |
| const | 0.462 | 0.217 |

v2 Brier gains (all CIs exclude 0): over design +0.033 [+0.029, +0.036]; over v1 +0.030; over naive +0.093; over original p +0.025.

Within-variable v2 AUC: contrast 0.73 [0.71, 0.76]; prev_choice 0.71 [0.69, 0.73]; prev_feedback 0.70 [0.69, 0.71]; rt 0.72 [0.70, 0.73].

Calibration (predicted → observed): 0.11→0.27, 0.30→0.46, 0.50→0.57, 0.70→0.69, 0.95→0.85.

### SECONDARY (21,012 attempts, rate 0.56)
- v2 AUC 0.792, Brier 0.199; design 0.639 / 0.247; v1 0.713 / 0.235; original p 0.685 / 0.222.
- v2 gain over design +0.049 [+0.047, +0.051].

### Verdict
**N1 PASS, N2 PASS.**

### Fair caveats
- On unseen variables the design rule has no variable-specific information; its Brier (0.217) equals the constant's. N1 shows that TWIN generalizes where outcome-trained rules cannot, but the opponent is handicapped by design.
- Calibration at the extremes needs work.
- All IBL data; cross-institution generalization untested in this test.
