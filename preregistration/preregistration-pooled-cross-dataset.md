# Pre-registration: POOLED CROSS-DATASET test, IBL + Steinmetz 2019 + Allen Visual Behavior Neuropixels (Nov 4, 2026)

Written BEFORE any Allen data were computed. TWIN v2 and the design rule are frozen. Code: `part6/allen_rows.py run` and `part6/replication.py confirm4`.

## Why this test exists (disclosure)
- In the Steinmetz-only test: X1 PASS, X3 PASS, X2 FAIL. The original-lab bootstrap was degenerate (one external "lab" held about 55% of attempts).
- That verdict stands; this is a NEW test with NEW data.
- The region-level bootstrap was chosen because of that failure, fixed before the Allen data existed. The original-lab bootstrap is also reported.

## Data
- IBL atlas rows (minus repeated site), Steinmetz rows, and Allen Visual Behavior Neuropixels (non-abnormal sessions, good units).
- **Allen variable mapping (pre-registered):** stimulus = image change vs catch (0–100 ms); choice = lick vs no lick (150–350 ms); feedback = hit vs miss on go trials (500–700 ms); prev_choice = previous trial's lick (−500–0 ms).
- Linear-shift valid test. **Caveat accepted in advance:** cross-task analogues.
- Scored: attempts involving an external dataset. PRIMARY excludes IBL cortexlab.

## Tests (PRIMARY; bootstrap over brain regions)
- **P1:** TWIN v2 AUC CI > 0.5.
- **P2:** v2 beats naive AND original p.
- **P3:** v2 beats the frozen design rule.

## RESULT (Nov 5, 10:43, Mac)
Allen: 102 sessions, 54 mice; Steinmetz: 21 sessions, 8 mice; 91 shared regions; 4,044 cross-dataset attempts (4,450 with cortexlab).

### PRIMARY (2,561 attempts, replication rate 0.76), region bootstrap (pre-registered)
| Predictor | AUC [95% CI] | Brier |
|---|---|---|
| **TWIN v2** | **0.811 [0.774, 0.838]** | **0.159** |
| design rule | 0.750 | 0.181 |
| TWIN v1 | 0.734 | 0.171 |
| naive | 0.650 | 0.206 |
| original p | 0.490 | 0.190 |
| const | 0.385 | 0.184 |

v2 Brier gains (all CIs exclude 0): over design **+0.022 [+0.006, +0.039]**; over naive **+0.046 [+0.028, +0.068]**; over original p **+0.030 [+0.011, +0.049]**.

By direction (v2 AUC vs design AUC): IBL/other claim → Allen, n = 448, 0.897 vs 0.599; Allen claim → others, n = 546, 0.793 vs 0.765; → Steinmetz, n = 754, 0.769 vs 0.734; Steinmetz claim → others, n = 927, 0.825 vs 0.778.

### PRIMARY, original-lab bootstrap (reported)
v2 gain over design +0.022 [+0.017, +0.035]. Over naive +0.046 [−0.015, +0.087] and over original p +0.030 [−0.007, +0.046]: CIs include 0 under this clustering.

### SECONDARY (4,044 attempts, rate 0.63)
v2 AUC 0.828, Brier 0.181; design 0.733 / 0.218; naive 0.728 / 0.247; original p 0.690 / 0.219. Gains over design / naive / original p: +0.037 / +0.067 / +0.038; CIs exclude 0 under both bootstraps.

### Verdict
**P1 PASS, P2 PASS, P3 PASS.**

### Must be reported with it
1. Under the original-lab bootstrap, the PRIMARY gains over naive and original p have CIs that include 0. The SECONDARY set passes under both.
2. Allen variables are cross-task analogues.
3. The Steinmetz-only test's X2 failure stays on record.
