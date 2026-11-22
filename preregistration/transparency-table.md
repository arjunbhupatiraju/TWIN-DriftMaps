# Transparency table: every pre-registration, amendment and verdict (PRIOR MAP onward), as of 2026-11-09

"Before data" means before any neural data were analysed with that pipeline. Earlier TWIN pre-registrations (twin v2, calibration v3, replication, cross-dataset, pooled) are in their own files in this folder. Later registrations (calibrate, calibrate-c4, mapquality, shared-prior, causal-validation) are also in their own files; their verdicts are in Table 2 of the paper.

| Analysis | Pre-registration file | Amendments (when, why) | Confirmatory verdicts |
|---|---|---|---|
| PRIOR MAP | preregistration-prior-map | A1, **after the registered run**: Monte Carlo precision only (48 plants, 1,000 nulls); both runs reported, and the registered run stays primary | G1, H1–H4. The registered run is primary. |
| B1, prior vs behaviour (across sessions) | preregistration-prior-behavior | none | **FAIL** (p = 0.066) |
| Step 1, prior vs movement | preregistration-prior-movement | A1 before data: pose-speed bug. A2 before data: wheel samples only on movement. A3, a reporting crash (too many open files) before any result printed | Gate F, G1, V1 PASS; **M1 PASS** |
| Step 2, DECOMPOSE | preregistration-decompose | A1 before data: wheel fix. A2, **after results**: diagnostic, then robust predictors (signed log, 1–99% clip), and a fast effect-size path that reproduced the registered number | F2, D1a, D1b PASS. D2 PASS by rule but **out of range** (1.51). Robust D2 = 0.449 is post hoc. |
| R1/R2, robustness | preregistration-robustness | none | **R1a, R1b PASS**; R2 descriptive |
| P1, prior predicts choice (single region) | preregistration-robustness | none | **FAIL** (p = 0.24) |
| P2, pooled prior predicts choice | preregistration-prior-choice-pooled | **registered after P1 failed**; the only follow-up | PASS (p = 0.040); does not survive correction across P1 and P2 (0.025) |
| Step 3, MAP reanalysis | preregistration-map-reanalysis | A1 during the download, before analysis: speed and safe saving only | Gate, **MA1a, MA1b, MA2, MA3 PASS**. MA2 replicates the robust D2 with the robust predictors fixed in advance. |
| MAP feature richness (69 time-resolved vs 5 marker features) | preregistration-map-rich | none | **RF1, RF2a, RF2b PASS** |
| Step 4, calcium (Stringer 2019) | preregistration-calcium | Design settings chosen on mock data before registration. A1 before results: memory only (single precision, 1 worker). | Gate, **CA1a, CA1b, CA2 PASS** |

## Post-hoc elements the paper must flag
1. The robust D2 (0.449) on IBL. It is confirmed independently by the pre-registered MA2 on MAP (0.375) and CA2 on calcium (0.222).
2. P2 was registered after P1 failed.
3. PRIOR MAP Amendment 1 is a precision change after the run.
4. The MAP area comparisons (medulla vs midbrain) and the Kilosort-drift link are descriptive.
