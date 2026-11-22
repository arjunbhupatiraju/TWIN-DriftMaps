# Pre-registration C4: bias-adjusted overstatement share (registered 2026-11-09, after C1 failed, before C4 was run)

## Why this test exists
The calibration results came back before this test was written.
- C1 FAILED. With no true code, the standard estimate is above zero: +0.0060, CI lower bound +0.0050. But when a true fast code was planted, the valid estimate was further from the truth than the standard one: mean |error| 0.0140 vs 0.0109, advantage −0.0031, CI lower bound −0.0078.
- The valid analysis is conservative. It is biased low by about 0.005 to 0.012 in IBL and by 0.010 to 0.020 in MAP. At truth 0 it reads −0.0023 (IBL) and −0.0103 (MAP).
- C3 PASSED. Pooled Spearman between the law and inflation is 0.706, mouse-bootstrap CI 0.662 to 0.741.

**Threat.** Part of the reported "share that does not survive" (IBL 45%, MAP 37 to 41%, calcium 22%) could come from the valid analysis's own downward bias, not from overstatement by the standard analysis. C4 tests this.

## Method
- **Code:** `part7/calibrate.py adjust`, SHA-256 prefix d4dc7b7d89ab9175.
- **Fit:** for each dataset and analysis, estimate = a + b × truth, fitted to the planted-code calibration (calibration_long.csv).
- **Adjust:** each population's standard and valid estimate in calibrate_rows.csv is mapped back to the truth scale as (estimate − a) / b.
- **Adjusted share:** 1 − Σ adjusted valid / Σ adjusted standard.
- **Uncertainty:** a mouse bootstrap (500 resamples) that resamples mice in the calibration and in the populations together, and refits the calibration each time.
- **Sensitivity:** an intercept-only correction (b = 1).

## Criterion
The adjusted share (intercept and slope) has a mouse-bootstrap lower bound above 0.10 in IBL and in MAP. This is the same 10% threshold as MA2.

## Interpretation fixed in advance
- **PASS:** the overstatement claim stands after both analyses are corrected for their own bias. The adjusted share replaces the unadjusted one as the headline magnitude, or is reported beside it.
- **FAIL:** the magnitude claim is withdrawn. The paper keeps the significance claim (D1, R1, MA1, CA1) and the drift law (C3), and reports that the size of the overstatement cannot be separated from the conservatism of the valid estimator.
- **Either way:** C1 is reported as FAILED, and the valid estimator's downward bias is stated in Results and Limitations.

## Check on synthetic data (not evidence)
The synthetic setup had a true share of about 0.38. The unadjusted share read 0.66. The adjusted share recovered 0.40 [0.38, 0.42] (IBL-like) and 0.36 [0.33, 0.39] (MAP-like).

## Will not do
- Change the calibration levels, the fit form, or the criterion after seeing C4.
