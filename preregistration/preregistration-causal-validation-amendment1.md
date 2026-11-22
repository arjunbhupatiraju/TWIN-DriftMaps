# CAUSAL VALIDATION: first run and Amendment 1 (2026-11-11, 19:35 CT)

Pre-registration: preregistration-causal-validation.md (hash 159744354915a5aa).

## Run 1 result (as pre-registered)
- **A1 (choice):** ρ standard −0.32, ρ valid −0.38; difference −0.07 [−0.46, +0.29]. NOT RESOLVED.
- **A2 (contrast):** ρ standard −0.43, ρ valid −0.42; difference +0.02 [−0.57, +0.86]. NOT RESOLVED.

## Problem found: the causal measure does not reproduce the published result
- **What we got:** VISp causal strength −0.001 and MOs −0.027, i.e. no reduction in contralateral choices.
- **What Zatka-Haas et al. 2021 report:** VIS and MOs inactivation strongly reduces contralateral choices (about −35% and −22%).
- **Likely cause:** the hemisphere convention of the laser ML coordinate relative to the left/right contrast columns is not stated in the .mat file. Run 1 assumed ML < 0 = left hemisphere. If that is backwards, every "contralateral" effect is measured on the wrong side, which would give near-zero effects.
- **Consequence:** run 1 is uninterpretable. It cannot count as a test of the hypothesis, because its causal map fails a validity check that is independent of our hypothesis.

## Amendment 1 (bug fix; decided before any rerun)
- **Rule:** the hemisphere convention is chosen as the one that makes the mean VISp-site effect most negative, i.e. the one that reproduces the PUBLISHED direction.
- **Why this is independent:** the rule uses only inactivation data at VISp sites and no encoding statistic. It cannot be tuned toward A1 or A2.
- **Also printed:** both conventions' VISp and MOs effects, for transparency.
- **Unchanged:** the tests, areas, encoding (encoding.csv is not rerun) and thresholds.
- **New hash:** c1536c8f4ed50883.

## Reporting
Both runs are reported:
- run 1 as a failed measurement-validity check;
- run 2 as the pre-registered test with the corrected hemisphere convention.

If neither convention reproduces the published VISp and MOs effects, the causal measure is still wrong. Spine A is then stopped and reported as not resolved.

## Outcome
Neither convention reproduced the published effect (VISp effect +0.039 vs about −35% published). The analysis was stopped and is reported as NOT RESOLVED.
