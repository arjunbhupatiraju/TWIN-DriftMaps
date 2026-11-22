# Pre-registration (FROZEN 2026-11-11, 19:20 CT): CAUSAL VALIDATION

Frozen after `causal.py check` and before `run`, `inact` or `test`; no encoding or inactivation statistic had been computed. Code: `part8/causal.py`, SHA-256 prefix **159744354915a5aa**. The changes from the draft hash 91351a5c are bug fixes only: the atlas surface is computed on demand, and a single-point label lookup now works.

## Data layout established by `check`
- **Steinmetz:** 39 sessions. Choice codes −1/0/1 (0 = NoGo). Wheel moves are present.
- **Zatka-Haas:** 47,002 trials, 5 mice, 91 sessions. Response column L, feedback M, left = 1; consistency with the contrasts 1.000.
- **Areas fixed by the rule** (≥ 30 good units in ≥ 3 sessions AND ≥ 1 inactivation site), 9 in total:

| Area | Recording sessions | Inactivation sites |
|---|---|---|
| MOs | 16 | 8 |
| VISp | 11 | 7 |
| VISam | 11 | 2 |
| PL | 8 | 2 |
| SCs | 5 | 2 |
| VISpm | 4 | 2 |
| VISa | 4 | 2 |
| VISl | 3 | 1 |
| MOp | 3 | 6 |

  Sites in RSPd, the SSp subfields and VISal have no matching recordings and are not used. Units labelled "root" (unassigned) are excluded.
- **Caveats noted now:**
  - Sites labelled PL and SCs come from the 300 µm surface rule at the medial frontal and far posterior edges. They are kept because the rule is fixed.
  - With 9 areas, the power caveat below applies fully.

## Background
- **The two datasets:** Zatka-Haas et al. 2021 (eLife) silenced dorsal cortex optogenetically during the same task that Steinmetz et al. 2019 recorded with Neuropixels.
- **Their finding:** coding is widespread, but only a few areas (mainly VIS and MOs) affect choice.
- **Our question:** does removing slow drift make the coding map match the causal map better?

## Encoding (per area-session; up to 150 units, square-root spike counts, fixed-α ridge decoder, α = 10)
- **A1 primary, choice:** left vs right (Go trials only), −150 to 0 ms before movement onset. Movement onset is the first wheel move after the stimulus and no later than the response time; if there is none, the response time is used.
- **A2 secondary, signed contrast:** right minus left, 0–150 ms after stimulus onset.
- **Effect size:** held-out r minus the mean r of that analysis's own null (40 draws).
- **Standard analysis:** interleaved 5-fold CV, trial-shuffle null.
- **Valid analysis:** the paper's cosine drift basis removed from neurons and target separately; contiguous 5-fold CV; linear-shift null (±T/4, central trials).

## Causal effect (per area)
- **Per site:** laser minus no-laser P(contralateral choice) on trials where the contralateral stimulus has the higher contrast, matched by session (weighted by laser trials).
- **Area causal strength:** −mean over the sites in the area.

## Tests
| Test | Statistic | PASS |
|---|---|---|
| A1 | ρ_valid − ρ_standard; ρ = Spearman across the 9 areas between mean encoding effect (choice) and causal strength | bootstrap (1,000) resampling recording mice and inactivation mice independently; lower 95% bound > 0 |
| A2 | the same, signed contrast | as A1 |
| A3 | areas whose encoding rank moves ≥ 3 places | descriptive |

## Synthetic check (not evidence)
- **No true code, shared slow drift:** the standard test gives p = 0.024 (a false positive); the valid test gives p = 0.29.
- **Drift artefact unrelated to the causal map:** A1 PASS.
- **No artefact:** NOT RESOLVED.

## Power caveat and interpretation (fixed)
- **A1 passes:** drift-corrected coding maps are closer to causal maps; the correction changes biology, not only numbers.
- **Otherwise:** "not resolved with current data". Report the area table and scatter. Never report it as "no effect".

## Not doing
- Changing windows, variables, α, null sizes, the area list, or primary vs secondary.
