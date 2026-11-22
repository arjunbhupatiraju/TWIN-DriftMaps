# Pre-registration: closing the core claim (calibration C1, mouse-level statistics C2, drift law C3)

Registered 2026-11-09, before any real data were analysed with this script. Code: `part7/calibrate.py`, SHA-256 prefix b8d368a624947cdb. It imports the frozen step 2 code (part6/decompose.py) and step 3 code (part7/map_decompose.py). It has been tested only on synthetic MAP-format data. Nothing new is downloaded.

## Data
- **IBL:** every 2nd session with step 2 features; all regions, 4 epochs; robust movement predictors.
- **MAP:** all sessions, insertions and epochs, with the step 3 filters.
- Units: up to 150 per population (random subsample).

## C1: calibration against a known truth (confirmatory)
**Design.**
- For each population, one drift twin of its neurons is made, with no movement link.
- A planted code is g = z-score(drift-removed movement features × random weights): a fast movement composite from the session's own movement.
- The code is planted into a random half of the neurons, so that the true population unique R² is 0, 0.01, 0.03 or 0.06.
- Both analyses (standard: interleaved CV, drift left in; valid: contiguous CV, drift removed) estimate movement unique R².
- Truth is defined as the planted fast-movement variance. By design, the valid analysis targets trial-to-trial (fast) encoding. Slow covariation between movement and neurons cannot be distinguished from drift, and this is stated as a scope limit.

**Criterion (PASS needs both):**
- With no true code, the standard estimate is above 0 (mouse-bootstrap lower bound > 0).
- With a true code (pooled over levels > 0), the valid estimate is closer to the truth: the mean |error| of the standard estimate minus the mean |error| of the valid estimate has a mouse-bootstrap lower bound above 0.

Biases at each level are reported for IBL, MAP and both pooled.

## C2: mouse-level intervals (descriptive)
Whole mice are resampled (500 resamples) for:
- C1 and C3;
- the overstatement share in this run;
- the earlier results via `calibrate.py mouse`: IBL D1 and R1, MAP MA1, MA2 and RF1, calcium CA1 and CA2, and step 1 M1 (EM estimator recomputed per resample, with row weights equal to the mouse counts).

The earlier pre-registered verdicts are not changed. These intervals are reported alongside them.

## C3: the drift law (confirmatory)
- **Predictor:** law = (slow share of neural variance) × (slow share of movement variance), where slow share = 1 − variance after drift removal / total variance, averaged over neurons or features.
- **Outcome:** inflation = standard − valid movement unique R², per population.
- **Criterion:** the pooled Spearman correlation between law and inflation has a mouse-bootstrap lower bound above 0. Per-dataset values and inflation by quarter of the law are reported.

## Mock check (before registration; not evidence)
On 3 synthetic sessions (24 populations, 3 mice): standard bias at truth 0 was +0.023 (lower bound > 0); valid bias −0.0004. At truths 0.01, 0.03 and 0.06, valid biases were −0.003 to −0.005 and standard biases +0.019 to +0.005. The |error| advantage of the valid estimate was +0.009, with a lower bound of −0.004 (only 3 mice). C3: Spearman 0.30.

The first draft required the standard bias to be above 0 at every level. The mock showed that standard bias shrinks as the true code grows, so the criterion was changed, before registration, to the pooled |error| comparison.

## Will not do
- Change the planted levels, the fraction of neurons, the composite, the CV schemes, the law definition or the criteria after seeing results.

## Amendment 1 (2026-11-09, after the run, before `test` was run; file format only)
- **Problem.** The `run` step appended IBL rows (no `probe` column) and MAP rows (with `probe` after `N`) to the same CSV under the IBL header. `test` failed while reading the file, at line 3723. No result had been computed.
- **Fix.** `read_rows()` now parses each line by its own length and inserts `probe` for MAP lines. A `fix` command rewrites the file with one header and keeps a backup. No rows are recomputed, and nothing in the analysis, criteria or populations changes.
- **Code.** SHA-256 prefix after the amendment: ca4c7070d162c9ee.
