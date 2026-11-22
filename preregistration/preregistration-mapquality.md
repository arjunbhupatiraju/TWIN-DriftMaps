# Pre-registration: MAP QUALITY, how much of a standard brain-wide encoding map can be trusted?

Registered 2026-11-09, before the script was run on real data. Code: `part7/mapquality.py`, SHA-256 prefix d79063c58dfb7b1e. It reads only the existing outputs of the Reliability Atlas (`part6/atlas.py`); no neural data are reanalysed. It has been tested only on synthetic data.

## Why
The paper may be reframed around one headline number: how much of a standard brain-wide map is trustworthy. This follows the precedent of Marek et al. (Nature, 2022) for human brain-wide association studies. The decision to reframe is fixed below, before the number is seen.

## Data
- `atlas_rows*.csv`: per session-region tests (variable, region, p_shuffle, p_valid, subject or session).
- `atlas.csv`: per region-variable power (sessions_needed_r0.1, sessions_available).
- Every variable and region in those files is included; nothing is dropped.

## Definitions (per region × variable cell)
- **Standard claim:** the Fisher-combined p_shuffle over the cell's sessions passes Benjamini–Hochberg q < 0.05 across all cells. This is what a standard map reports.
- **Valid claim:** the same, using p_valid.
- **Answerable:** sessions_needed (80% power at r = 0.1, valid test) ≤ sessions_available.
- **Settled:** a valid claim, or answerable with no valid claim (a powered negative).

## Primary outcomes
- **X:** the share of standard claims that are also valid claims, meaning they survive drift.
- **Y:** the share of all cells that are settled.
- **N:** the median sessions needed for unsettled cells.

All three are reported overall and by variable. Uncertainty comes from a subject bootstrap (500 resamples), with cells and BH recomputed each time; the power columns are held fixed.

## Decision rule (fixed now)
- **Reframe:** the paper is reframed around map reliability only if the upper 95% bounds of X and of Y are both below 50%.
- **Otherwise:** the current framing ("distorted in both directions") is kept, and X, Y and N are reported as one paragraph in the Results.

## Known limits, stated in advance
- **Variables:** the atlas covers task variables, not movement.
- **Power assumptions:** power is per-region pooled correlation at r = 0.1. Population decoders have more power, so Y is conservative for decoding-based maps.
- **Variable mix:** X depends on which variables the atlas includes, so per-variable values are always shown.

## Synthetic check (not evidence)
Two variables, 120 cells, 7 subjects: standard claims covered 98% of cells and valid claims 38%; X = 39% [35, 44] and Y = 58% [55, 62]; decision KEEP, because Y's upper bound is above 50%. The rule behaves as intended.

## Will not do
- Change q, r, the power threshold, the variables or the decision rule after seeing the result.
