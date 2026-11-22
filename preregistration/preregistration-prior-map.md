# Pre-registration: PRIOR MAP — how brain-wide is the block prior, really?

Registered 2026-11-05, before any real data were analysed with this pipeline. Code: `part6/prior_map.py`, SHA-256 prefix 996bb383f2c4075d (frozen at registration; tested only on mock data). Any change after the first real run is recorded below as a dated amendment, and both versions are reported.

## Question
The IBL Brain-Wide Map work (Findling et al., Nature 2025) reported that the block prior can be decoded from pre-stimulus activity across much of the brain. A region where the prior is not detected may lack it, or the test there may be too weak to find it. A region where it is detected may not hold up in other labs. TWIN can separate these cases, because it measures each recording's own power to detect a prior code.

We ask three things:
1. What fraction of brain regions truly carry the prior, after correcting for each region's power?
2. Which regions are unanswerable with existing data?
3. Can TWIN predict, without seeing any outcomes, which labs will reproduce a region's prior signal?

## Data
- All cached IBL Brain-Wide Map sessions (the same cache as the atlas).
- Per session × region, per-trial spike counts in the window from stimOn −600 ms to −100 ms.
- Trials kept: those with a choice and a valid stimOn time, excluding the first 90 trials (unbiased block).
- At least 150 trials, and at least 10 units with variance; regions with more than 150 units are randomly subsampled to 150.

## Methods (fixed)
**Target.** The Bayes-optimal prior: a forward filter over block identity, using the IBL block statistics (0.2/0.8, mean block length 60), updated only on trials with nonzero contrast. The secondary target is the true block probability.

**Decoder.** Ridge regression decoded with 5 contiguous cross-validation folds:
- units are z-scored within each training fold;
- the ridge strength is chosen per target by GCV over 9 values (10^-3 to 10^2 × the number of training trials);
- the score is the Pearson r between held-out predictions and the target.

**Valid test.** The score against 200 pseudo-sessions drawn from the IBL task generator, with each pseudo prior computed the same way.

**Comparison test (invalid, for reference only).** The score against 200 trial shuffles.

**Drift twins.** 40 multivariate phase-randomized twins per session-region. These keep each neuron's autocorrelation and the cross-neuron correlations, and contain no prior code by construction.

**Power.** For each per-neuron r on the TWIN grid (0.02 to 0.3), we plant 12 NEW pseudo-session priors (never the real prior) into a random 20% of the real neurons, then run the valid test.

**Region level:**
- Fisher-combine the valid p across a region's sessions, then apply Benjamini–Hochberg across regions.
- For the power-corrected estimate, each region has one code size shared by its sessions. A session's likelihood under each code size comes from its own planted p-values (Gaussian kernel on log10 p, bandwidth 0.35; uniform for r = 0). The brain-wide distribution of code size across regions is fitted by EM on the grid. Confidence intervals come from a region bootstrap (500 resamples).

## Gate G1: is the estimator valid on these recordings? (must pass before H3 is read)
Both checks use only the real recordings' own held-out semi-synthetic results:
- (a) Every session's observed p is replaced by a held-out pseudo-session target's p (no code anywhere). The estimated fraction with r ≥ 0.04 must be **≤ 0.10**.
- (b) In a random 30% of regions, all sessions use a fresh planted code at r = 0.1; the rest use the held-out null. The estimate must be **within 0.20–0.40**.

If G1 fails, H3 is reported but not interpreted.

## Hypotheses and criteria
**H1 (validity of the decoder test).** On drift twins, the pseudo-session decoder test fires at **≤ 0.07**, with the session-bootstrap 95% upper bound **≤ 0.08**. The trial-shuffle rate is reported alongside.

**H2 (descriptive, no pass/fail).** We report the fraction of regions significant at q = 0.05 and q = 0.01 under the valid null and under the shuffle null, and the single-session significance rates. Our decoder differs from the IBL pipeline, so this is not a replication of their numbers.

**H3 (main finding).** Let π be the estimated fraction of regions with code size r ≥ 0.04, with its 95% CI:
- BRAIN-WIDE if the lower bound is above 0.5;
- LOCALIZED if the upper bound is below 0.5;
- otherwise INCONCLUSIVE.

We also report:
- the fractions at r ≥ 0.02 and r ≥ 0.1;
- the detection gap (π minus the detected fraction), with its CI;
- per region, the posterior probability that it carries the prior;
- the regions that are undetected but probably carry it (posterior > 0.5);
- the regions that are unanswerable with existing data (posterior 0.2–0.8).

We make no directional prediction. Every outcome changes how the prior map should be read.

**H4 (zero-shot replication of the prior map).**
- Claims are each lab's Fisher-combined valid p per region.
- Attempts are the same region in every other lab; success means that lab's Fisher p is below 0.05.
- TWIN v3 (fitted only on claim p-values, never on outcomes) must reach **AUC ≥ 0.65** and **beat the original p**, with the original p's predictor fitted leaving out the lab being predicted. The lab-bootstrap 95% CI of the AUC difference must lie above 0.
- TWIN v1, TWIN v2 and naive power are reported alongside.

## What each outcome means
| Outcome | Reading |
|---|---|
| H3 BRAIN-WIDE, large positive gap | The prior is even more widespread than detected. Many regions IBL could not confirm are underpowered, not empty. |
| H3 LOCALIZED | The prior is concentrated. A "brain-wide" description over-reads the map, and TWIN names which regions hold it. |
| H3 INCONCLUSIVE | The current data cannot settle how brain-wide the prior is. The sample-size map says what would. |
| H1 FAIL | The decoder test is not valid on these recordings. H2–H4 are reported but not interpreted. |
| H4 PASS / FAIL | Whether TWIN can say in advance which parts of the prior map will reproduce. |

## Will not do
- Change the target, window, decoder, grid, thresholds or criteria after seeing real results.
- Drop sessions or regions based on their results.
- Report subsets selectively.

## Amendments
**Amendment 1 (2026-11-05, after the registered run; a precision fix only, and both runs are reported).**

The registered run's G1 check showed that the estimator runs low: a planted 30% was recovered as 24.9%. A simulation on synthetic data, never on real results, traced this to Monte Carlo noise from using only 12 planted codes per code size. With 12 plants, a true fraction of 0.514 was recovered as 0.448; with 48 plants and 1,000 nulls, it was recovered as 0.502, with confidence intervals of the same width.

The amended run therefore changes only the Monte Carlo sizes: 48 plants per code size and 1,000 pseudo-session nulls (the shuffle null scales with it). Nothing else changes: target, window, decoder, grid, kernel, thresholds and all criteria.

H1 is taken from the registered run, because the amended run uses 1 twin per region to save time. G1, H2, H3 and H4 are re-evaluated with the same criteria. The registered run remains the primary result. If the two runs disagree on any verdict, both are reported and the registered verdict stands as the confirmatory one.

The code differs only by a `--tag` option for separate output files (SHA-256 prefix 67b0a273f7d81406). The command is:

`prior_map.py run --plant 48 --null 1000 --twins 1 --tag amend1`
