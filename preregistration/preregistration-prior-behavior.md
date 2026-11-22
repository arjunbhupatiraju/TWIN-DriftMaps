# Pre-registration B1: do the weak, hidden prior codes matter for behavior?

Registered 2026-11-06, before this analysis was run on real data. Code: `part6/prior_followup.py behavior`, SHA-256 prefix a35c296b2bfcd8e3, tested only on mock data. Input: the REGISTERED prior-map run (`results_prior/prior_rows.csv`). Any change is recorded as a dated amendment.

## Question
The prior-map run found that the block prior is carried far more widely than detected, but mostly at weak code sizes. A reviewer will ask whether such weak codes relate to behavior. B1 asks whether sessions whose recorded regions carry more prior information are sessions where the mouse's choices follow the prior more.

## Measures (fixed)
- **Neural prior information, per session.** The mean over the session's regions of the decoder score minus that region's pseudo-session null mean. This is the excess over chance, so drift and session structure are already subtracted.
- **Behavioral prior use, per session.** The prior weight of a logistic choice model fitted to trials after the first 90. Its inputs are the signed stimulus, the Bayes-optimal prior, and 4 lags each of choice and rewarded choice (TWIN's ChoiceModel, ridge λ = 1). The weight is signed by the stimulus weight, so following the prior is positive.
- **Covariates:** the number of regions, the mean number of units, and the number of trials.

## Test (fixed)
- **Statistic:** the partial Spearman correlation between neural and behavioral prior measures, after rank-regressing both on the covariates.
- **Null:** 5,000 permutations of behavior across sessions within each lab. This removes lab-level confounds such as rig, training protocol and probe placement.
- **B1 PASS** if ρ > 0 and the one-sided permutation p < 0.05.

Descriptive, with no pass/fail:
- the within-mouse rank correlation for mice with 3 or more sessions;
- the distribution of behavioral prior weights.

## Interpretation
| Outcome | Reading |
|---|---|
| PASS | Sessions whose brains carry more prior information are the sessions where the mouse uses the prior more. The hidden codes are behaviorally relevant, not noise. This is correlational. Because choice history drives both the prior and behavior, it shows prior information that goes with prior use, not a separate causal pathway. |
| FAIL | Across sessions, the amount of decodable prior does not track behavioral prior use. This would be reported. It limits the claim to "the prior is represented", not "it is used". |

## Will not do
- Change the measures, covariates, permutation scheme or criterion after seeing results.
- Swap in the amended run or select regions.

The amended run may be reported as a descriptive robustness check only.

## Amendments
(none)
