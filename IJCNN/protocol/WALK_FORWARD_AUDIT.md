# Annual walk-forward development study

**Status:** Exploratory, inspected development data, not an independent holdout or paper-quality result. Date: 21 November 2026. Uses the same source snapshot and after-close, equal-weight overlapping-universe proxy specified in `PILOT_AUDIT.md`. This does not evaluate a graph, GNN, calibration, portfolio or investment strategy.

## Setup

`src/walk_forward.py` reuses `pilot.construct`: 20-trading-day future RMS of the market-proxy log returns and past-only 5/20/60-day volatility, 20-day return, and signed 60-day average stock correlation. Each annual fold uses an expanding training history; all training labels mature before the evaluation year begins. Evaluation labels mature within that same year, so dates near year end without a fully matured label are omitted. The 80th percentile target threshold is computed using only matured training outcomes and fixed during each evaluation year. Thus the definition of stress differs across folds; do not pool the labels as if a single economic threshold applied throughout.

Model specifications are fixed without tuning: training-prevalence probability; logistic models with volatility/return features, with and without correlation; histogram gradient-boosted classifiers with identical feature sets and fixed hyperparameters. No calibration. The 2022 cohort and period were already inspected in the previous pilot; all 2022–2024 results are development analyses, not a final holdout. Existing AxiomAlpha reports also examined 2023–2024. Price history remains a fixed, selected 30-stock universe with unresolved source, adjustment and survivorship limitations. Same-universe market proxy overlaps the features.

## Observations

| Model | 2022 AP | 2022 Brier | 2023 AP | 2024 AP |
|---|---:|---:|---:|---:|
| Training prevalence | 0.3939 | 0.2760 | Not defined | Not defined |
| Logistic: volatility | 0.4632 | 0.2628 | Not defined | Not defined |
| Logistic: volatility + correlation | 0.4154 | 0.2757 | Not defined | Not defined |
| Boosted: volatility | 0.4040 | 0.3583 | Not defined | Not defined |
| Boosted: volatility + correlation | 0.5813 | 0.3019 | Not defined | Not defined |

The 2022 development fold has 91 positives among 231 dates (39.4%). The 2023 and 2024 evaluation folds have **zero positives** under their respective training-derived thresholds, among 230 and 231 eligible dates. AP and ROC-AUC are reported as null for a single-class fold; Brier is still computable but cannot show the ability to detect an event that did not occur. Do not infer generalization or select a best model from these results. The event threshold changes from 0.015657 in 2022 to 0.016712 in 2023 and 0.015483 in 2024, in unannualized daily log-return RMS units. This demonstrates the event definition and short sample may be unsuitable for comparing ranking across these annual folds.

2022 logistic correlation-minus-volatility average-precision difference is approximately -0.0478. An exploratory paired moving-block percentile interval (block length 25 observations, 1,000 seeded resamples) is approximately [-0.116, 0.024]. It is conditional on this short, previously inspected fold and fixed predictions, does not reflect training/model-selection uncertainty, and is sensitive to block length and few clusters of stress labels. It is not proof of equivalence or a confirmatory significance test. AP bootstrap replicates with only one class are excluded and counted; none were valid in the zero-event folds.

The 2022 AP difference between boosted models changes in the opposite direction, while their Brier scores differ. This illustrates why model family, metric, probability calibration and evaluation period matter. There has been no tuning or uncertainty analysis for that comparison; do not infer correlation's incremental value from it.

## Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python IJCNN/src/walk_forward.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python -m pytest IJCNN/tests -q
```

Outputs are overwritten under `IJCNN/results/walk_forward/`; change with `--output`. Bootstrap repetitions can be changed with `--bootstrap-repetitions`. The JSON records input hash, versions, fold sample sizes, thresholds, event frequencies, metrics, and exploratory interval. No existing notebook or `data/` artifact is modified.

## Decision before next model

Do not train or claim an improvement from a GNN yet. First acquire longer, well-documented data and choose a meaningful stress-event definition using development data only. Evaluate whether thresholds frozen on a training fold or updated from past matured labels answer the desired question; changing the rule changes the target and must be explicitly documented. Count separate episodes and check enough positives in each evaluation period. Predefine the target, folds, threshold policy, primary metric, baselines and uncertainty procedure before opening a genuinely uninspected independent test. Do not tune the threshold to make 2023 or 2024 appear favorable.
