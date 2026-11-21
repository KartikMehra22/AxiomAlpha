# IJCNN pilot: data audit, protocol and results

**Status:** Exploratory development run; not a preregistered experiment, independent holdout, paper result, statistical significance test, graph-neural result, or investment recommendation. 2022 has now been inspected and cannot be treated as an untouched holdout. The project's 2023–2024 history has previously been examined elsewhere and is not claimed as untouched either. Prepared 21 November 2026.

## 1. What was audited

Input: `data/raw/prices.csv`, SHA-256 `446483c93d9c590d2349b25714d212fc7e66d9d9e1d6d1e110499edc9475a321`.

- 1,509 daily price rows, 30 stocks, 2019-01-02 through 2024-12-30; no missing values, duplicate dates or non-positive prices observed in this snapshot.
- Recomputed `log(price_t / price_{t-1})` has the same index and columns as `data/processed/log_returns.csv`; maximum absolute numeric difference was approximately `9.99e-17`. This is a numerical consistency check, not proof that source prices are point-in-time accurate.
- Notebook 01 currently requests `yfinance` downloads with `auto_adjust=True`; this documents current code, **not proof of when or how the existing file was acquired**, its data-provider revision history, or historical point-in-time adjustment factors. Verify snapshot lineage and availability before making a paper claim.
- The stock set is a fixed selected universe, not verified point-in-time index membership. Survivorship and selection bias remain; yfinance data provenance, rights, corporate actions, venue closes, delistings and publication availability require further work.
- Network artifacts in the project have different frequencies, and exploratory notebooks construct some full-history graphs. This pilot constructs no graph and does not use cached graph features.
- Current environment: Python 3.13.5; NumPy 1.26.4; pandas 2.2.3; scikit-learn 1.6.1. These are observed runtime versions, not a claim of compliance with all repository dependency pins.

## 2. Target and forecast timing

At the close of day `t`, compute equal-weight arithmetic return across the 30 stocks (as if rebalanced daily), then `log1p` of that average. This overlapping-universe proxy is **not the S&P 500** and is not a separately observed tradable index. It assumes synchronous availability; trading, rebalancing and data-availability costs are not modeled.

The target is RMS of proxy log returns for the next 20 trading dates, `t+1..t+20`. Only rows whose complete target is observed are retained. A positive label means this RMS exceeds the 80th percentile calculated from **fully matured training targets only**. The frozen training threshold was `0.0156574257382859` in per-day log-return units, not annualized.

Historical features at `t`: rolling RMS of proxy returns over 5, 20 and 60 days and cumulative 20-day proxy log return. A second logistic model adds the trailing 60-day mean of the signed Pearson correlations for unique stock pairs. Features include day `t` and thus require an after-close cutoff. No graph, neural network, hyperparameter search, calibration, bootstrap, trading model or synthetic news is used.

## 3. Development split and safeguards

- Training: eligible feature dates with labels ending on or before 2021-12-31, 677 observations.
- Validation: eligible forecast dates in 2022 whose labels mature by 2022-12-31, 231 observations. Cross-boundary labels are excluded; later dates are not scored.
- Fitted scaler and logistic coefficients use training only. Validation is a development period. The threshold is held fixed from training.
- Tests check future-price mutation cannot change earlier features, exact target alignment, incomplete target exclusion, matured-label splits and training-threshold independence from validation outcomes. They do not prove all possible leakage is absent.
- This is one historical training/validation split with overlapping 20-day labels, no untouched final evaluation and no uncertainty intervals. Correlated daily labels must not be treated as independent evidence.

## 4. Development results

Training positives: 136/677 (20.09%); validation positives: 91/231 (39.39%). There are nine contiguous runs of positive daily labels in this validation window; runs are only a rough description, not a validated count of independent crises.

| Model | Average precision | ROC-AUC | Brier score |
|---|---:|---:|---:|
| Historical training prevalence | 0.3939 | 0.5000 | 0.2760 |
| Logistic: volatility and return | 0.4632 | 0.6046 | 0.2628 |
| Logistic: volatility, return and mean correlation | 0.4154 | 0.5754 | 0.2757 |

For average precision, the constant-score baseline equals the 2022 evaluation prevalence, not the training prevalence used as its probability. Its Brier score uses the training prevalence. Brier is lower-is-better; AP and ROC-AUC are higher-is-better. Differences are descriptive only: no confidence interval or statistically defensible claim is computed. Models are fixed specifications, uncalibrated, and may be sensitive to the selected training dates, 2020 volatility, regularization and 2022 regime. Correlation worsening this one development comparison does **not** establish that correlation or graph features are useless. Nothing here assesses graph topology or GNNs.

## 5. Reproduction

From repository root, with dependencies already installed:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python IJCNN/src/pilot.py
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python -m pytest IJCNN/tests/test_pilot.py tests/test_pipeline_static.py -q
```

For portability substitute an environment's Python interpreter with compatible NumPy, pandas and scikit-learn. Outputs `IJCNN/results/pilot/validation_metrics.json` and `validation_predictions.csv` are regenerated and overwritten by this command. `--prices` and `--output` permit an alternate input and output directory. The script does not modify `data/` or existing notebooks. Record dataset SHA, configuration, dependencies and code revision for any run used in research.

## 6. Next decisions

1. Verify source history, adjustment and permissible usage; secure longer data and a genuinely uninspected test period or independent dataset before claim-making.
2. Obtain legally accessible full texts for the two publisher-restricted references; record uncertainty and gaps without claiming omissions.
3. Predefine walk-forward validation folds and primary/secondary analyses before further model selection; measure uncertainty with a justified time-block bootstrap and report distinct stress episodes.
4. Add HAR-inspired/persistence and strong nonlinear baselines and only then a non-graph temporal neural model; compare on identical dates and compute budgets.
5. Build past-only graph snapshots and null controls in a separate experiment; compare fixed and rolling topology before proposing a reliability-aware gate.
6. Treat 2022 results above as exploratory. No final IJCNN novelty or performance claim is established.
