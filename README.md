# AxiomAlpha

AxiomAlpha is an experimental quantitative-research pipeline for historical US equity data. It combines price and market-network features, walk-forward machine-learning predictions, portfolio construction and a monthly backtest. It is not a live trading system or evidence of future performance.

## Research flow

1. Notebooks 01–05 ingest prices and macro series, construct returns and features, and build rolling correlation networks.
2. Notebook 06 evaluates price-direction, forward-volatility and regime models using date-ordered folds. Notebook 07 produces out-of-sample direction probabilities with expanding training windows and a five-trading-day gap.
3. Notebooks 08–10 demonstrate sentiment and retrieval on **synthetic news**, some of which is generated using same-day returns. These outputs are not independent news evidence and are excluded from portfolio construction.
4. Notebook 11 and the walk-forward simulation in notebook 12 call the same `src.portfolio.construct_portfolio` function. It uses trailing historical returns, dated out-of-sample direction signals, a network penalty and a volatility-regime cash overlay. The probability tilt is an uncalibrated assumption, not a proven expected return. There is no CVaR constraint.
5. Notebook 14 summarizes artifacts and blocks its strategy decision if risk data is absent, a drawdown guardrail fails or the candidate allocation violates allocation bounds. This is a deterministic reporting workflow, not an LLM trading agent. Notebook 13 is an exploratory demonstration.

Signals are used only if dated at or before the rebalance and no more than ten trading days old. Each carries a training cutoff and must have a five-trading-day gap between its training and prediction dates. Models are refitted at the five walk-forward fold boundaries, not at every monthly rebalance, so a recent signal can still come from an older model. The strategy starts once complete out-of-sample signals are available, so its evaluation period can be shorter than the raw data period. Equal-weight and Markowitz baselines use that same test window. Returns include cash yield and estimated transaction costs; initial allocation costs are not charged.

## Run

```bash
pip install -r requirements.txt
python run_all.py      # Jupyter/papermill; rewrites executed notebooks
python run_all.py --direct  # isolated Python processes, no Jupyter kernel
python run_all.py --from 11 --through 15 --direct  # reuse upstream artifacts
python -m pytest
```

Notebook 01 downloads stocks and macro indicators from yfinance; it does not currently fetch FRED series. A full run requires external data and model access. `--direct` does not save notebook cell outputs. Commands write regenerated data under `data/`; preserve any existing artifacts before rerunning. The static check does not prove financial validity. Cached artifacts may reflect older code until their producing notebooks and all downstream stages have been rerun.

## Outputs and limitations

`data/processed/RESULTS.md` summarizes the last completed backtest, and `performance_metrics.csv`, `backtest_results.csv` and `attribution_results.csv` contain its supporting outputs. Daily attribution includes cash and transaction costs and sums to net daily return; adding daily contributions over time does not give compounded attribution.

The data currently covers a fixed, selected 30-stock universe and is subject to selection and survivorship bias; it is not a point-in-time constituents dataset. Results are sensitive to the sample, signal assumptions, cash exposure, transaction costs, model selection and network construction. Probability signals are not calibrated return forecasts. Validate with strong baselines, an untouched holdout, risk-matched comparisons and uncertainty estimates before making research or investment claims. Earlier notebook prose and plots may describe legacy experiments and should not be treated as the canonical methodology; executable code and regenerated outputs control.
