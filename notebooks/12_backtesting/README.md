# NB-12: Backtesting

## Purpose
Walk-forward backtest applying NB11's optimal weights to real returns; computes Sharpe, drawdown, attribution; auto-generates RESULTS.md. Validates the full pipeline end-to-end against actual price data with no lookahead and realistic transaction costs.

## Inputs
| Artifact | Producer | Loaded via |
|---|---|---|
| `log_returns` | NB02 | `src.io.loaders` |
| `simple_returns` | NB02 | `src.io.loaders` |
| `optimal_weights` | NB11 | `src.io.loaders` |
| `regimes` | NB03 | `src.io.loaders` |
| `systemic_risk_scores` | NB04/NB05 | `src.io.loaders` |
| `macro` | NB01 | `src.io.loaders` |

## Outputs
| Artifact | Path | Schema | Consumer |
|---|---|---|---|
| `backtest_results.csv` | `data/processed/backtest_results.csv` | `Date, strategy, portfolio_return, cumulative_return` | NB13, NB14, NB15 |
| `equity_curves.csv` | `data/processed/equity_curves.csv` | `Date, strategy, equity` | NB15 |
| `performance_metrics.csv` | `data/processed/performance_metrics.csv` | `strategy, sharpe, max_drawdown, cagr, sortino` | NB13, NB14, NB15 |
| `attribution_results.csv` | `data/processed/attribution_results.csv` | `Date, ticker, contribution` | NB14, NB15 |
| `regime_performance.csv` | `data/processed/regime_performance.csv` | `regime, strategy, mean_return, sharpe` | NB15 |
| `RESULTS.md` | `data/processed/RESULTS.md` | human-readable summary | none |
| `RESULTS.json` | `data/processed/RESULTS.json` | machine-readable metrics | none |

## Key methods
- Walk-forward loop with explicit assertion: weights at time t are sourced from data up to t-1 only
- 5 bps transaction cost applied on weight turnover at each rebalance
- Equal-weight and Markowitz baselines computed for comparison
- Monte Carlo stress paths using `np.random.standard_t` (legitimate use, allowlist-tagged) for tail-risk simulation

## How to run
```bash
python run_all.py
```

## Verified after fix
- ✅ No `np.random.*` data fabrication
- ✅ Reads exclusively via `src.io.loaders`
- ✅ Outputs schema-validated by downstream loaders
