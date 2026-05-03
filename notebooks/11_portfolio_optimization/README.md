# NB-11: Portfolio Optimization

## Purpose
Solves a CVaR-constrained, long-only portfolio optimization combining four real signal sources via cvxpy. Produces per-date optimal weights that are consumed directly by the walk-forward backtest in NB12.

## Inputs
| Artifact | Producer | Loaded via |
|---|---|---|
| `log_returns` | NB02 | `src.io.loaders` |
| `master_signals` (alpha) | NB06/NB07 | `src.io.loaders` |
| `sentiment_features` | NB08 | `src.io.loaders` |
| `risk_metrics` | NB03 | `src.io.loaders` |
| `systemic_risk_scores` | NB04/NB05 | `src.io.loaders` |

## Outputs
| Artifact | Path | Schema | Consumer |
|---|---|---|---|
| `optimal_weights.csv` | `data/processed/optimal_weights.csv` | `Date, ticker, weight, mu_combined, expected_vol` | NB12 |

## Key methods
- `mu_historical = log_returns.mean() * 252`: annualized historical mean from real returns
- ML alpha avg per ticker (lagged 1 day before use)
- Sentiment avg per ticker (lagged 1 day before use)
- Reversal signal: `-log_returns.tail(20).mean() * 252`
- `LedoitWolf().fit(log_returns).covariance_ * 252`: shrunk annualized covariance
- cvxpy minimize: risk-aversion objective with `OPT_RISK_AVERSION` and per-asset cap `OPT_MAX_WEIGHT` from config

## How to run
```bash
python run_all.py
```

## Verified after fix
- ✅ No `np.random.*` data fabrication
- ✅ Reads exclusively via `src.io.loaders`
- ✅ Outputs schema-validated by downstream loaders
