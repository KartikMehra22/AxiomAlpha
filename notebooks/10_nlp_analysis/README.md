# NB-10: NLP Analysis

## Purpose
Computes Information Coefficient (IC) of sentiment signals against forward returns, with 1-day execution lag (no lookahead). Validates that NB08 sentiment features carry predictive power at a 1-day horizon before they are passed to the portfolio optimizer.

## Inputs
| Artifact | Producer | Loaded via |
|---|---|---|
| `sentiment_features` | NB08 | `src.io.loaders` |
| `log_returns` | NB02 | `src.io.loaders` |
| `regimes` | NB03 | `src.io.loaders` |

## Outputs
| Artifact | Path | Schema | Consumer |
|---|---|---|---|
| `nlp_metrics.json` | `data/processed/nlp_metrics.json` | `{ic_lagged, ic_per_ticker, n_tickers, sentiment_lag_days}` | NB11 (informational), NB12 (RESULTS.md) |

## Key methods
- `sentiment_score.shift(1)` per ticker before merge with returns; enforces 1-day execution lag eliminating lookahead bias
- Pearson IC per ticker: `corr(sentiment_lagged, forward_return)`
- Mean IC across tickers reported as `ic_lagged`

## How to run
```bash
python run_all.py
```

## Verified after fix
- ✅ No `np.random.*` data fabrication
- ✅ Reads exclusively via `src.io.loaders`
- ✅ Outputs schema-validated by downstream loaders
