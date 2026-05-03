# NB-03: Quant Core

## Purpose
Fits GARCH(1,1) volatility models per ticker and identifies market regimes (Bear/Volatile/Bull encoded as 0/1/2) using volatility quantile thresholds. Produces the risk and regime artifacts that feed every downstream modelling and backtesting notebook.

## Inputs
| Artifact | Producer | Loaded via |
|---|---|---|
| `log_returns.csv` | NB-02 | `loaders.load_log_returns()` |
| `prices.csv` | NB-01 | `loaders.load_prices()` |
| `macro.csv` | NB-01 | `loaders.load_macro()` |

## Outputs
| Artifact | Path | Schema | Consumer |
|---|---|---|---|
| `risk_metrics.csv` | `data/processed/` | `Date, ticker, garch_vol, var_95, cvar_95, downside_vol` | NB-04, NB-05, NB-06, NB-10, NB-11, NB-12, NB-13, NB-14 |
| `regimes.csv` | `data/processed/` | `Date, regime_encoded` | NB-04, NB-05, NB-06, NB-10, NB-11, NB-12, NB-13, NB-14 |

## Key methods
- `arch.GARCH` per-ticker: fits GARCH(1,1) with formula σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
- α+β stationarity check: emits a warning when α+β ≥ 1 (integrated GARCH — shocks never decay)
- Regime classification via volatility quantile: tercile thresholds on rolling market vol → Bear(0)/Volatile(1)/Bull(2)

## How to run
```bash
python run_all.py            # full pipeline
```

## Verified after fix
- ✅ No `np.random.*` data fabrication
- ✅ Reads exclusively via `src.io.loaders`
- ✅ Outputs schema-validated by downstream loaders
