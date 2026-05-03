# NB-02: Data Processing

## Purpose
Cleans raw price data, computes log and simple returns, and builds normalized feature matrices used by every downstream modelling notebook. Applies only forward-fill imputation to prevent future-data leakage and produces cross-sectionally z-scored features via sklearn StandardScaler.

## Inputs
| Artifact | Producer | Loaded via |
|---|---|---|
| `prices.csv` | NB-01 | `loaders.load_prices()` |
| `macro.csv` | NB-01 | `loaders.load_macro()` |
| `sector_map.json` | NB-01 | `loaders.load_sector_map()` |

## Outputs
| Artifact | Path | Schema | Consumer |
|---|---|---|---|
| `log_returns.csv` | `data/processed/` | `Date, <ticker columns>` | NB-03, NB-04, NB-05, NB-06, NB-07, NB-08, NB-09, NB-10, NB-11, NB-12 |
| `simple_returns.csv` | `data/processed/` | `Date, <ticker columns>` | NB-11 |
| `features_raw.csv` | `data/processed/` | `Date, ticker, <feature columns>` | NB-06 |
| `features_normalized.csv` | `data/processed/` | `Date, ticker, <feature columns>` | NB-05, NB-06, NB-07 |
| `feature_scaler.pkl` | `data/processed/` | sklearn StandardScaler | NB-06, NB-07 |

## Key methods
- `ffill()` only (no bfill): eliminates future-data leakage during gap imputation
- Rolling features `.shift(1)`: all predictors shifted one day forward for predictor safety
- Per-date cross-sectional normalization: z-scores relative to cross-section of 30 tickers on each date
- sklearn StandardScaler: persisted for consistent transform in inference notebooks

## How to run
```bash
python run_all.py            # full pipeline
```

## Verified after fix
- ✅ No `np.random.*` data fabrication
- ✅ Reads exclusively via `src.io.loaders`
- ✅ Outputs schema-validated by downstream loaders
