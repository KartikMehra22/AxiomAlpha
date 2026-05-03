# NB-01: Data Pipeline

## Purpose
Ingests 5 years of daily prices and macro data via yfinance and FRED for the 30-ticker universe defined in `configs/settings.py`. This is the entry point of the pipeline — it fetches all raw data from external APIs, validates quality, and persists raw artifacts consumed by every downstream notebook.

## Inputs
| Artifact | Producer | Loaded via |
|---|---|---|
| yfinance API | External | `yfinance.download()` |
| FRED API | External | `fredapi.Fred()` |

## Outputs
| Artifact | Path | Schema | Consumer |
|---|---|---|---|
| `prices.csv` | `data/raw/` | `Date, <ticker columns>` | NB-02, NB-08 |
| `macro.csv` | `data/raw/` | `Date, VIX, SP500, TNX` | NB-02, NB-12 |
| `sector_map.json` | `data/raw/` | `{ticker: GICS_sector}` | NB-02, NB-04, NB-08 |

## Key methods
- yfinance bulk download: fetches all tickers in a single request with rate-limited fallback for reliability
- FRED API via `fredapi`: pulls VIX, SP500 index, and 10-year treasury yield (TNX)
- Quality validation: missing data heatmap and outlier detection to flag anomalies before downstream use

## How to run
```bash
python run_all.py            # full pipeline
```

## Verified after fix
- ✅ No `np.random.*` data fabrication
- ✅ Reads exclusively via `src.io.loaders`
- ✅ Outputs schema-validated by downstream loaders
