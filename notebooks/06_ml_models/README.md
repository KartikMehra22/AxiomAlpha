# NB-06: ML Models

## Purpose
Trains direction (binary classification), volatility (regression), and regime (multiclass) models using a strict walk-forward TimeSeriesSplit to prevent future-data leakage. Produces persisted model artifacts and the `ml_signals` table consumed by NB-07 and the agent layer.

## Inputs
| Artifact | Producer | Loaded via |
|---|---|---|
| `features_with_graph.csv` | NB-05 | `loaders.load_features_with_graph()` |
| `regimes.csv` | NB-03 | `loaders.load_regimes()` |
| `dynamic_graph_metrics.csv` | NB-04 | `loaders.load_dynamic_graph_metrics()` |

## Outputs
| Artifact | Path | Schema | Consumer |
|---|---|---|---|
| `ml_signals.csv` | `data/processed/` | `Date, ticker, direction_prob, vol_forecast, regime_pred` | NB-07, NB-13 |
| `model_direction.pkl` | `data/processed/` | sklearn-compatible estimator | NB-07, NB-13 |
| `model_volatility.pkl` | `data/processed/` | sklearn-compatible estimator | NB-07, NB-13 |
| `model_regime.pkl` | `data/processed/` | sklearn-compatible estimator | NB-07, NB-13 |

## Key methods
- TimeSeriesSplit (5 folds) with explicit walk-forward assertion: each fold's test set is strictly after its train set; assertion raises if chronological order is violated
- RandomForest, XGBoost, Ridge baselines: ensemble and linear models compared per task; best selected by held-out metric
- Regime feature lagged 1 day: `regimes.shift(1)` applied before joining to features, preventing same-day leakage
- Targets: next-day return direction (y_dir), next-day realized vol (y_vol), next-day regime (y_regime)

## How to run
```bash
python run_all.py            # full pipeline
```

## Verified after fix
- ✅ No `np.random.*` data fabrication
- ✅ Reads exclusively via `src.io.loaders`
- ✅ Outputs schema-validated by downstream loaders
