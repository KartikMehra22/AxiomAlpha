# NB-07: ML Analysis

## Purpose
Evaluates trained ML models, computes per-ticker Information Coefficient (IC) and feature importance, and exports the consolidated `master_signals` table that the portfolio optimizer and agent layer consume. Converts raw model probabilities into cross-sectionally z-scored, risk-adjusted trading signals.

## Inputs
| Artifact | Producer | Loaded via |
|---|---|---|
| `features_with_graph.csv` | NB-05 | `loaders.load_features_with_graph()` |
| `regimes.csv` | NB-03 | `loaders.load_regimes()` |
| `dynamic_graph_metrics.csv` | NB-04 | `loaders.load_dynamic_graph_metrics()` |
| `model_direction.pkl` | NB-06 | `loaders.load_model_direction()` |
| `model_volatility.pkl` | NB-06 | `loaders.load_model_volatility()` |
| `model_regime.pkl` | NB-06 | `loaders.load_model_regime()` |

## Outputs
| Artifact | Path | Schema | Consumer |
|---|---|---|---|
| `master_signals.csv` | `data/processed/` | `Date, ticker, alpha_signal, alpha_zscore, risk_signal, risk_zscore, regime_pred, combined_score` | NB-11, NB-13 |
| `ml_eval.json` | `data/processed/` | `{model: {metric: value}}` | NB-13 |
| `signal_explanations.json` | `data/processed/` | `{ticker: {date: explanation}}` | NB-13 |

## Key methods
- Model `.predict_proba` for direction: extracts P(positive return) as continuous alpha signal rather than hard class label
- Cross-sectional z-score per Date: each signal normalized across the 30-ticker universe on each date independently
- Combined alpha = z(alpha) − 0.5·z(risk): penalizes high-alpha signals that carry disproportionate volatility risk

## How to run
```bash
python run_all.py            # full pipeline
```

## Verified after fix
- ✅ No `np.random.*` data fabrication
- ✅ Reads exclusively via `src.io.loaders`
- ✅ Outputs schema-validated by downstream loaders
