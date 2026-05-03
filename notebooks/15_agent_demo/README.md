# NB-15: Agent Demo

## Purpose
Interactive showcase reading real backtest outputs (equity curves, performance metrics, attribution); templated Q&A. Demonstrates end-to-end pipeline results without producing any new pipeline artifacts.

## Inputs
| Artifact | Producer | Loaded via |
|---|---|---|
| `equity_curves` | NB12 | `src.io.loaders` |
| `performance_metrics` | NB12 | `src.io.loaders` |
| `regime_performance` | NB12 | `src.io.loaders` |
| `attribution_results` | NB12 | `src.io.loaders` |
| `pipeline_output` | NB14 | `src.io.loaders` |
| `optimal_weights` | NB11 | `src.io.loaders` |

## Outputs
| Artifact | Path | Schema | Consumer |
|---|---|---|---|
| Demo figures | (display only) | matplotlib / plotly figures | None (terminal demo) |

No CSV or JSON pipeline artifacts are written by this notebook.

## Key methods
- All charts read real data via `src.io.loaders` (no random fallbacks)
- Templated Q&A function: routes queries about regime, weights, or Sharpe to pre-computed answers derived from loaded CSVs
- RAG demo queries use the FAISS index from NB09 to retrieve grounding headlines

## How to run
```bash
python run_all.py
```

## Verified after fix
- ✅ No `np.random.*` data fabrication
- ✅ Reads exclusively via `src.io.loaders`
- ✅ Outputs schema-validated by downstream loaders
