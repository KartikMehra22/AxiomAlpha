# NB-14: Agent Orchestration

## Purpose
Wires the 7 agents defined in NB13 into an autonomous LangGraph state machine; produces a final pipeline output JSON. State transitions are driven by real CSV data; no random fabrication is used at any node.

## Inputs
| Artifact | Producer | Loaded via |
|---|---|---|
| `master_signals` | NB06/NB07 | `src.io.loaders` |
| `optimal_weights` | NB11 | `src.io.loaders` |
| `regimes` | NB03 | `src.io.loaders` |
| `risk_metrics` | NB03 | `src.io.loaders` |
| `sentiment_features` | NB08 | `src.io.loaders` |
| `centrality_scores` | NB04/NB05 | `src.io.loaders` |
| `systemic_risk_scores` | NB04/NB05 | `src.io.loaders` |
| `performance_metrics` | NB12 | `src.io.loaders` |
| `attribution_results` | NB12 | `src.io.loaders` |

## Outputs
| Artifact | Path | Schema | Consumer |
|---|---|---|---|
| `pipeline_output.json` | `data/processed/pipeline_output.json` | `{tickers, period, latest_regime, optimal_weights, sharpe, per_agent_state}` | NB15 |

## Key methods
- LangGraph state machine: agents called sequentially through typed graph edges
- State populated exclusively from real CSV loaders (no `np.random.randn` fabrication)
- Conditional routing: DataAgent quality check gates downstream execution

## How to run
```bash
python run_all.py
```

## Verified after fix
- ✅ No `np.random.*` data fabrication
- ✅ Reads exclusively via `src.io.loaders`
- ✅ Outputs schema-validated by downstream loaders
