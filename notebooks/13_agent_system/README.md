# NB-13: Agent System

## Purpose
Defines 7 specialized AI agents (Data, Quant, Graph, ML, NLP, Risk, Strategy) — each reads its assigned upstream CSVs through loaders. Agent state is derived deterministically from real upstream data; no random fabrication occurs.

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

## Outputs
| Artifact | Path | Schema | Consumer |
|---|---|---|---|
| Agent class definitions | (in-memory) | Python classes | NB14, NB15 |

No CSV artifacts are written by this notebook. The agent classes defined here are imported and instantiated by NB14 and NB15.

## Key methods
- Each agent class loads its data slice via `src.io.loaders` at instantiation
- Templated explanation methods generate human-readable summaries without an LLM call
- Deterministic state: all agent attributes derived from real upstream CSVs

## How to run
```bash
python run_all.py
```

## Verified after fix
- ✅ No `np.random.*` data fabrication
- ✅ Reads exclusively via `src.io.loaders`
- ✅ Outputs schema-validated by downstream loaders
