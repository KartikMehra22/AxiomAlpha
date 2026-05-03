# NB-05: Graph Analysis

## Purpose
Computes graph centrality, community structure, systemic risk scores, and Granger-causal leader/follower relationships from the network built in NB-04. Merges all graph-derived signals with price features to produce the unified `features_with_graph` ML-ready matrix.

## Inputs
| Artifact | Producer | Loaded via |
|---|---|---|
| `log_returns.csv` | NB-02 | `loaders.load_log_returns()` |
| `mst_edges.csv` | NB-04 | `loaders.load_mst_edges()` |
| `dynamic_graph_metrics.csv` | NB-04 | `loaders.load_dynamic_graph_metrics()` |
| `features_normalized.csv` | NB-02 | `loaders.load_features_normalized()` |

## Outputs
| Artifact | Path | Schema | Consumer |
|---|---|---|---|
| `centrality_scores.csv` | `data/processed/` | `ticker, degree, betweenness, eigenvector, pagerank` | NB-06, NB-11, NB-13 |
| `community_assignments.csv` | `data/processed/` | `ticker, community_id` | NB-06, NB-13 |
| `systemic_risk_scores.csv` | `data/processed/` | `ticker, systemic_risk_score` | NB-06, NB-11, NB-13 |
| `granger_causality.csv` | `data/processed/` | `source, target, f_stat, p_value` | NB-06, NB-14 |
| `leader_follower.csv` | `data/processed/` | `ticker, lf_score` | NB-13, NB-14 |
| `features_with_graph.csv` | `data/processed/` | `Date, ticker, <price+graph features>` | NB-06, NB-11, NB-13, NB-14 |
| `graph_intelligence.csv` | `data/processed/` | `ticker, <summary graph stats>` | NB-13 |

## Key methods
- Eigenvector centrality × volatility for systemic impact: amplifies centrality by realized vol to flag "too-connected-to-fail" assets
- Louvain community detection: maximizes modularity Q = (1/2m)·Σ[A_ij − k_i·k_j/2m]·δ(c_i, c_j)
- Granger causality tests: VAR-based F-tests on 870 directed pairs with Bonferroni-corrected α

## How to run
```bash
python run_all.py            # full pipeline
```

## Verified after fix
- ✅ No `np.random.*` data fabrication
- ✅ Reads exclusively via `src.io.loaders`
- ✅ Outputs schema-validated by downstream loaders
