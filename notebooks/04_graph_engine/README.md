# NB-04: Graph Engine

## Purpose
Builds dynamic correlation networks from rolling 60-day return windows, extracts the Minimum Spanning Tree (MST) backbone, and tracks how graph topology evolves over time. These network artifacts expose systemic connectivity and contagion pathways that pure price-based features cannot capture.

## Inputs
| Artifact | Producer | Loaded via |
|---|---|---|
| `log_returns.csv` | NB-02 | `loaders.load_log_returns()` |
| `sector_map.json` | NB-01 | `loaders.load_sector_map()` |

## Outputs
| Artifact | Path | Schema | Consumer |
|---|---|---|---|
| `mst_edges.csv` | `data/processed/` | `source, target, weight` | NB-05, NB-13 |
| `dynamic_graph_metrics.csv` | `data/processed/` | `Date, density, path_length, clustering, modularity` | NB-05, NB-06, NB-13 |

## Key methods
- Pearson correlation → distance metric: d_ij = sqrt(2·(1 − ρ_ij)) maps correlation to a proper metric space
- Kruskal MST: extracts the minimum spanning tree over the distance matrix to form the market backbone
- networkx for community detection: Louvain algorithm for modularity-maximizing community partitions

## How to run
```bash
python run_all.py            # full pipeline
```

## Verified after fix
- ✅ No `np.random.*` data fabrication
- ✅ Reads exclusively via `src.io.loaders`
- ✅ Outputs schema-validated by downstream loaders
