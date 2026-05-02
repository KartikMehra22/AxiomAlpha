# Graph Analysis: Extracting Market Intelligence from Network Topology

## Research Purpose
|
| **1. Topology** | Is the market robust or fragile? Scale-free or random? | Structural metrics (density, path length, clustering) |
| **2. Community Detection** | Do assets cluster by GICS sector or by actual market behaviour? | Community assignments, NMI vs GICS |
| **3. Centrality Signals** | Does graph centrality predict future returns or risk? | Centrality-sorted return analysis |
| **4. Granger Intelligence** | Which assets lead price discovery and which follow? | Leader/Follower classification |
| **5. Systemic Risk** | Which assets are "too connected to fail"? | Traditional vs. network risk comparison |
| **6. ML Feature Matrix** | How do graph features combine with price features for ML? | `features_with_graph.csv` |

> **Data contract**: Reads all Phase 4 outputs from `data/processed/`. Final output: `features_with_graph.csv` — the complete ML-ready feature matrix combining price, macro, and graph features.

## Quantitative Logic & Methodology
## 1. Environment Setup & Data Loading

We load the complete multi-layer dataset generated in Phase 4:
- `log_returns.csv` — Price return matrix for graph reconstruction.
- `centrality_scores.csv` — Per-asset degree, betweenness, eigenvector, PageRank.
- `granger_causality.csv` — Directed edge list of significant causal relationships.
- `mst_edges.csv` — Minimum Spanning Tree backbone edges.
- `dynamic_graph_metrics.csv` — Time-series of network density, clustering, avg weight.
- `systemic_risk_scores.csv` — Composite Systemic Risk Score per asset.
- `risk_metrics.csv` — Traditional risk measures (VaR, volatility, GARCH params) from Phase 3.

The correlation graph `G` is reconstructed from the returns matrix using the same $\tau = 0.40$ threshold from Phase 4 for consistency.


## Key Mathematical Formulas
$$LR_i = \frac{\text{out-degree}_i - \text{in-degree}_i}{\text{out-degree}_i + \text{in-degree}_i + 1}$$

$$H_1: E[r_{t+1} | \text{high PR}] \neq E[r_{t+1} | \text{low PR}]$$

$$H_0: E[r_{t+1} | \text{high PR}] = E[r_{t+1} | \text{low PR}]$$

$$\text{Spike}_i^{(t)} = \mathbb{1}\left[\frac{C_i^{(t)} - \bar{C}_i^{(60d)}}{\hat{\sigma}_{C_i}^{(60d)}} > 2\right]$$

$$NMI(X, Y) = \frac{2 \cdot I(X;Y)}{H(X) + H(Y)} \in [0, 1]$$

$$Q = \frac{1}{2m} \sum_{ij} \left[A_{ij} - \frac{k_i k_j}{2m}\right] \delta(c_i, c_j)$$

