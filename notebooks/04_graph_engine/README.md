# Graph Engine: Market as a Network

## Research Purpose
|
| **1. Correlation Graph** | Threshold-filtered undirected graph from Pearson ρ | Network visualisation + centrality scores |
| **2. Granger Causality** | Vector Autoregression F-tests on 870 directed pairs | Directed influence network |
| **3. Minimum Spanning Tree** | Kruskal's algorithm on correlation-distance matrix | Market backbone (noise-free skeleton) |
| **4. Dynamic Graphs** | Rolling 60-day correlation windows over time | Network density evolution vs VIX |
| **5. Risk Propagation** | SIR-style shock cascade on correlation graph | Systemic risk scores per asset |

> **Data contract**: Reads from `data/processed/log_returns.csv` and `data/processed/risk_metrics.csv`. Outputs: `centrality_scores.csv`, `granger_causality.csv`, `mst_edges.csv`, `dynamic_graph_metrics.csv`, `systemic_risk_scores.csv`.

## Quantitative Logic & Methodology
## 1. Environment Setup & Data Loading

Key library dependencies:

- `networkx` — Graph construction, centrality algorithms (degree, betweenness, eigenvector, PageRank), MST computation.
- `statsmodels` — Granger causality F-test via vector autoregression (VAR).
- `scipy` — Distance matrix operations for MST.

All visualisations use a **dark premium theme** (`#0d0d0d` background, 150 DPI) for publication-quality figures.


## Key Mathematical Formulas
$$r_{j,t} = \alpha_0 + \sum_{k=1}^{p} \alpha_k r_{j,t-k} + \sum_{k=1}^{p} \beta_k r_{i,t-k} + \epsilon_t$$

$$d_{ij} = \sqrt{2(1 - \rho_{ij})}$$

$$\alpha_{\text{adjusted}} = \frac{0.05}{870} \approx 5.7 \times 10^{-5}$$

$$\delta_j^{(t+1)} = \sum_{i \in N(j)} w_{ij} \cdot \delta_i^{(t)} \cdot (1 - \text{absorption}_j)$$

$$SRS_i = \alpha \cdot C_E(i) + \beta \cdot PR(i) + \gamma \cdot C_B(i) + \delta \cdot \text{PropagationImpact}(i)$$

$$r_{j,t} = \alpha_0 + \sum_{k=1}^{p} \alpha_k r_{j,t-k} + \epsilon_t$$

$$F = \frac{(RSS_R - RSS_U)/p}{RSS_U/(T - 2p - 1)}$$

$$w_{ij} = \rho_{ij} \quad \text{(larger weight = stronger relationship)}$$

$$\rho_{ij} = \frac{\text{Cov}(r_i, r_j)}{\sigma_i \cdot \sigma_j}$$

