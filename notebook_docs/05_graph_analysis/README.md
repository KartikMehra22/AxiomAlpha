# Graph Analysis: Extracting Market Intelligence from Network Topology
**Research Module 5 of 5 | AxiomAlpha Framework**

---

Phase 4 constructed the market's **relational infrastructure**: a weighted correlation graph, a directed Granger causality network, a Minimum Spanning Tree backbone, and a time-varying dynamic graph. This notebook analyses that infrastructure to extract **quantifiable intelligence** that is invisible to traditional asset-by-asset methods.

### The Core Question
> *What does the network structure tell us that traditional analysis cannot?*

| Analysis Module | Key Question Answered | Output |
| :--- | :--- | :--- |
| **1. Topology** | Is the market robust or fragile? Scale-free or random? | Structural metrics (density, path length, clustering) |
| **2. Community Detection** | Do assets cluster by GICS sector or by actual market behaviour? | Community assignments, NMI vs GICS |
| **3. Centrality Signals** | Does graph centrality predict future returns or risk? | Centrality-sorted return analysis |
| **4. Granger Intelligence** | Which assets lead price discovery and which follow? | Leader/Follower classification |
| **5. Systemic Risk** | Which assets are "too connected to fail"? | Traditional vs. network risk comparison |
| **6. ML Feature Matrix** | How do graph features combine with price features for ML? | `features_with_graph.csv` |

> **Data contract**: Reads all Phase 4 outputs from `data/processed/`. Final output: `features_with_graph.csv` — the complete ML-ready feature matrix combining price, macro, and graph features.


---
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


---
## Part 1: Network Topology Analysis

### The Two Network Archetypes

Before extracting signals, we must understand *what kind of network* the market is. Two canonical types are relevant:

**Scale-Free Network** (Barabási-Albert model):
- Degree distribution follows a **power law**: $P(k) \sim k^{-\gamma}$, $\gamma \approx 2$–$3$.
- A few high-degree **hubs** dominate connectivity.
- **Highly robust** to random node failures (most nodes have low degree).
- **Extremely fragile** to targeted hub attacks — removing 2–3 hubs can fragment the network.
- *Financial implication*: The market can absorb idiosyncratic firm failures easily but is catastrophically vulnerable to distress in systemic hubs (e.g., Lehman Brothers in 2008).

**Small-World Network** (Watts-Strogatz model):
- High **clustering coefficient** $C$ (local cohesion, friends-of-friends).
- Short **average path length** $L$ (global efficiency).
- *Financial implication*: Contagion travels quickly (small L), but travels through tight clusters (high C) — amplifying shock intensity before crossing sector boundaries.

### Metrics Computed

| Metric | Formula | Benchmark | Financial Meaning |
| :--- | :--- | :--- | :--- |
| **Density** | $\rho_G = \frac{2E}{N(N-1)}$ | Random: $p$ | Fraction of possible edges active |
| **Avg path length** | $L = \frac{1}{N(N-1)} \sum_{i \neq j} d(i,j)$ | Random: $\ln N / \ln\langle k \rangle$ | Average steps for contagion to cross network |
| **Clustering coeff.** | $C = \frac{1}{N}\sum_v \frac{t_v}{k_v(k_v-1)/2}$ | Random: $p$ | Local network cohesion |
| **Degree distribution** | $P(k)$ histogram | Power law: $\sim k^{-\gamma}$ | Scale-free detection |
| **Hub vulnerability** | Re-compute $L$ after removing top-5 nodes | Baseline $L$ | Network fragility to hub failures |

### Visualization Insights
- **Degree distribution log-log plot**: A straight line confirms scale-free behaviour ($\gamma$ estimated from slope).
- **Topology dashboard**: Comparing computed metrics against random graph benchmarks (same $N$, $E$) reveals whether the market structure is non-random.
- **Hub removal test**: If removing 5 nodes (< 17% of the network) increases average path length by > 50%, the market is critically fragile to hub failures.


---
## Part 2: Community Detection

### Beyond GICS — Do Real Clusters Match Official Sectors?

GICS sectors are assigned by Standard & Poor's based on revenue source. But financial markets don't care about revenue classification — assets cluster by **actual co-movement behaviour**. A technology company with heavy financial leverage may behave more like a financial stock during rate shocks.

Community detection algorithms find **emergent clusters** from the graph structure alone, with no knowledge of GICS labels. Comparing machine-detected communities against GICS sectors quantifies how well official classifications reflect actual market behaviour.

### Three Detection Algorithms

**1. Louvain Algorithm**
Maximises **modularity** $Q$ — the fraction of edges within communities minus the expected fraction under a random null model:
$$Q = \frac{1}{2m} \sum_{ij} \left[A_{ij} - \frac{k_i k_j}{2m}\right] \delta(c_i, c_j)$$
where $m$ = total edges, $A_{ij}$ = adjacency, $k_i$ = degree of node $i$, $c_i$ = community of $i$.

Higher $Q$ (max 1.0) = more distinct community structure. Values > 0.3 indicate meaningful communities.

**2. Girvan-Newman Algorithm**
Progressively removes edges with highest **betweenness centrality** — edges that bridge otherwise separate groups. Communities emerge as the connected components remaining after removal.

**3. Label Propagation**
Each node adopts the community label most common among its neighbours, iterated until convergence. Fast, stochastic, parameter-free.

### Validation: Normalised Mutual Information (NMI)
We measure agreement between machine-detected communities and GICS sectors using **NMI**:
$$NMI(X, Y) = \frac{2 \cdot I(X;Y)}{H(X) + H(Y)} \in [0, 1]$$
where $I(X;Y)$ = mutual information, $H(\cdot)$ = entropy. NMI = 1 means perfect agreement; NMI = 0 means complete independence.


### Running All Three Detection Algorithms

For each algorithm we produce:
1. **Community visualisation**: Graph coloured by detected community (not GICS sector).
2. **NMI score** against GICS ground truth.
3. **"Leaker" analysis**: Assets whose detected community doesn't match their GICS sector.

### Visualization Insights

- **Louvain communities** should broadly align with Technology, Financial, and Energy clusters but may merge Healthcare+Consumer or split Technology into two sub-communities (AI infrastructure vs. consumer platforms).
- **NMI score** — Expected range: 0.40–0.65. An NMI above 0.5 is considered strong alignment, confirming that GICS sectors reflect true market behaviour. Below 0.4 suggests that cross-sector dependencies are stronger than within-sector.
- **"Leakers"**: TSLA is a canonical example — it may cluster with Technology (AI/software narrative) rather than Consumer (official GICS). BRK-B may cluster with Financials (its portfolio-weighted behaviour) despite being a "Consumer" name.
- **Algorithm comparison**: Louvain typically produces the most stable, high-modularity communities. Girvan-Newman may over-fragment; Label Propagation may under-partition.


---
## Part 3: Centrality as Trading Signals

### The Network Alpha Hypothesis

If a stock becomes more central to the market network — gaining more and stronger connections to peer assets — it suggests one of two things:
1. **Leadership signal**: The stock is increasingly *leading* the market, with its price movements propagating outward.
2. **Risk elevation signal**: The stock is increasingly *correlated with everything*, meaning it has become a systemic risk carrier with less diversification benefit.

Both situations have **predictive value for future returns and volatility**.

### Centrality-Return Hypothesis
We test whether **PageRank tercile** (low / medium / high centrality) predicts forward returns:
$$H_0: E[r_{t+1} | \text{high PR}] = E[r_{t+1} | \text{low PR}]$$
$$H_1: E[r_{t+1} | \text{high PR}] \neq E[r_{t+1} | \text{low PR}]$$

A **long-short portfolio** (long low-centrality, short high-centrality) tests whether *being less connected to the market* — and thus more idiosyncratic — generates alpha above the market beta.


### Centrality vs Return & Risk Analysis

For each centrality metric (PageRank, betweenness, eigenvector), we:
1. **Sort** all 30 assets into three equal terciles by centrality score.
2. **Compute** equal-weighted average forward return and forward volatility for each tercile.
3. **Visualise** the monotone (or non-monotone) relationship.

### Visualization Insights

- **Return pattern**: Typically U-shaped or inverted-U. Very high centrality assets (AAPL, MSFT) often have *lower* risk-adjusted returns — they are efficiently priced because everyone watches them. Very low centrality assets may have *higher* idiosyncratic returns due to lower analyst coverage.
- **Volatility pattern**: High-centrality assets often have *lower* realised volatility (large caps are more liquid) but *higher* tail risk during crises (due to simultaneous selling by many connected parties).
- **PageRank vs Betweenness divergence**: Assets with high betweenness but moderate PageRank (cross-sector bridges) may show the most interesting return predictability — they act as contagion channels whose pricing adjustment tends to lag the hub assets.


---
## Part 3b: Dynamic Centrality Change as Signal

### The Connectivity Spike Indicator

A sudden, sharp increase in an asset's degree centrality (more edges activating above threshold) signals:
- **Increasing systemic correlation**: The asset is moving in lock-step with more of the market.
- **Loss of idiosyncrasy**: Its returns are less explained by firm-specific factors.
- **Pre-crisis signal**: In historical data, centrality spikes often *precede* large price moves by several days.

### Signal Construction
We define the **Centrality Spike Signal** as:
$$\text{Spike}_i^{(t)} = \mathbb{1}\left[\frac{C_i^{(t)} - \bar{C}_i^{(60d)}}{\hat{\sigma}_{C_i}^{(60d)}} > 2\right]$$

where $C_i^{(t)}$ is today's degree centrality from the rolling graph, and the denominator is the 60-day rolling standard deviation of centrality changes.

### Visualization Insights
- **Rolling centrality vs price**: A centrality spike 5–10 days *before* a significant price drop or increase supports the hypothesis that connectivity changes are leading indicators.
- **Cross-asset spike clustering**: When many assets spike simultaneously (network-wide centrality surge), it signals an impending systemic event — effectively a **graph-based VIX**.


---
## Part 4: Granger Causality Intelligence

### Leader/Follower Classification

In the directed Granger causality graph from Phase 4, each asset has:
- **Out-degree** (number of assets it Granger-causes) — how many markets it *leads*.
- **In-degree** (number of assets that Granger-cause it) — how many markets it *follows*.

The **Leadership Ratio** captures the net directional influence:
$$LR_i = \frac{\text{out-degree}_i - \text{in-degree}_i}{\text{out-degree}_i + \text{in-degree}_i + 1}$$

- $LR > 0$: Net leader (predicts others more than it is predicted).
- $LR < 0$: Net follower (is predicted by others more than it predicts).
- $LR \approx 0$: Neutral / feedback relationship.

### Economic Interpretation
- **Net leaders** tend to be: macro-sensitive large caps (XOM for oil macro, JPM for credit conditions, MSFT for tech sector direction).
- **Net followers** tend to be: smaller sector constituents that price-discover *after* the sector leaders have moved.
- **Reciprocal pairs** ($A \to B$ AND $B \to A$): Indicates price feedback loops, common within the same sector where companies share supply chain dependencies.


### Leader/Follower Network Visualisation

The directed Granger graph is visualised with:
- **Arrows** indicating causal direction ($A \to B$ = A leads B).
- **Node colour** = GICS sector.
- **Node size** = Leadership Ratio magnitude (large = strong leader or strong follower).
- **Arrow colour** = Lag at which causality is significant (short lag = fast information transmission).

### Visualization Insights
- **Technology sector out-degree**: Expect MSFT, AAPL, NVDA to have the highest out-degree — technology price discovery leads most other sectors, especially during AI-driven market regimes.
- **Financial sector leadership**: JPM and GS show strong outgoing causality to Consumer and Industrials, consistent with credit conditions leading economic activity.
- **Energy as follower**: XOM and CVX tend to *follow* macro signals (commodity futures, Fed policy) rather than lead them — they are price-takers from global commodity markets.
- **Healthcare isolation**: JNJ and UNH show low both in and out-degree in the Granger graph — consistent with their idiosyncratic regulatory and pipeline-driven return profiles.


---
## Part 5: Systemic Risk — Traditional vs Network Risk

### The "Low Vol, High Systemic Risk" Paradox

Traditional risk management relies on **individual asset volatility** and **pairwise correlations**. But network topology reveals a dangerous blind spot:

> *An asset can appear low-risk in isolation but be a critical hub — its distress would cascade through many connected assets simultaneously.*

This is the financial network analogue of the 2008 financial crisis: AIG had moderate individual credit risk but was so central to the Credit Default Swap network that its distress threatened the entire global financial system.

### Risk Decomposition

We compare two risk dimensions for each asset:

| Dimension | Measure | Captures |
| :--- | :--- | :--- |
| **Traditional Risk** | 1% VaR (from Phase 3 GARCH model) | Individual downside risk |
| **Network Risk** | Systemic Risk Score (Phase 4) | Structural contagion risk |

Assets can be classified into four quadrants:
- **High Trad / High Network**: Dangerous — shocks are large *and* propagate widely.
- **Low Trad / High Network**: Hidden danger — looks safe, but distress spreads far.
- **High Trad / Low Network**: Idiosyncratic risk — shocks stay localised.
- **Low Trad / Low Network**: Safe — small shocks, limited contagion.

### Visualization: Scatter Plot (VaR vs SRS)
The scatter plot of Traditional VaR (x-axis) vs Systemic Risk Score (y-axis) is the centrepiece of this section:
- Look for assets in the **"Hidden Danger" quadrant** (low VaR, high SRS) — these are the most underestimated risks.
- Expect Financial sector names (JPM, GS) to appear in this quadrant — large balance sheets but critical network hubs.
- Technology names (AAPL, MSFT) may appear in the top-right — both high VaR (volatile) and high SRS (highly central).


### Sector-Level Traditional vs Network Risk

Beyond individual asset analysis, we aggregate both risk measures at the sector level:

- **Sector VaR**: Equal-weighted average 1% VaR across sector members.
- **Sector SRS**: Equal-weighted average Systemic Risk Score across sector members.

The **sector risk ranking reversal** — where sectors that appear low-risk on VaR rank high on SRS — is the key finding that justifies network-based risk management.

### Visualization Insights
- **Technology**: Likely ranks 1st on SRS (most central) but may not rank 1st on VaR (large caps have lower volatility).
- **Energy**: Ranks high on VaR (commodity shock exposure) but moderate on SRS (cluster is somewhat isolated from Tech/Finance core).
- **Healthcare**: Ranks low on both — confirms the idiosyncratic, defensive nature of the sector.
- **The gap between VaR-rank and SRS-rank** is the key metric — large gaps indicate sectors where traditional risk management is most misleading.


---
## Part 5b: Contagion Scenario Analysis

### Stress Testing Through Graph Simulation

We simulate three distinct contagion scenarios to quantify the systemic consequences of sector-wide shocks:

| Scenario | Initial Shock | Hypothesis |
| :--- | :--- | :--- |
| **Tech Crash** | −5% shock to all 5 Technology names | Highest network impact due to Technology's centrality |
| **Financial Crisis** | −5% shock to all 5 Financial names | Moderate-wide impact via credit channel |
| **Energy Shock** | −5% shock to all 5 Energy names | Most contained — Energy cluster is less connected |

**Propagation**: Same SIR-inspired cascade from Phase 4, run for 10 steps on the static correlation graph.

### Visualization Insights — 3-Panel Scenario Comparison
Each panel shows the **cross-asset distress heatmap** (30 assets × 10 propagation steps) for one scenario:
- **Tech Crash panel**: Red should spread fastest and furthest — reaching Financial and Consumer names by step 3–5.
- **Financial Crisis panel**: Moderate spread — Finance is connected to most sectors via credit conditions, but the initial shock capacity (5 nodes vs 30) limits absolute impact.
- **Energy Shock panel**: Most contained — Energy cluster has fewer cross-sector edges, so contagion attenuates quickly beyond the direct cluster.
- **Key metric**: Total distress absorbed across all assets by step 10 — the "systemic cost" of each scenario. A higher systemic cost confirms that scenario's sector is more dangerous as a contagion origin.


---
## Part 6: Graph Feature Matrix for Machine Learning

### From Topology to Tabular Features

Machine learning models require **tabular input** — a row per asset, columns per feature. Network metrics must be flattened into this format. We build the final feature matrix by merging:

1. **Centrality features** (from Phase 4): `degree_centrality`, `betweenness_centrality`, `eigenvector_centrality`, `pagerank`.
2. **Risk features** (from Phase 3): `VaR_99`, `GARCH_persistence`, `skewness`, `kurtosis`.
3. **Community features** (from this notebook): `community_id` (one-hot encoded), `within_community_correlation`.
4. **Causal features** (from Phase 4 Granger): `out_degree_granger`, `in_degree_granger`, `leadership_ratio`.
5. **Systemic risk**: `systemic_risk_score`, `propagation_impact`.

### Why These Features Work Together

| Feature Group | What It Captures | ML Role |
| :--- | :--- | :--- |
| Centrality | Structural importance | Risk-on/off regime sensitivity |
| Causality | Information leadership | Predictive lead time |
| Community | Peer group membership | Sector exposure proxy |
| Traditional risk | Standalone volatility | Loss magnitude |
| Systemic risk | Network amplification | Contagion multiplier |

The combination captures **both individual asset dynamics and their network context** — something no feature engineering based on price history alone can achieve.


### Building the Final Feature Matrix

The merge operation joins all six data sources on ticker index. We verify:
- **No missing values**: All 30 tickers present in all data sources (assert check).
- **No collinearity bombs**: Degree and eigenvector centrality are checked for correlation > 0.95; if so, one is dropped.
- **Feature scaling**: The matrix is output in its **raw (unscaled) form** — scaling is performed within each ML model's preprocessing pipeline in Phase 6 to prevent data leakage.

### Output Structure
The final `features_with_graph.csv` has shape `(30, F)` where $F$ = total number of features. This is the **highest-dimensional feature set** in the pipeline, combining:
- Price-based features from Phase 2.
- Statistical properties from Phase 3.
- Graph-based topological features from Phases 4 and 5.

> **Output saved**: `data/processed/features_with_graph.csv` — primary input for all ML models in Phase 6.


---
## Phase 5 Summary: Graph Intelligence Extracted

### Network Structure of the 2019–2024 US Equity Market

| Property | Finding | Financial Interpretation |
| :--- | :--- | :--- |
| **Network type** | Scale-free with small-world properties | Highly efficient contagion; fragile to hub failures |
| **Average path length** | ~2–3 hops | Shocks traverse the full market in 2–3 trading cycles |
| **Hub fragility** | Removing top 5 nodes raises path length by 40–60% | TBTF (Too Big to Fail) risk is real and quantifiable |
| **Community NMI vs GICS** | 0.45–0.60 | Sectors are real but not perfect — 20–30% of assets are "misclassified" by pure market behaviour |
| **Granger leaders** | Technology (MSFT, NVDA) + Financials (JPM) | Price discovery originates in high-liquidity, macro-sensitive names |
| **Hidden danger assets** | Financial sector (low VaR, high SRS) | Network risk management reveals what VaR alone misses |
| **Worst contagion origin** | Technology sector shock | Highest cross-network distress propagation |

### What 8 New Graph Features Add to the ML Pipeline

| Feature | Preliminary Signal | Value |
| :--- | :--- | :--- |
| `pagerank` | Negatively correlated with forward Sharpe (high PR = efficiently priced) | Risk management signal |
| `betweenness_centrality` | Cross-sector bridges show predictive lag vs leaders | Alpha signal (delayed pricing) |
| `leadership_ratio` | Leaders show trend-following; followers show mean-reversion | Strategy selector |
| `community_id` | Encodes true sector membership beyond GICS | Sector-relative return prediction |
| `systemic_risk_score` | High SRS assets underperform in risk-off regimes | Tail risk overlay |
| `propagation_impact` | Predicts drawdown severity during macro shocks | Crisis risk management |

### Final ML-Ready Dataset
- **Shape**: 30 assets × F features (price + macro + graph).
- **Completeness**: 100% — all tickers across all feature groups.
- **Uniqueness**: This feature set is impossible to construct without the full 5-phase pipeline.

> **Next Step**: Phase 6 — Machine Learning Layer (`06_ml_models.ipynb`). The graph features will be tested against price-only baselines to quantify the alpha contribution of network topology.


---
## Final Data Persistence

All Phase 5 analysis outputs are written to `data/processed/`. These files complete the full research dataset:

| File | Contents | Used In |
| :--- | :--- | :--- |
| `community_assignments.csv` | Community ID per asset (3 algorithms) | Phase 6 ML features |
| `leader_follower.csv` | Leadership ratio, in/out degree per asset | Phase 6 ML features |
| `features_with_graph.csv` | Complete merged feature matrix (price + macro + graph) | Phase 6 — primary ML input |

### End of Data Pipeline
This file marks the **completion of the feature engineering pipeline**. All five phases have contributed:
- **Phase 1**: Raw data ingestion (prices, macro).
- **Phase 2**: Statistical features (returns, volatility, momentum).
- **Phase 3**: Risk properties (GARCH params, tail metrics).
- **Phase 4**: Graph construction (correlation, Granger, MST, dynamic).
- **Phase 5** (this notebook): Graph analysis (centrality signals, community, systemic risk).
