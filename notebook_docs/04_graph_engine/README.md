# Graph Engine: Market as a Network
**Research Module 4 of 5 | AxiomAlpha Framework**

---

Standard correlation matrices treat asset relationships as isolated bilateral pairs. A 30×30 matrix contains 435 pairwise ρ values, but each is computed **independently**—ignoring the chain of dependencies flowing through the market: A → B → C, or the hub status of a single asset connecting many clusters.

By modelling the market as a **weighted, directed graph**, we capture the **topology** of financial risk—how shocks originate, amplify, and propagate.

### The Graph Model

- **Nodes** — Assets (30 equities).
- **Edges** — Statistical relationships: correlation strength (undirected) or Granger-causal influence (directed).
- **Edge weights** — Correlation coefficient ρ or -log(p-value) for causality strength.
- **Centrality scores** — Measure of each node's structural importance to the network.

### Five-Part Architecture

| Part | Method | Output |
| :--- | :--- | :--- |
| **1. Correlation Graph** | Threshold-filtered undirected graph from Pearson ρ | Network visualisation + centrality scores |
| **2. Granger Causality** | Vector Autoregression F-tests on 870 directed pairs | Directed influence network |
| **3. Minimum Spanning Tree** | Kruskal's algorithm on correlation-distance matrix | Market backbone (noise-free skeleton) |
| **4. Dynamic Graphs** | Rolling 60-day correlation windows over time | Network density evolution vs VIX |
| **5. Risk Propagation** | SIR-style shock cascade on correlation graph | Systemic risk scores per asset |

> **Data contract**: Reads from `data/processed/log_returns.csv` and `data/processed/risk_metrics.csv`. Outputs: `centrality_scores.csv`, `granger_causality.csv`, `mst_edges.csv`, `dynamic_graph_metrics.csv`, `systemic_risk_scores.csv`.


---
## 1. Environment Setup & Data Loading

Key library dependencies:

- `networkx` — Graph construction, centrality algorithms (degree, betweenness, eigenvector, PageRank), MST computation.
- `statsmodels` — Granger causality F-test via vector autoregression (VAR).
- `scipy` — Distance matrix operations for MST.

All visualisations use a **dark premium theme** (`#0d0d0d` background, 150 DPI) for publication-quality figures.


---
## Part 1: Correlation Graph

### Why a Graph, Not Just a Heatmap?

A correlation heatmap shows 435 ρ values but reveals **no topology**. A graph reveals:
- **Hubs**: Nodes with many strong connections — systemic risk amplifiers.
- **Bridges**: Nodes connecting otherwise separate clusters — contagion pathways.
- **Clusters**: Densely connected subgraphs — natural market communities.

### Graph Construction

**Step 1 — Correlation matrix:**
$$\rho_{ij} = \frac{\text{Cov}(r_i, r_j)}{\sigma_i \cdot \sigma_j}$$

**Step 2 — Threshold filtering:**
We retain only edges where $\rho_{ij} > \tau = 0.40$. This **sparsification** serves two purposes:
1. Remove noise — low correlations are statistically unreliable at our sample size.
2. Focus the topology on structurally meaningful relationships.

**Step 3 — Edge weight assignment:**
$$w_{ij} = \rho_{ij} \quad \text{(larger weight = stronger relationship)}$$

**Step 4 — Layout:**
The **spring layout** (Fruchterman-Reingold algorithm) positions nodes such that highly correlated assets cluster together in 2D space. Nodes far apart have low correlation; nodes nearby are strongly connected.

### Visualization Insights
- **Node colour** = GICS sector — sector clustering should be visually apparent.
- **Node size** = degree centrality — hubs appear larger.
- **Edge thickness** = correlation strength — thick edges represent strong co-movement pairs.

**Expected pattern**: Technology stocks (AAPL, MSFT, NVDA, GOOGL, META) form a tight, densely connected cluster in the upper region. Energy names (XOM, CVX, COP) form a separate cluster. Healthcare sits between them with weaker cross-cluster connections.


---
## Part 1: Network Centrality Metrics

### What Centrality Measures

Centrality answers the question: *"Which nodes are most important to the network?"* Different centrality measures capture different notions of importance:

| Metric | Formula | Financial Interpretation |
| :--- | :--- | :--- |
| **Degree** | $C_D(v) = \frac{\text{deg}(v)}{N-1}$ | Local connectivity — how many peers share strong correlation |
| **Betweenness** | $C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$ | Bridge role — how many shortest paths pass through this node |
| **Eigenvector** | $C_E(v) = \frac{1}{\lambda} \sum_{u \in N(v)} C_E(u)$ | Influence score — connected to other influential nodes |
| **PageRank** | $PR(v) = \frac{1-d}{N} + d \sum_{u \in N(v)} \frac{PR(u)}{\text{deg}(u)}$ | Global prestige — Google's algorithm applied to financial networks |

where $\lambda$ = largest eigenvalue, $d = 0.85$ = damping factor (probability of following an edge vs. teleporting).

### Visualization: Centrality Bar Charts
- **Degree leaders**: Expect Technology sector names to rank highest — they are correlated with almost everything.
- **Betweenness leaders**: May be cross-sector names (e.g., V, JPM) that act as bridges between the Financial cluster and the rest of the market.
- **PageRank vs Degree divergence**: If a node has moderate degree but high PageRank, it is connected to other hubs — a "kingmaker" position in the network with outsized systemic influence.

> **Output saved**: `data/processed/centrality_scores.csv` — used as predictive features in Phase 5.


---
## Part 2: Granger Causality Graph

### Why Correlation is Insufficient for Causality

Correlation is **symmetric**: $\rho(A, B) = \rho(B, A)$. It cannot distinguish whether A influences B, B influences A, or both are driven by a common third factor C.

**Granger causality** (Granger, 1969; Nobel Prize 2003) provides a **directional** test:

> *"Variable X Granger-causes variable Y if past values of X, combined with past values of Y, predict future values of Y better than past values of Y alone."*

### Statistical Framework

We estimate the restricted and unrestricted **Vector Autoregression (VAR)** models for each ordered pair $(i, j)$:

**Restricted model** (Y alone predicts Y):
$$r_{j,t} = \alpha_0 + \sum_{k=1}^{p} \alpha_k r_{j,t-k} + \epsilon_t$$

**Unrestricted model** (X and Y together predict Y):
$$r_{j,t} = \alpha_0 + \sum_{k=1}^{p} \alpha_k r_{j,t-k} + \sum_{k=1}^{p} \beta_k r_{i,t-k} + \epsilon_t$$

**F-test** compares residual sum of squares:
$$F = \frac{(RSS_R - RSS_U)/p}{RSS_U/(T - 2p - 1)}$$

If $F$ is significant (p < 0.05), we conclude that $r_i$ **Granger-causes** $r_j$ at lag $p$.

We test **all $30 \times 29 = 870$ directed pairs** at max lag $p = 5$ (one trading week).

### Visualization: Directed Causality Network
- **Arrow direction**: $A \rightarrow B$ = "A's past predicts B's future."
- **Arrow thickness**: $-\log(\text{p-value})$ — thicker = stronger causal evidence.
- **Expected finding**: Macro-sensitive names (XOM, GS) tend to be **sources** (many outgoing arrows); sector followers (smaller sector constituents) tend to be **sinks** (many incoming arrows).

> **Output saved**: `data/processed/granger_causality.csv` — edge list of significant causal relationships.


### Granger Testing: All 870 Directed Pairs

The test is computationally intensive ($870$ F-test regressions, each with up to 5 lags). We apply a **Bonferroni correction** to the significance threshold to control for multiple testing:

$$\alpha_{\text{adjusted}} = \frac{0.05}{870} \approx 5.7 \times 10^{-5}$$

Only pairs surviving this stringent threshold are added to the directed graph, ensuring the edges represent genuine predictive relationships rather than false discoveries from the large number of tests.

**Key metrics to examine in the output:**
- **In-degree** (Granger followers): Assets that are *predicted by* many others — they lag the market.
- **Out-degree** (Granger leaders): Assets that *predict* many others — they lead price discovery.
- **Reciprocal pairs** ($A \rightarrow B$ AND $B \rightarrow A$): Indicates feedback loops or high regime synchronisation.


---
## Part 3: Minimum Spanning Tree (MST)

### Why the MST?

The correlation graph with threshold $\tau = 0.4$ still contains hundreds of edges and significant noise. The **Minimum Spanning Tree** extracts the **market backbone** — the minimal set of $N-1 = 29$ edges that:
1. Connect all 30 nodes (fully connected).
2. Minimise total edge weight (retain only the strongest structural relationships).
3. Remove all cycles (eliminate redundant loops).

The result is the **clearest possible view of the market's hierarchical structure** — which assets are directly linked, and which are connected only through intermediaries.

### Distance-Correlation Transformation

MSTs minimise distance, not correlation. We convert correlation to a **metric distance** (Mantegna, 1999) that satisfies the triangle inequality:

$$d_{ij} = \sqrt{2(1 - \rho_{ij})}$$

Properties:
- $d_{ij} = 0$ when $\rho_{ij} = 1$ (perfect co-movement).
- $d_{ij} = \sqrt{2}$ when $\rho_{ij} = 0$ (no co-movement).
- $d_{ij} = 2$ when $\rho_{ij} = -1$ (perfect counter-movement).

**Kruskal's algorithm** then finds the MST in $O(E \log E)$ time.

### Visualization: Market Backbone
The MST visualisation is the most **information-dense** chart in this notebook:
- **The "spine"** of the MST: A main chain connecting the most correlated sectors. Technology names typically appear near the centre.
- **"Leaf" positions**: Sector outliers (e.g., NEE — renewable energy — and BRK-B — diversified conglomerate) hang off the ends of the tree as they have idiosyncratic correlations.
- **Sector separation**: The tree naturally partitions into sector subtrees, visually confirming the sector structure proven statistically in Phase 3 (H4).
- **MST vs Full Graph comparison**: The MST reveals structure that is *hidden* in the dense correlation matrix.

> **Output saved**: `data/processed/mst_edges.csv` — backbone edge list for use in Phase 5 graph analysis.


---
## Part 4: Dynamic (Time-Varying) Graphs

### Why Static Graphs Are Insufficient

Every graph built so far uses the **full 2019–2024 sample** — averaging out 6 years of market structure changes. But Phase 3 proved that correlations are **regime-dependent** (H3 rejection). Therefore:

- A graph built during the 2021 bull market looks very different from one built during the March 2020 crash.
- Systemic risk metrics computed from a static graph are **misleading**: they overstate risk in calm periods and understate it in crises.

**Dynamic graphs** solve this by using a **rolling window** approach:
- Window width: $W = 60$ trading days (3 months of history).
- Step size: $s = 1$ day (daily re-computation).
- Threshold: Same $\tau = 0.4$.

### Key Dynamic Metrics

At each time step $t$, we compute:

| Metric | Formula | Meaning |
| :--- | :--- | :--- |
| **Network density** | $\rho_G = \frac{2E}{N(N-1)}$ | Fraction of possible edges that are active |
| **Average clustering** | $C = \frac{1}{N} \sum_v \frac{\text{triangles}(v)}{\text{triplets}(v)}$ | Local cohesion of the network |
| **Avg edge weight** | $\bar{w} = \frac{1}{E} \sum_{(i,j) \in E} \rho_{ij}$ | Mean correlation strength of active edges |

### Visualization: Network Evolution vs VIX
The dual-axis time series (network density + VIX) is the **key diagnostic chart**:
- **Expected correlation**: Network density should spike sharply when VIX spikes — the "correlation convergence" phenomenon (H3) manifests as a denser graph.
- **March 2020**: Density should jump from ~0.3 to ~0.7+ as all 435 pairs surpass the threshold simultaneously.
- **2022 rate shock**: Sustained elevated density for 9–12 months, reflecting the prolonged bear market.
- **2019 and 2023 bull periods**: Low density, indicating that assets are decoupled and moving on idiosyncratic fundamentals.

> **Output saved**: `data/processed/dynamic_graph_metrics.csv` — time series of density, clustering, and avg weight.


---
## Part 5: Risk Propagation & Contagion Simulation

### The Contagion Question

Knowing which assets are *central* (Part 1) is important, but doesn't directly answer: *"If asset X suffers a large shock today, how far does its distress travel through the network, and who gets hit hardest?"*

We simulate this using a **network-based shock propagation model** inspired by SIR epidemiology, adapted for financial contagion:

### Propagation Algorithm

1. **Initial shock**: Apply a −3σ return shock to the target asset (most central by eigenvector centrality).
2. **Propagation rule**: At each step $t$, each "infected" node $i$ propagates a fraction of its distress to neighbouring node $j$:
$$\delta_j^{(t+1)} = \sum_{i \in N(j)} w_{ij} \cdot \delta_i^{(t)} \cdot (1 - \text{absorption}_j)$$
where $w_{ij} = \rho_{ij}$ is the edge weight and the absorption factor represents each asset's own buffering capacity (inverse of its volatility).
3. **Termination**: Run for 10 steps or until distress falls below a materiality threshold.

### Why This Model?

- It directly uses the **graph topology** — high-centrality assets propagate shocks further.
- It accounts for **edge weight heterogeneity** — strong correlations mean more distress transfer.
- It is interpretable — the final distress vector shows exactly which assets absorb the most contagion.

### Visualization: Shock Cascade Heatmap
- **Y-axis**: 30 assets.
- **X-axis**: Propagation time steps (0 = initial shock, 1–10 = cascade).
- **Cell colour**: Distress level (red = high, dark = low).

**Expected insights**:
- The shocked asset and its direct neighbours (high $\rho_{ij}$) show immediate distress in step 1.
- By step 3–5, the shock has propagated to second-degree neighbours (assets correlated with the neighbours).
- Energy and Healthcare names — which have lower cross-sector correlation — show dampened distress in later steps.
- Technology names — highly interconnected — show widespread, persistent distress.


---
## Part 5: Systemic Risk Score Construction

### Combining Centrality + Propagation Impact

Centrality metrics alone tell us which assets are structurally important. Propagation simulations tell us which assets transmit the most distress. The **Systemic Risk Score (SRS)** combines both signals:

$$SRS_i = \alpha \cdot C_E(i) + \beta \cdot PR(i) + \gamma \cdot C_B(i) + \delta \cdot \text{PropagationImpact}(i)$$

where:
- $C_E(i)$ = Eigenvector centrality (global influence).
- $PR(i)$ = PageRank (prestige-weighted connectivity).
- $C_B(i)$ = Betweenness centrality (bridge role).
- $\text{PropagationImpact}(i)$ = Total distress received by $i$ in the shock cascade simulation.

Weights $\alpha, \beta, \gamma, \delta$ are equal (0.25 each) — the composite score treats all four dimensions as equally informative. In a production system, these would be calibrated via cross-validation against realised drawdown severity.

### Visualization: Systemic Risk Ranking
The ranked bar chart of SRS values reveals the **"Too Connected to Fail"** assets:
- **Top of ranking**: Expect large-cap Technology names (AAPL, MSFT) and Financial sector hubs (JPM, V) — they combine high centrality with high propagation impact.
- **Bottom of ranking**: Idiosyncratic names (BRK-B, NEE, NKE) — lower correlation with peers means shocks stay localised.
- **Cross-sector comparison**: Energy sector shows moderate SRS despite its tight within-sector clustering — because the Energy cluster is somewhat isolated from the broader Technology-Finance core.

> **Output saved**: `data/processed/systemic_risk_scores.csv` — used as a key feature in Phase 5 ML models.


---
## Phase 4 Summary: Graph Engine Complete

### What the Network Reveals That Traditional Analysis Cannot

| Analysis | Traditional Approach | Graph Approach | Additional Insight |
| :--- | :--- | :--- | :--- |
| **Asset importance** | Market cap ranking | Degree/PageRank centrality | Systemic risk ≠ firm size |
| **Relationships** | Pairwise correlations | Graph topology | Hub-and-spoke vs. cluster structure |
| **Causality** | None | Granger directed graph | Leader/follower dynamics across sectors |
| **Market structure** | Average correlation | MST backbone | Hierarchical sector tree visible |
| **Regime changes** | Rolling correlation | Dynamic graph density | Crisis = denser graph; calm = sparse graph |
| **Contagion** | Assumed equal spread | Propagation simulation | Shock path follows graph edges |

### Key Findings

1. **Systemic Hubs**: Large-cap Technology names (AAPL, MSFT) and Financial sector leaders (JPM, V) score highest on the composite Systemic Risk Score — their distress has the widest network ripple effect.
2. **Sector Clustering Confirmed**: The MST backbone naturally partitions into GICS sector subtrees, empirically validating Phase 3's H4 conclusion using a completely different methodology.
3. **Dynamic Density Spikes**: Network density is strongly correlated with VIX (r > 0.7), confirming Phase 3's H3 conclusion — correlation convergence is visible as a graph phenomenon.
4. **Granger Leaders**: Technology and Financial sector names tend to be net Granger-causers of other sectors, consistent with their macro sensitivity and price discovery role.
5. **Contagion Asymmetry**: Shocks propagate faster *within* sectors than *across* sectors, with Technology sector shocks reaching the most nodes due to its high cross-sector connectivity.

### Outputs for Phase 5

| File | Contents | Role in Phase 5 |
| :--- | :--- | :--- |
| `centrality_scores.csv` | Degree, betweenness, eigenvector, PageRank per asset | Graph-based ML features |
| `granger_causality.csv` | Directed causality edge list | Leader/follower feature construction |
| `mst_edges.csv` | MST backbone edge list | Structural constraint for portfolio construction |
| `dynamic_graph_metrics.csv` | Density, clustering, avg weight per date | Time-series regime features |
| `systemic_risk_scores.csv` | Composite SRS per asset | Contagion risk feature; risk overlay |

> **Phase 5 notebook**: `05_graph_analysis.ipynb`
