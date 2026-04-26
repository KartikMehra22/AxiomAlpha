# Phase 9: Multi-Agent Decision System
## AxiomAlpha — AI Quant Research System

### Why Multi-Agent Architecture

Classical quant systems are monolithic:
- One script → one output
- Hard to debug, impossible to explain
- Single point of failure

AxiomAlpha uses a team of specialized agents:
- Each agent = one domain expert
- Agents communicate through structured messages
- Final decision synthesizes all agent outputs
- Every step is auditable and explainable

### The Agent Team

| Agent | Role & Responsibility | Primary Outputs |
| :--- | :--- | :--- |
| 📡 **DataAgent** | Loads + validates latest market data. Detects anomalies. | Clean data bundle, Data Quality Score |
| 📊 **QuantAgent** | Detects current market regime and computes volatility state. | Regime classification, Vol assessment |
| 🌐 **GraphAgent** | Computes network centrality and systemic risk spikes. | Systemic risk scores, Penalty matrices |
| 🤖 **MLAgent** | Generates machine learning direction predictions and forecasts volatility. | ML alpha signals, Prediction conviction |
| 📰 **NLPAgent** | Retrieves relevant news context and computes sentiment signals. | Sentiment scores, RAG Context |
| ⚠️ **RiskAgent** | Computes portfolio CVaR/VaR and flags concentration limits. | Risk audit report, Hard constraint flags |
| 🎯 **StrategyAgent** | Synthesizes all agent outputs into a portfolio optimization engine. | Final portfolio weights, Execution rationale |

### Agent Communication Protocol

Each agent receives an `AgentState` object:
- Contains all data produced so far
- Each agent reads what it needs
- Each agent writes its output to state
- State flows sequentially through the pipeline

### Message Schema

```python
AgentState = {
    # Raw data
    'returns': DataFrame,
    'features': DataFrame,
    'macro': DataFrame,
    
    # Agent outputs (filled progressively)
    'data_report': dict,
    'regime': str,
    'regime_confidence': float,
    'volatility_state': str,
    'systemic_scores': Series,
    'risk_flags': list,
    'ml_signals': DataFrame,
    'sentiment_signals': DataFrame,
    'news_context': str,
    'risk_report': dict,
    'portfolio_weights': Series,
    'cash_position': float,
    'explanation': str,
    'confidence_score': float,
    
    # Metadata
    'timestamp': datetime,
    'errors': list,
    'warnings': list
}
```


### Agent 1: DataAgent

**What we did**: Implemented a strict data quality gating mechanism as the first step of our multi-agent pipeline.

**Why we did it**: To completely avoid "garbage in, garbage out" scenarios. Institutional quantitative systems must automatically halt or flag anomalies if market data is missing, stale, or contains extreme errors (fat fingers, data feed glitches).

**Formulas & Metrics**:
- **Z-Score Anomaly Detection**: We standardize daily returns to detect extreme outliers (events far beyond typical market moves).
  \[ Z_i = \frac{R_i - \mu_i}{\sigma_i} \]
  *We flag any $Z_i > 5$ as an extreme anomaly.*
- **Data Quality Score**: A heuristic deduction system starting at 100:
  \[ \text{Score} = 100 - (\text{Missing Tickers} \times 10) - (\text{Stale Days} \times 20) - (\text{Anomalies} \times 15) \]

**Insights**: If the `data_quality_score` falls below 70, `is_valid` becomes False, and the system would ideally trigger a human-in-the-loop review rather than trade on corrupt intelligence.


### DataAgent Output
Quality Score: [X]/100
Key findings:
- [N] anomalies detected
- Data is [fresh/stale]
- Pipeline will [proceed/halt]

---

### Agent 2: QuantAgent

#### Role
The QuantAgent is the market statistician.
It answers the fundamental question:
"What is the current state of the market?"

Two key outputs:
  1. Market Regime (Bull/Bear/Volatile)
  2. Volatility State (Low/Normal/High/Extreme)

#### Regime Detection Method
Uses Hidden Markov Model (HMM) or KMeans
on rolling market features:
  - 20d realized volatility
  - 5d market return
  - 20d market return
  - VIX level (if available)

#### Volatility State Classification
  vol_zscore = (current_vol - mean_vol) / std_vol
  Low:     vol_zscore < -1
  Normal:  -1 ≤ vol_zscore ≤ 1
  High:    1 < vol_zscore ≤ 2
  Extreme: vol_zscore > 2

#### Why This Drives Everything Downstream
Regime determines:
  - Which ML model weights to use
  - Portfolio concentration limits
  - Cash position size
  - Risk aversion parameter (λ)
  
Volatility state determines:
  - Position sizing scalar
  - CVaR threshold
  - Rebalancing urgency


### Agent 2: QuantAgent

**What we did**: Deployed a K-Means clustering algorithm over rolling volatility and momentum windows to dynamically classify the current market state into discrete "Regimes" (e.g., Bull, Bear, Volatile).

**Why we did it**: Financial markets are non-stationary. A momentum strategy that works beautifully in a steady bull market will get destroyed in a volatile bear market. Detecting the regime allows downstream agents to adjust their aggressiveness.

**Formulas & Metrics**:
- **Rolling Volatility (Annualized)**:
  \[ \sigma_{20d} = \text{std}(R_{t-20 \dots t}) \times \sqrt{252} \]
- **Rolling Momentum**:
  \[ \text{Mom}_{5d} = \sum_{i=0}^{4} R_{t-i} \quad , \quad \text{Mom}_{20d} = \sum_{i=0}^{19} R_{t-i} \]
- **Clustering**: We use `KMeans` to group these 3 features ($\sigma_{20d}$, Mom$_{5d}$, Mom$_{20d}$) into $k=3$ regimes based on historical distances.

**Insights**: A "Volatile" state with negative momentum usually dictates a cash-heavy or minimum-variance defensive posture, while a low-volatility "Bull" state allows maximum alpha extraction.


### Agent 3: GraphAgent

**What we did**: Built a dynamically evolving network topology of the asset universe based on rolling 60-day Pearson correlations, calculating centrality metrics for every stock.

**Why we did it**: Traditional risk models miss hidden contagion channels. If AAPL and MSFT are highly correlated, bad news for one will likely crash the other. By mapping the market as a graph, we quantify the "Systemic Risk" of holding highly interconnected assets.

**Formulas & Metrics**:
- **Adjacency Matrix**: An edge exists if the correlation exceeds a threshold (e.g., $\rho_{i,j} > 0.4$):
  \[ A_{i,j} = \begin{cases} 1 & \text{if } \rho_{i,j} > 0.4 \\ 0 & \text{otherwise} \end{cases} \]
- **Degree Centrality**: The fraction of nodes a stock is connected to. High degree = high systemic footprint.
- **Systemic Risk Score**: A synthesized 0-100 metric. Assets closer to 100 are heavily interwoven and dangerous during market shocks.

**Insights**: If we construct a portfolio out of high-centrality assets, we are making a massive, concentrated bet on a single macro factor. The GraphAgent penalizes these assets to force diversification.


### Agent 4: MLAgent

**What we did**: Integrated pre-trained machine learning models (Gradient Boosting, Random Forests) into the pipeline to forecast asset direction probabilities and upcoming volatility.

**Why we did it**: While QuantAgent handles macro-level regimes, MLAgent handles micro-level (asset-specific) alpha generation by finding complex non-linear patterns in technical and fundamental features.

**Formulas & Insights**:
- **Direction Probability**: The output is a raw probability $P(\text{Up})$ for the next holding period. 
- **Alpha Signal ($A_i$)**: We map the probabilities (typically 0.3 to 0.7) into a conviction multiplier for our expected returns matrix later.
- **Regime-Conditional Execution**: Notice that the MLAgent maintains a dictionary of `model_accuracy` keyed by Market Regime. ML models often degrade in 'Volatile' regimes, so their output weights can be dialed down when the QuantAgent flags turbulence.


### Agent 5: NLPAgent

**What we did**: Incorporated a Retrieval-Augmented Generation (RAG) system to ingest unstructured textual data (financial news, macro reports) alongside pre-computed numerical sentiment momentum.

**Why we did it**: Markets are driven by narratives. Purely mathematical systems miss the "why" behind price movements. By extracting positive/negative ratios and querying a vector database with an LLM, we generate human-readable context and hard sentiment overlays.

**Formulas & Insights**:
- **Sentiment Mean**: Calculated as the rolling average of BERT/FinBERT polarity scores extracted from headlines.
- **Market Sentiment Flag**: If the aggregate market sentiment drops below a critical threshold (e.g., -0.3), the NLPAgent throws a hard `risk_flag` into the `AgentState`.
- **RAG Context**: The system explicitly queries its knowledge base asking: *"Market [Regime] regime current risks opportunities"* to provide a qualitative explanation for the portfolio managers.


### Agent 6: RiskAgent

**What we did**: Instituted a strict, rules-based compliance auditor that checks proposed portfolios against institutional limits before trade execution.

**Why we did it**: Unconstrained optimizers will often put 90% of your capital into a single stock if the math looks marginally better. The RiskAgent enforces legal, sector, and concentration limits.

**Formulas & Metrics**:
- **Conditional Value at Risk ($CVaR_{95\%}$)**: Measures the expected loss in the worst 5% of scenarios.
  \[ CVaR_{\alpha} = \frac{1}{1-\alpha} \int_{0}^{1-\alpha} VaR_{\gamma}(X) d\gamma \]
- **Herfindahl-Hirschman Index (HHI)**: Measures portfolio concentration.
  \[ HHI = \sum_{i=1}^{N} w_i^2 \]
  *If HHI > 0.15, the portfolio is dangerously top-heavy.*
- **Effective Assets**: $1 / HHI$. Tells us the "true" number of diversified bets we are making.


### Agent 7: StrategyAgent

**What we did**: Synthesized the macro (Quant), micro (ML), textual (NLP), and structural (Graph) signals into a mathematical portfolio optimization engine (Ledoit-Wolf shrinkage + Mean-Variance).

**Why we did it**: To calculate the globally optimal capital allocation vector $\mathbf{w}$ that maximizes return for a given level of risk, considering all agent inputs.

**Formulas & Optimization Framework**:
1. **Expected Returns ($\mu$) Generation**: 
   Starts with historical mean, then multiplied by ML directional probabilities and NLP sentiment scores. 
   \[ \mu_{final} = \mu_{hist} \times \text{ML}_{mult} \times \text{NLP}_{mult} \]

2. **Systemic Covariance Penalty**: 
   We calculate the base covariance matrix $\Sigma$ (using Ledoit-Wolf shrinkage for numerical stability). We then apply the GraphAgent's systemic scores as a penalty matrix $S$ (a diagonal matrix of scores).
   \[ \Sigma_{systemic} = S \Sigma S \]
   *This artificially inflates the variance of highly connected assets, forcing the optimizer to avoid them.*

3. **Objective Function**: Maximize Sharpe Ratio (minimize negative Sharpe):
   \[ \min_{w} \frac{w^T \Sigma_{systemic} w}{w^T \mu} \]
   *Subject to: $\sum w_i = 1$, $0 \le w_i \le 0.15$*

4. **Regime Cash Overlay**: If the QuantAgent detects a 'Volatile' regime, we forcefully reduce total market exposure (e.g., $w_{final} = w \times 0.5$) and hold the rest in cash.


## 🚀 AxiomAlpha System: Final Insights & Orchestration

We have now defined and tested all 7 specialized agents. 

### Core Insights Generated by the System:
1. **Regime Awareness is Critical**: By having the `QuantAgent` detect regimes, the `StrategyAgent` knows when to retreat to cash. A pure mathematical optimizer would blindly buy the dip in a volatile bear market.
2. **Systemic Penalties Work**: The `GraphAgent` successfully identifies assets that are dangerously interconnected. The `StrategyAgent` then actively shrinks their weights, forcing true diversification rather than just mathematical mean-variance optimization.
3. **Unstructured to Structured**: The `NLPAgent` brings human-like qualitative reasoning (RAG) and sentiment overlays into a purely quantitative pipeline, providing a narrative safety-net.
4. **Hard Risk Gates**: The `RiskAgent` acts as an institutional backstop, ensuring the final allocations never violate strict internal compliance limits (like CVaR and max sector concentration).

Below, we synthesize the entire run into a finalized System Output Report.