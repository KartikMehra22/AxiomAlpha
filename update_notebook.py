import json

with open('notebooks/14_agent_orchestration.ipynb', 'r') as f:
    nb = json.load(f)

intro_markdown = """# Phase 9B: LangGraph Orchestration
## AxiomAlpha — AI Quant Research System

### 1. Why We Did It (The Purpose)
In quantitative finance, decision-making is not a single linear step. It requires continuous coordination between data engineering, statistical modeling, machine learning, natural language processing, risk management, and portfolio optimization. 
Traditionally, these components operate in silos or via rigid sequential pipelines. By using **LangGraph**, we transform AxiomAlpha into an autonomous, state-driven multi-agent system. This allows the system to:
- **Adapt to failures** (e.g., if risk limits are breached, it can recursively route back to the strategy agent).
- **Execute in parallel** (e.g., ML predictions and NLP sentiment analysis can happen concurrently, saving execution time).
- **Maintain a robust audit trail** of every decision made by every agent at every step of the orchestration.

### 2. What We Did (The Implementation)
We orchestrated 7 specialized AI Agents into a state machine with conditional routing:
1. **DataAgent**: Validates incoming market data and features.
2. **QuantAgent**: Identifies the current market regime (Bull, Bear, Volatile).
3. **GraphAgent**: Analyzes systemic risk and network centrality among assets.
4. **MLAgent**: Generates alpha signals using machine learning models.
5. **NLPAgent**: Parses news and macro reports to gauge market sentiment.
6. **RiskAgent**: Acts as the system's "brakes," auditing the proposed portfolio against strict risk limits.
7. **StrategyAgent**: The final decision-maker that synthesizes all signals into target portfolio weights.

### 3. Key Concepts & Formulas Used
During the orchestration, the agents rely on several quantitative models:
- **Value at Risk (VaR) & Conditional Value at Risk (CVaR)**: Used by the RiskAgent to assess tail risk. $VaR$ answers "what is the most I can expect to lose with 95% confidence?", while $CVaR$ answers "if I do lose more than $VaR$, how bad will it be?".
- **Regime Classification**: Used by the QuantAgent to classify the market state based on volatility ($v$) and momentum ($m$).
- **Mean-Variance Optimization**: Used by the StrategyAgent to maximize expected return for a given level of risk: $\\max_w (w^T \\mu - \\frac{\\lambda}{2} w^T \\Sigma w)$
- **Network Centrality**: Used by the GraphAgent to penalize assets that are too highly interconnected, thus reducing systemic risk exposure.

### 4. Insights Gained from the Orchestrated Analysis
Running this pipeline gives us a comprehensive, unified view of the market:
- **Dynamic Asset Allocation**: The system shifts cash and asset weights dynamically based on the identified market regime.
- **Risk-Adjusted Alpha**: We don't just chase returns; every signal is rigorously audited by the RiskAgent, ensuring that risk limits are respected.
- **Explainable AI in Finance**: Because each agent contributes to the `GraphState`, we can precisely trace *why* the StrategyAgent chose a specific allocation (e.g., "Cash was raised to 30% because the RiskAgent flagged elevated volatility").

---

### AxiomAlpha Graph Architecture

  ┌──────────┐
  │  START   │
  └────┬─────┘
       │
       ▼
  ┌──────────┐    quality_score < 60
  │DataAgent │─────────────────────→ END (abort)
  └────┬─────┘
       │ quality_score ≥ 60
       ▼
  ┌──────────┐
  │QuantAgent│
  └────┬─────┘
       │
       ▼
  ┌──────────────────────────────┐
  │   PARALLEL EXECUTION         │
  │  ┌──────────┐ ┌──────────┐  │
  │  │GraphAgent│ │ MLAgent  │  │
  │  └──────────┘ └──────────┘  │
  │  ┌──────────┐               │
  │  │ NLPAgent │               │
  │  └──────────┘               │
  └──────────────────────────────┘
       │
       ▼
  ┌──────────┐
  │RiskAgent │
  └────┬─────┘
       │
       ├─ violations > 0 ──→ StrategyAgent (with constraints)
       │                           │
       └─ no violations ──→ StrategyAgent
                                   │
                                   ▼
                              ┌──────────┐
                              │  Output  │
                              └──────────┘

### State Transitions
- **Data quality check** → abort or continue
- **Risk violations** → re-optimize with constraints
- **Agent errors** → log and continue with defaults
- **Max retries** → abort with best available result
"""

def string_to_lines(s):
    lines = s.split('\n')
    return [line + '\n' for line in lines[:-1]] + [lines[-1]]

nb['cells'][0]['source'] = string_to_lines(intro_markdown)

with open('notebooks/14_agent_orchestration.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
