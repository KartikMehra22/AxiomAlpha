# Phase 9B: LangGraph Orchestration
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
- **Mean-Variance Optimization**: Used by the StrategyAgent to maximize expected return for a given level of risk: $\max_w (w^T \mu - \frac{\lambda}{2} w^T \Sigma w)$
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


### LangGraph State Schema

LangGraph requires a typed state definition.
Every field that agents read or write
must be declared here.

This is the "contract" between agents:
  Each agent knows exactly what it will receive
  Each agent knows exactly what it must produce

### LangGraph Node Functions

Each agent becomes a node function.
Node functions take state and return state.
LangGraph calls them in graph order.

Key difference from raw agent classes:
  Node functions must be pure functions
  They instantiate agents internally
  They handle all errors and return valid state

### Building the LangGraph

Now we wire all nodes together with edges.
The graph defines:
  Which agents run in which order
  What conditions trigger which paths
  Where the pipeline ends

**Visualization 1: AxiomAlpha Agent Graph**
The following code generates a networkx diagram illustrating the state machine flow, conditional routes, and terminal nodes of the agent orchestration framework.

### Running The Full Pipeline

This is the main execution cell.
We initialize the state with all data and
let the graph run to completion.

One execution = one complete investment decision.
In production this would run daily at market open.

**Visualization 2: Pipeline Execution Trace**
The code also renders a dashboard outlining execution time per agent, state size accumulation, logging trails, and final outputs.

### Complete Pipeline Output

This is the final investment decision
produced by the full AxiomAlpha system.
Every number is traceable to its source agent.

**Visualization 3: Complete Decision Dashboard**
Provides a unified, real-time overview of the current investment stance, aggregated from all underlying models and intelligence.

# Orchestration Complete

## Pipeline Performance
- Total execution time: [X]s
- Agents completed: 7/7
- Errors encountered: [N]
- Warnings: [N]

## Decision Summary
- Regime: [CURRENT] ([X]% confidence)
- Cash position: [X]%
- Invested: [X]% across [N] assets
- Top holding: [ticker] ([X]%)
- Portfolio CVaR 95%: [X]%

## What Notebook 15 Does
Runs a live interactive demo where you can:
  1. Ask questions about the portfolio
  2. Get LLM-generated explanations
  3. Simulate different regimes
  4. Run stress tests on the pipeline