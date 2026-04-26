import json

with open('notebooks/15_agent_demo.ipynb', 'r') as f:
    nb = json.load(f)

intro_markdown = """# AxiomAlpha — Live System Demo
## AI-Powered Quantitative Research System

### 1. Why We Did It (The Purpose)
Traditional algorithmic trading strategies often rely on rigid, single-domain models (like momentum or mean-reversion) that fail when market regimes shift or unprecedented events occur. We built AxiomAlpha to overcome this fragility. 
The purpose of this demo is to showcase how a composite system—integrating statistical rigor, machine learning, natural language processing, graph theory, and portfolio optimization—can adapt autonomously to new data. By orchestrating these domains via LangGraph, we create a resilient, self-correcting system capable of dynamic risk mitigation and alpha generation.

### 2. What We Did (The Implementation)
We constructed a cohesive showcase (this notebook) that simulates a complete daily production run of the AxiomAlpha pipeline. This notebook demonstrates the system's capabilities through 5 interactive scenarios:
1. **Demo 1**: Generates the "Current Decision" based on today's market data.
2. **Demo 2**: Performs a "Regime Simulation," forcing the system into Bull, Bear, and Volatile states to visualize how the portfolio dynamically adapts.
3. **Demo 3**: Executes "Stress Tests," mimicking crises (like a COVID shock or liquidity dry-up) to prove the system's downside protection mechanisms.
4. **Demo 4**: Employs an LLM to translate raw quantitative outputs into a professional, human-readable investment thesis.
5. **Demo 5**: Implements an interactive Q&A RAG system that grounds LLM answers in the exact portfolio state and parsed news data.

### 3. Key Concepts & Formulas Used
The system's intelligence relies on blending disparate mathematical disciplines:
- **Network Density & Centrality**: Used by the GraphAgent to quantify systemic risk. High network density (interconnectedness) means shocks propagate faster. Centrality formula: $C_e(v) = \\frac{1}{\\lambda} \\sum_{t \\in M(v)} C_e(t)$
- **Regime Switching (GARCH)**: Used by the QuantAgent to detect periods of volatility clustering. $\\sigma_t^2 = \\omega + \\alpha \\epsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2$
- **Machine Learning Classifiers**: XGBoost and Random Forests predict price direction by synthesizing technicals, macro factors, and sentiment features (AUC = 0.545).
- **Sentiment Information Coefficient (IC)**: Measures the predictive power of FinBERT sentiment scores over future returns. $IC = \\text{corr}(S_t, R_{t+1})$
- **Conditional Value at Risk (CVaR)**: The optimization constraint used to limit tail risk, ensuring the expected loss in the worst 5% of outcomes remains below a strict threshold.

### 4. Expected Insights & Analysis Takeaways
This demo synthesizes the previous 7 phases of research into clear conclusions:
- **Adaptive Allocation Works**: The regime simulation proves that dynamic cash overlay and sector rotation significantly protect capital during Bear and Volatile regimes.
- **Systemic Risk Precedes Price Crashes**: The stress tests demonstrate that network graph metrics (centrality, density) can flag fragility *before* broad sell-offs occur.
- **LLMs Enhance Quant Finance**: By chaining deterministic quant models with generative AI, we achieve both mathematical rigor and perfect interpretability. The "black box" is opened, making AI decisions explainable to stakeholders.

---

### System Overview

AxiomAlpha is a research-grade quantitative
trading system that models financial markets as:
  📊 Stochastic processes (GARCH, regime switching)
  🌐 Information networks (graph theory)
  🤖 Learnable patterns (ML + NLP)
  🎯 Optimization problems (portfolio theory)

### What Happens In One Pipeline Run

  1. DataAgent validates 30 assets × 1500 days
  2. QuantAgent detects Bull/Bear/Volatile regime
  3. GraphAgent computes network centrality
     and systemic risk for each asset
  4. MLAgent generates directional signals
     (XGBoost AUC=0.545, Random Forest F1=0.78)
  5. NLPAgent processes 13,576 news articles
     and extracts sentiment (lag-1 IC=0.0503)
  6. RiskAgent enforces hard risk limits
     (CVaR ≤ 2.5%, HHI ≤ 0.15)
  7. StrategyAgent synthesizes everything into
     an optimal portfolio with LLM explanation
"""

def string_to_lines(s):
    lines = s.split('\n')
    return [line + '\n' for line in lines[:-1]] + [lines[-1]]

nb['cells'][0]['source'] = string_to_lines(intro_markdown)

with open('notebooks/15_agent_demo.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
