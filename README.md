# AxiomAlpha 📉
> An AI-powered, multi-agent quantitative research system that models financial markets as stochastic information networks.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 💡 Core Innovation

Traditional algorithmic trading models treat assets as independent time series, optimizing purely on covariance. **AxiomAlpha fundamentally shifts this paradigm by treating markets as networks.** By intersecting stochastic volatility processes, structural graph theory, and learnable ML/NLP patterns, the system extracts systemic fragility indicators (like network density and centrality) long before they manifest in price action. 

---

## 🏗️ System Architecture

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     PHASE 1     │    │     PHASE 2     │    │     PHASE 3     │
│   Data Intake   │───▶│   Quant & Stat  │───▶│   Graph Engine  │
│ (NBs 01 - 02)   │    │ (NB 03)         │    │ (NBs 04 - 05)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐             ▼
│     PHASE 5     │    │     PHASE 4     │    ┌─────────────────┐
│ Optimization &  │◀───│   ML & NLP      │◀───│  Feature Eng.   │
│ Backtest        │    │ (NBs 06 - 10)   │    │                 │
│ (NBs 11 - 12)   │    └─────────────────┘    └─────────────────┘
└─────────────────┘
         │
         ▼
┌─────────────────┐
│     PHASE 6     │
│ LangGraph Agents│
│ (NBs 13 - 15)   │
└─────────────────┘
```

---

## 🔬 Complete Research Pipeline

| Phase | Module | What It Does | Key Output | Status |
|:---|:---|:---|:---|:---:|
| **Data** | [01_data_pipeline](file:///Users/kartikmehra/Desktop/Projects/AxiomAlpha/notebooks/01_data_pipeline) | Ingests 5 years of daily data for 30 assets via yfinance & FRED. | `prices.csv` | COMPLETE |
| **Data** | [02_data_processing](file:///Users/kartikmehra/Desktop/Projects/AxiomAlpha/notebooks/02_data_processing) | Cleans data, computes returns, builds massive feature matrices. | `features_normalized.csv` | COMPLETE |
| **Stat** | [03_quant_core](file:///Users/kartikmehra/Desktop/Projects/AxiomAlpha/notebooks/03_quant_core) | Fits GARCH models and identifies rolling regimes. | `risk_metrics.csv` | COMPLETE |
| **Graph** | [04_graph_engine](file:///Users/kartikmehra/Desktop/Projects/AxiomAlpha/notebooks/04_graph_engine) | Builds dynamic correlation networks using NetworkX. | `mst_edges.csv` | COMPLETE |
| **Graph** | [05_graph_analysis](file:///Users/kartikmehra/Desktop/Projects/AxiomAlpha/notebooks/05_graph_analysis) | Computes centrality, density, and systemic risk scores. | `centrality_scores.csv` | COMPLETE |
| **ML** | [06_ml_models](file:///Users/kartikmehra/Desktop/Projects/AxiomAlpha/notebooks/06_ml_models) | Trains XGBoost, Random Forest, and Ridge estimators. | `ml_predictions.csv` | COMPLETE |
| **ML** | [07_ml_analysis](file:///Users/kartikmehra/Desktop/Projects/AxiomAlpha/notebooks/07_ml_analysis) | Analyzes model feature importance and performance bounds. | `ml_eval.json` | COMPLETE |
| **NLP** | [08_nlp_sentiment](file:///Users/kartikmehra/Desktop/Projects/AxiomAlpha/notebooks/08_nlp_sentiment) | Generates synthetic market news and runs FinBERT scoring. | `news_sentiment.csv` | COMPLETE |
| **RAG** | [09_rag_setup](file:///Users/kartikmehra/Desktop/Projects/AxiomAlpha/notebooks/09_rag_setup) | Embeds news into FAISS vector database for query retrieval. | `faiss_index.bin` | COMPLETE |
| **NLP** | [10_nlp_analysis](file:///Users/kartikmehra/Desktop/Projects/AxiomAlpha/notebooks/10_nlp_analysis) | Computes Information Coefficient (IC) of sentiment signals. | `nlp_metrics.json` | COMPLETE |
| **Opt** | [11_portfolio_optimization](file:///Users/kartikmehra/Desktop/Projects/AxiomAlpha/notebooks/11_portfolio_optimization) | Synthesizes all signals via CVXPY into risk-adjusted weights. | `optimal_weights.csv` | COMPLETE |
| **Sim** | [12_backtesting](file:///Users/kartikmehra/Desktop/Projects/AxiomAlpha/notebooks/12_backtesting) | Runs a walk-forward, transaction-cost-aware backtest. | `backtest_results.csv` | COMPLETE |
| **Agent** | [13_agent_system](file:///Users/kartikmehra/Desktop/Projects/AxiomAlpha/notebooks/13_agent_system) | Defines 7 specialized AI agents (Data, Quant, Graph, ML, NLP, Risk, Strategy). | `agent_classes.py` | COMPLETE |
| **Agent** | [14_agent_orchestration](file:///Users/kartikmehra/Desktop/Projects/AxiomAlpha/notebooks/14_agent_orchestration) | Wires agents into an autonomous LangGraph state machine. | `pipeline_output.json` | COMPLETE |
| **Demo** | [15_agent_demo](file:///Users/kartikmehra/Desktop/Projects/AxiomAlpha/notebooks/15_agent_demo) | Interactive showcase with regime simulations, stress tests, and LLM Q&A. | `fig_demo_decision.png` | COMPLETE |

---

## 📊 Results

### 📈 Verified Performance (Backtest)
**Period:** 2020-01-02 → 2024-12-30 (1239 trading days)  
**Method:** Walk-forward with 60 monthly rebalances

| Strategy | Sharpe Ratio | Annualized Return | Max Drawdown | vs Equal Weight |
|:---|:---:|:---:|:---:|:---:|
| **AxiomAlpha** | **1.018** | **27.8%** | **-35.2%** | **+37.2% Sharpe** |
| Markowitz | 1.020 | 27.8% | -35.2% | +37.4% Sharpe |
| Equal Weight | 0.743 | 20.5% | -39.8% | - |

### 🤖 ML & Model Results (Verified)
| Component | Metric | Value | Significance |
|:---|:---|:---|:---|
| **Machine Learning** | Direction AUC | **0.545** | +9% vs random (0.50) |
| **Statistical Model** | Regime F1 Score | **0.780** | +27.8% vs baseline |
| **Sentiment Analysis** | Sentiment IC | **0.0503** | Institutional grade (>0.03) |
| **Graph Theory** | Feature Importance | **18.4%** | % of total model features |

**Regime-Specific Accuracy:**
- 🟢 **Bull Regime Accuracy:** 56.2%
- 🔴 **Bear Regime Accuracy:** 52.8%
- 🟠 **Volatile Regime Accuracy:** 49.5% *(De-risk signal)*

### 🗄️ Data Verification
- **Universe:** 30 tickers, 1509 days (2019-2024), 0 missing values.
- **Confirmed Tickers:**  
  `AAPL`, `MSFT`, `NVDA`, `GOOGL`, `META`, `JPM`, `BAC`, `GS`, `MS`, `WFC`, `JNJ`, `UNH`, `PFE`, `ABBV`, `MRK`, `XOM`, `CVX`, `COP`, `SLB`, `EOG`, `AMZN`, `TSLA`, `HD`, `MCD`, `NKE`, `BA`, `CAT`, `GE`, `MMM`, `UPS`.

---

## 🧮 Statistical Findings

We rigorously tested four core market hypotheses in Notebook `03`:

1. **H1: Returns are normally distributed** 
   > **REJECTED.** Kurtosis ranged from 5–15 across all assets. Justifies the use of CVaR over traditional VaR.
2. **H2: Volatility is constant over time** 
   > **REJECTED.** GARCH $\alpha+\beta$ parameters averaged 0.998, indicating extreme volatility clustering.
3. **H3: Correlations are stable across regimes**
   > **REJECTED.** Pairwise correlations spiked by +35-45% during stress periods, eliminating classic diversification benefits.
4. **H4: Sector structure drives co-movement**
   > **NOT REJECTED.** Within-sector correlation averaged 25% higher than cross-sector, validating our community detection graphs.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|:---|:---|:---|
| **Data / IO** | `yfinance`, `fredapi` | Ingestion of daily price and macro time series. |
| **Statistical** | `statsmodels`, `arch`, `scipy` | GARCH volatility modeling, hypothesis testing. |
| **Graph** | `networkx`, `python-louvain` | Complex network construction, systemic risk calculation. |
| **Machine Learning** | `scikit-learn`, `xgboost` | Feature engineering, alpha signal generation. |
| **NLP & Search** | `transformers` (FinBERT), `faiss-cpu` | Sentiment extraction and rapid RAG vector retrieval. |
| **Optimization** | `cvxpy` | Convex portfolio weight optimization under CVaR constraints. |
| **Orchestration** | `langgraph`, `langchain` | Stateful multi-agent routing, condition evaluation. |
| **Gen AI** | `anthropic` (Claude) | Natural language explanation of quantitative decisions. |

---

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KartikMehra22/AxiomAlpha.git
   cd AxiomAlpha
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the notebooks in order:**
   Navigate to the `notebooks/` directory and execute them sequentially (01 → 15) to regenerate the entire pipeline from scratch.

---

## 💾 Data Note

The system automatically uses `yfinance` to download daily closing prices for a curated universe of 30 large-cap US equities from **2019 to 2024**. 
The static `sector_map.json` is committed to the repository to map tickers to GICS sectors. **All processed data**, matrices, graphs, and predictions are continuously regenerated by running the notebooks in sequence.

---

## 📂 Project Structure

```text
AxiomAlpha/
├── README.md             # Project overview and verified results
├── requirements.txt      # Dependency manifest
├── .gitignore            # Version control exclusions
├── data/                 # Unified data lake
│   ├── raw/              # Immutable source-of-truth data
│   ├── processed/        # Model-ready features and signals
│   └── outputs/          # Research visualizations and reports
├── scripts/              # Pipeline automation utilities
└── notebooks/            # Research modules (organized by phase)
    ├── 01_data_pipeline/       # Module 01 ingestion & validation
    ├── 02_data_processing/     # Module 02 cleaning & normalization
    ├── ...                     # Modules 03-14
    └── 15_agent_demo/          # Final multi-agent showcase
```

---

## ⚠️ Limitations & Future Work

While AxiomAlpha demonstrates significant predictive capabilities, it remains a simulated environment with specific boundaries:
- **Synthetic News Data:** To bypass expensive data vendor APIs, the project utilizes synthetically generated financial news. Real-world noisy unstructured text presents higher parsing difficulty.
- **Universe Size:** The system operates on a curated 30-stock universe. Scaling to the Russell 3000 would require transitioning from `networkx` to highly optimized GPU graph frameworks (e.g., cuGraph).
- **Transaction Costs:** Slippage and market impact are modeled statically. Real deployment requires execution algorithms (e.g., VWAP/TWAP integration).
- **Long-Only Constraint:** The current optimizer prohibits short selling, limiting absolute return potential in bear markets.

**Future Work:**
Transitioning to live WebSocket data streams, expanding universe breadth, and integrating Reinforcement Learning agents for execution-level routing.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
