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

| Phase | Notebook | What It Does | Key Output | Status |
|:---|:---|:---|:---|:---:|
| **Data** | `01_data_pipeline.ipynb` | Ingests 5 years of daily data for 30 assets via yfinance & FRED. | `raw_prices.csv` | ✅ |
| **Data** | `02_data_processing.ipynb` | Cleans data, computes returns, builds massive feature matrices. | `features.csv` | ✅ |
| **Stat** | `03_quant_core.ipynb` | Fits GARCH models and identifies rolling regimes. | `risk_metrics.csv` | ✅ |
| **Graph** | `04_graph_engine.ipynb` | Builds dynamic correlation networks using NetworkX. | `network_edges.csv` | ✅ |
| **Graph** | `05_graph_analysis.ipynb` | Computes centrality, density, and systemic risk scores. | `graph_features.csv` | ✅ |
| **ML** | `06_ml_models.ipynb` | Trains XGBoost, Random Forest, and Ridge estimators. | `ml_predictions.csv` | ✅ |
| **ML** | `07_ml_analysis.ipynb` | Analyzes model feature importance and performance bounds. | `ml_eval.json` | ✅ |
| **NLP** | `08_nlp_sentiment.ipynb` | Generates synthetic market news and runs FinBERT scoring. | `news_sentiment.csv` | ✅ |
| **RAG** | `09_rag_setup.ipynb` | Embeds news into FAISS vector database for query retrieval. | `faiss_index.bin` | ✅ |
| **NLP** | `10_nlp_analysis.ipynb` | Computes Information Coefficient (IC) of sentiment signals. | `nlp_metrics.json` | ✅ |
| **Opt** | `11_portfolio_optimization.ipynb` | Synthesizes all signals via CVXPY into risk-adjusted weights. | `optimal_weights.csv` | ✅ |
| **Sim** | `12_backtesting.ipynb` | Runs a walk-forward, transaction-cost-aware backtest. | `backtest_results.csv` | ✅ |
| **Agent** | `13_agent_system.ipynb` | Defines 7 specialized AI agents (Data, Quant, Graph, ML, NLP, Risk, Strategy). | `agent_classes.py` | ✅ |
| **Agent** | `14_agent_orchestration.ipynb` | Wires agents into an autonomous LangGraph state machine. | `pipeline_output.json` | ✅ |
| **Demo** | `15_agent_demo.ipynb` | Interactive showcase with regime simulations, stress tests, and LLM Q&A. | `fig_demo_decision.png` | ✅ |

---

## 📈 Key Results

| Component | Metric | Value | vs Baseline |
|:---|:---|:---|:---|
| **Machine Learning** | Direction Prediction AUC | **0.545** | +9.0% over random (0.500) |
| **Natural Language** | Sentiment Lag-1 IC | **0.0503** | Institutional grade threshold (>0.03) |
| **Statistical Model** | Regime Classification F1 | **0.78** | +27.8% over static baseline |
| **Graph Theory** | Total Feature Importance | **18.4%** | Unlocks entirely new alpha orthogonal to price |
| **Data Processing** | Synthetic News Articles | **13,576** | Processed via local embedding |

*Regime-Specific Accuracy:*
- 🟢 **Bull Regime Accuracy:** 56.2%
- 🔴 **Bear Regime Accuracy:** 52.8%
- 🟠 **Volatile Regime Accuracy:** 49.5% *(Triggers portfolio de-risking)*
- 📉 **GARCH Volatility Persistence:** ~0.998 *(Highly clustered shocks)*

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
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── create_notebook_15.py
│   ├── update_demo_notebook.py
│   └── update_notebook.py
├── notebook_docs/        # Auto-generated documentation for every notebook
└── notebooks/
    ├── 01_data_pipeline.ipynb
    ├── 02_data_processing.ipynb
    ├── 03_quant_core.ipynb
    ├── 04_graph_engine.ipynb
    ├── 05_graph_analysis.ipynb
    ├── 06_ml_models.ipynb
    ├── 07_ml_analysis.ipynb
    ├── 08_nlp_sentiment.ipynb
    ├── 09_rag_setup.ipynb
    ├── 10_nlp_analysis.ipynb
    ├── 11_portfolio_optimization.ipynb
    ├── 12_backtesting.ipynb
    ├── 13_agent_system.ipynb
    ├── 14_agent_orchestration.ipynb
    ├── 15_agent_demo.ipynb
    └── data/
        ├── raw/          # Raw downloaded tickers
        ├── processed/    # Generated features and signals
        └── outputs/      # Exported publication-ready visualizations
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
