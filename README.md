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

### 📈 Live Performance

> Every headline number lives in **[`data/processed/RESULTS.md`](data/processed/RESULTS.md)** and is regenerated by `python run_all.py`. Reading it from there (and not from this README) is the only way to be sure the numbers haven't drifted.
>
> The README intentionally does not duplicate metric values — that's how prior versions of this file ended up with figures that didn't match the artifacts.

### 🗄️ Data

- **Universe:** 30 large-cap US equities (see [`configs/settings.py:TICKERS`](configs/settings.py)).
- **Date range:** 2019-01-01 to 2024-12-31 (see `START_DATE` / `END_DATE` in `configs/settings.py`).
- **Trading days:** ~1500 daily observations after market-calendar filtering (see `RESULTS.json["n_trading_days"]` for the exact count).
- **Missing values:** 0 in the cleaned `prices.csv` (NB01 forward-fills + drops post-IPO holes).

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
| **Orchestration** | `langgraph` | Stateful multi-agent routing skeleton (NB14). |
| **NLP Embeddings** | `sentence-transformers` | Real embeddings (all-MiniLM-L6-v2) for synthetic-news RAG. |

`langchain` and `anthropic` are intentionally not dependencies — agent explanations are templated rather than LLM-generated.

---

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KartikMehra22/AxiomAlpha.git
   cd AxiomAlpha
   ```
2. **Install dependencies (Python 3.11+):**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set environment variables.** Copy [`.env.example`](.env.example) to `.env` and fill in:
   - `FRED_API_KEY` — required by NB01 for macro data ([free key](https://fred.stlouisfed.org/docs/api/api_key.html))
4. **Run the pipeline:**
   ```bash
   python run_all.py                    # full end-to-end (15 notebooks, ~10-30 min)
   python run_all.py --from 02          # skip NB01 (uses cached data/raw/)
   python scripts/check_pipeline.py     # static smoke test (no fabricated data)
   python scripts/check_determinism.py  # back-to-back reproducibility check
   pytest                               # unit + property tests
   ```

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
├── configs/settings.py     # Centralized config (tickers, seeds, hyperparameters)
├── src/io/loaders.py       # Single source of truth for every read
├── scripts/check_pipeline.py     # Smoke test for random-data regressions
├── scripts/check_determinism.py  # Back-to-back reproducibility check
├── run_all.py              # papermill orchestrator: one command runs everything
├── tests/                  # pytest suite for loaders & checks
├── data/processed/RESULTS.md     # Auto-generated headline numbers
├── docs/superpowers/       # Specs and implementation plans
├── scripts/              # Pipeline automation utilities
└── notebooks/            # Research modules (organized by phase)
    ├── 01_data_pipeline/       # Module 01 ingestion & validation
    ├── 02_data_processing/     # Module 02 cleaning & normalization
    ├── ...                     # Modules 03-14
    └── 15_agent_demo/          # Final multi-agent showcase
```

---

## ⚠️ Limitations & Known Gaps

This is a research environment with explicit boundaries. The points below are also enforced by tests / static checks where possible.

**Data**
- **Synthetic news (NB08).** Headlines are templated. FinBERT scoring and sentence-transformer embeddings are real, but the underlying text is generated. Lead-lag analysis between sentiment and price is therefore informative about *the FinBERT model's behavior*, not about real-world media-driven alpha.
- **30-ticker universe.** Scaling beyond ~100 names requires moving from in-process `networkx` to a streaming/GPU graph stack (e.g. cuGraph).
- **NB01 non-determinism.** yfinance returns slightly different bytes across calls (split-adjustment backfills), so `scripts/check_determinism.py` runs `--from 02` against a frozen `data/raw/`.

**Modeling**
- **Long-only optimizer.** Short selling is disabled in `configs/settings.py:OPT_LONG_ONLY`.
- **Static transaction costs.** 5 bps round-trip; no slippage / market-impact model.
- **Walk-forward, but with shared full-sample artifacts.** NB12's walk-forward uses fold-local mu/cov, but `regimes` and `systemic_risk_scores` are computed once on the full sample (NB03 / NB05). Removing this last bit of look-ahead is on the roadmap.
- **NB05 graph features.** Currently passed through as zeros into the ML feature matrix. Computing real per-day centrality features from NB04's dynamic graph is queued as Tier-2 work.
- **NB14 orchestration is scaffolded, not running.** The LangGraph node bodies are placeholders; the agent state-machine wires the graph but doesn't actually execute the per-agent logic yet.

**Reproducibility**
- The pipeline is deterministic from NB02 onward, given a frozen `data/raw/`. `scripts/check_determinism.py` will hash CSV / NPY / JSON outputs across two runs and fail on any byte difference (timestamps in JSON are stripped before hashing — see `VOLATILE_JSON_KEYS`).
- All model RNGs derive from `configs.settings.RANDOM_SEED`. Hardcoded `random_state=42` literals are forbidden by the static check.

---

## ✅ Honesty Notes

- All headline numbers come from the live pipeline (`data/processed/RESULTS.md`), regenerated each run. The README does not duplicate them, by design.
- Synthetic news (NB08): the headlines are synthetic; FinBERT scoring and sentence-transformer embeddings are real.
- Agent explanations (NB13–15) are templated, not LLM-generated. The `anthropic` and `langchain` packages are intentionally not used.
- Determinism: bit-identical CSV/NPY/JSON across runs, *given a frozen `data/raw/`*. NB01 itself depends on yfinance and is therefore excluded — see `scripts/check_determinism.py`.
- Backtest assumes 5 bps transaction cost, no slippage modeling, long-only constraint.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
