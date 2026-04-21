# AxiomAlpha: AI-Powered Quant Research System

A professional-grade quantitative research platform for cross-sectional and time-series equity analysis.

## 📁 Repository Structure

```text
AxiomAlpha/
├── data/
│   ├── raw/            # Immutable raw data (prices, macro, maps)
│   ├── processed/      # Feature-engineered datasets
│   └── outputs/        # Backtest results, plots, and models
├── notebooks/          # Step-by-step research pipeline
├── src/                # Modular system components
│   ├── data_layer/     # Ingestion & cleaning
│   ├── quant_core/     # Math, risk, and portfolio ops
│   ├── graph_engine/   # Relationship mapping
│   ├── ml_layer/       # Prediction models
│   ├── nlp_layer/      # Sentiment & text analysis
│   └── agents/         # AI Research Agents
├── configs/            # System settings & parameters
├── requirements.txt    # Environment dependencies
└── README.md
```

## 🚀 Research Pipeline

1.  **Phase 1: Data Pipeline** (`notebooks/01_data_pipeline.ipynb`)
    *   Multi-sector universe ingestion (30 tickers).
    *   Macro indicator synchronization.
    *   Visual data profiling & Heatmap-based missing value handling.
2.  **Phase 2: Feature Engineering** (Upcoming)
3.  **Phase 3: Alpha Generation** (Upcoming)

## 🛠 Setup

```bash
pip install -r requirements.txt
```
