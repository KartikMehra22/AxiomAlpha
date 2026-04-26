# Quant Research Pipeline: Multi-Asset Data Ingestion
**Research Module 1 of 5 | AxiomAlpha Framework**

---

This notebook is the **primary ingestion layer** of the quantitative research pipeline. High-fidelity research demands a data foundation free of survivorship bias, corporate action distortions, and time-series discontinuities.

### Scope & Objectives
1. **Universe Definition** — Constructing a stratified 30-stock investment universe across 6 GICS sectors, ensuring broad cross-sectional coverage.
2. **Market Data Acquisition** — Downloading split- and dividend-adjusted daily close prices, the only correct input for return and risk calculations.
3. **Exogenous Factor Integration** — Augmenting stock data with 5 macroeconomic risk drivers that govern regime shifts (volatility, rates, commodities).
4. **Data Integrity Validation** — Systematic quality-control including missing-value detection, duplicate-date checks, and continuity audits.
5. **Raw Data Persistence** — Immutable storage of all canonical raw files to `data/raw/`, forming the sole source of truth for all downstream phases.

> **Pipeline contract**: Every notebook in Phases 2–5 reads exclusively from outputs produced here. Any change to universe composition or date range requires re-running this notebook first.


---
## 1. Environment Configuration & Library Imports

A reproducible quantitative environment requires explicit dependency management. We use `pathlib.Path` for all file operations to guarantee OS-agnostic path handling—a critical property when the pipeline is ported between local development machines and cloud compute nodes.

The `START_DATE` and `END_DATE` constants define the **in-sample period** (2019–2024), chosen to span a complete market cycle that includes the COVID-19 crash and recovery, the 2022 rate-hiking cycle, and the subsequent AI-driven bull market—providing regime diversity essential for robust model training.


---
## 2. Investment Universe Construction

The investment universe is the fundamental building block of the research system. Rather than using all S&P 500 constituents (survivorship bias risk) or a random sample, we deliberately select 5 representative names from each of 6 GICS sectors.

### Sector Rationale

| Sector | Representative Tickers | Analytical Role |
| :--- | :--- | :--- |
| **Technology** | AAPL, MSFT, NVDA, GOOGL, META | Growth driver; high beta to interest rates and risk appetite. |
| **Financials** | JPM, BAC, GS, BRK-B, V | Systemic backbone; key transmitter of monetary policy shocks. |
| **Healthcare** | JNJ, UNH, PFE, ABBV, LLY | Defensive growth; idiosyncratic regulatory and R&D cycles. |
| **Energy** | XOM, CVX, SLB, COP, NEE | Commodity-linked inflation proxy; geopolitical shock absorber. |
| **Consumer** | AMZN, TSLA, WMT, HD, NKE | Sentiment barometer; spans defensive (WMT) to speculative (TSLA). |
| **Industrials** | BA, CAT, HON, UPS, GE | Global trade and capex cycle proxy; sensitive to supply chains. |

The `SECTOR_MAP` dictionary (ticker → sector) is serialised to `sector_map.json` and used in Phases 3–5 for community detection and cross-sector causality analysis.


---
## 3. Market Data Acquisition: Dividend & Split-Adjusted Close Prices

### Why Adjusted Prices?

Raw (unadjusted) closing prices contain artificial discontinuities introduced by stock splits and dividend distributions. For example, a 4-for-1 split instantaneously halves the price on the record date—any return computed across that date would be wildly distorted.

We use the **Adjusted Close** series $P_{adj}$, which applies a retrospective correction factor to all historical prices:

$$P_{adj,\ t} = P_{raw,\ t} \times \prod_{i=t}^{T} k_i$$

where $k_i$ is the corporate-action adjustment factor at event date $i$. This ensures that **log-returns** $r_t = \ln(P_{adj,t} / P_{adj,t-1})$ reflect true economic total returns (excluding transaction costs), making them suitable for volatility estimation, correlation analysis, and signal construction.

### Data Quality Contract
- We download all 30 tickers in a single batch call to minimize API overhead.
- Any ticker returning all-NaN is flagged as `[FAIL]` but never crashes the pipeline.
- The resulting price matrix has shape `(T, 30)` where `T ≈ 1,500` trading days.


---
## 4. Exogenous Risk Factors: Macroeconomic Indicators

A purely endogenous model—one that sees only stock price history—is blind to the macro regime driving correlated moves across the entire universe. We augment the dataset with 5 exogenous indicators:

| Symbol | Clean Name | Economic Role |
| :--- | :--- | :--- |
| `^VIX` | VIX | Implied volatility of S&P 500 options. High VIX = risk-off regime. |
| `^GSPC` | SP500 | Market-wide beta reference for CAPM residual calculation. |
| `^TNX` | TNX | 10-Year US Treasury yield = risk-free rate $R_f$. Drives equity risk premium (ERP = $E[r] - R_f$). |
| `GC=F` | GOLD | Safe-haven demand proxy; negatively correlated with risk appetite. |
| `CL=F` | OIL | WTI Crude = energy cost + global industrial demand indicator. Drives Energy sector idiosyncratic risk. |

These five signals form the **macro feature backbone** used in Phase 3 for regime detection and in Phase 5 for graph-based contagion analysis (e.g., how VIX spikes propagate across the correlation network).


---
## 5. Raw Data Visualisation

Before any cleaning or transformation, we visually inspect the raw time series to catch anomalies that summary statistics alone may miss:

- **Flat-line segments** → indicate data provider outages or stale prices.
- **Extreme single-day spikes** → may be erroneous ticks that inflate volatility estimates.
- **Structural breaks** → e.g. the COVID crash in March 2020 or the 2022 inflation shock.

### What to Look For in the Charts
- **Stock panel**: NVDA and TSLA should show multi-fold appreciation over the 2019–2024 window; defensive names (WMT, JNJ) should be more stable.
- **Macro panel**: VIX should spike above 40 in March 2020 (COVID) and above 30 in 2022 (rate shock). TNX should rise sharply from ~1.5% in 2021 to ~5% in 2023. Gold should show a step-up post-2020.

Deviations from these expected patterns indicate data quality issues requiring investigation.


---
## 6. Systematic Data Quality Validation

Before persisting any files, we run a three-stage quality-control pass using `assert` statements. A failing assertion immediately surfaces the exact issue rather than propagating corrupted data silently into downstream models.

### Quality Checks

| Check | Threshold | Rationale |
| :--- | :--- | :--- |
| **Missing values** | > 1% per ticker triggers a warning | Sparse tickers inflate portfolio variance and distort correlation estimates. |
| **Duplicate dates** | Zero tolerance | Duplicate index entries corrupt `.pct_change()`, `.shift()`, and all rolling operations. |
| **Date continuity** | No gap > 5 calendar days | Gaps beyond typical holiday clusters indicate data provider failures or ticker delistings. |

A ticker with > 5% missing data will be **dropped** from the universe entirely to preserve downstream model integrity.


---
## 6a. Missing Data Heatmap — Pre-Treatment

The heatmap below maps the full time-series range (x-axis) against all 30 tickers (y-axis). Each cell is coloured by data presence:

- **Dark cell** → price data present.
- **White/light streak** → missing data at that date for that ticker.

### Interpretation
- Vertical white bands (spanning many tickers at the same date) → market closure or API failure.
- Horizontal white streaks (single ticker, extended period) → delayed listing or data provider gap.

This visual diagnostic guides decisions on whether to forward-fill, backward-fill, or drop a ticker entirely. Tickers with isolated, infrequent gaps are candidates for forward-filling; those with systematic long-run absences should be dropped.


---
## 6b. Missing Value Treatment

We apply a two-pass imputation strategy consistent with **no-look-ahead** constraints:

1. **Forward Fill (`.ffill()`)**: Carries the last known price forward into the gap. This is the industry-standard treatment for daily equity prices—it reflects the economic reality that if a market is closed or a price is unavailable, the best estimate of the current value is the last observed price.

2. **Backward Fill (`.bfill()`)**: Applied only to fill any residual NaN values at the *beginning* of a series (where ffill cannot operate). This is a pragmatic choice for short warm-up gaps; series with > 5% missing are dropped before this step.

> **Look-ahead bias note**: Forward-fill is safe here because we are imputing intra-series gaps, not future-to-past. The `bfill` applied to start-of-series gaps is also safe because these dates precede our training window start.


---
## 6c. Missing Data Heatmap — Post-Treatment

The heatmap should now appear **entirely solid** (no white streaks), confirming that all gaps have been resolved. If any gaps remain visible, the corresponding ticker should be investigated and potentially dropped from the universe.

### Visual Confirmation Criteria
- No vertical white bands remaining.
- No horizontal white streaks for any ticker.
- Uniform colour density across the full date range.

A clean post-treatment heatmap is the **final quality gate** before raw data is persisted to disk. Only data that passes this gate is written to `data/raw/prices.csv`.


---
## 7. Raw Data Persistence — Immutability Principle

All raw artifacts are written to `data/raw/` and treated as **immutable canonical sources**. No downstream notebook modifies these files. This design pattern—sometimes called "raw data lake" architecture—ensures:

- **Reproducibility**: Any phase can be re-run from scratch by reading from `data/raw/`.
- **Audit trail**: Raw data is never overwritten, preserving the original download for debugging.
- **Decoupling**: Processing logic (Phase 2+) is independent of ingestion logic (Phase 1).

| Output File | Format | Contents | Size (approx.) |
| :--- | :--- | :--- | :--- |
| `prices.csv` | CSV | Adjusted daily close for 30 tickers | ~500 KB |
| `macro.csv` | CSV | 5 macro indicators (VIX, SP500, TNX, GOLD, OIL) | ~80 KB |
| `sector_map.json` | JSON | Ticker → GICS sector mapping | < 1 KB |


---
## Phase 1 Summary: Data Pipeline Complete

### What Was Ingested

| Dataset | Coverage | Storage |
| :--- | :--- | :--- |
| Stock Prices | 30 equities × 6 sectors × ~1,500 trading days | `data/raw/prices.csv` |
| Macro Indicators | VIX, SP500, TNX, GOLD, OIL × ~1,500 days | `data/raw/macro.csv` |
| Sector Map | 30 ticker → GICS sector assignments | `data/raw/sector_map.json` |

### Key Analytical Properties Established
- **Time range**: 2019–2024 covers **three distinct macro regimes**: pre-COVID expansion, pandemic shock+recovery, and post-COVID rate-hiking cycle—essential for regime-robust model training.
- **Universe breadth**: 30 tickers across 6 sectors ensures the correlation matrix (30×30 = 900 pairs) contains meaningful cross-sector structure for graph construction in Phase 4.
- **Data integrity**: All tickers validated for missing values, duplicate dates, and continuity. Forward-fill applied for isolated gaps; gapped tickers dropped if > 5% missing.
- **Macro augmentation**: 5 exogenous risk factors capture the macro environment that drives correlated stock moves and network density changes over time.

### What the Data Reveals
Even without any modelling, the raw time series tells a clear story:
- **Correlation clustering** is visible in price action: Technology stocks move together during risk-on/risk-off episodes.
- **VIX spikes** (COVID March 2020, rate-hike panic 2022) compress cross-asset correlations toward 1—exactly the "correlation breakdown" phenomenon that graph-based models (Phase 4) are designed to detect and measure.
- **Defensive vs. cyclical divergence** (WMT, JNJ vs. NVDA, TSLA) is already apparent, motivating the community-detection analysis in Phase 5.

### Next Phase: Feature Engineering (Phase 2)

| Feature Group | Description |
| :--- | :--- |
| Log Returns | $r_t = \ln(P_t / P_{t-1})$; the fundamental input to all statistical models. |
| Rolling Volatility | 20-day and 60-day realised volatility as risk features. |
| Technical Signals | RSI, MACD, Bollinger Band z-scores. |
| Macro Regime Flags | VIX-based regime classification (calm / elevated / crisis). |

> **Phase 2 notebook**: `02_data_processing.ipynb`
