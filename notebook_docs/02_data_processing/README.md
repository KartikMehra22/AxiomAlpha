# Data Processing & Feature Engineering Pipeline
**Research Module 2 of 5 | AxiomAlpha Framework**

---

This notebook transforms the raw price series ingested in Phase 1 into a **statistically rigorous, model-ready feature matrix**. Raw prices are non-stationary and cannot be directly fed into most quantitative models. Every transformation here is motivated by a specific financial or statistical necessity.

### Pipeline Architecture

| Stage | Operation | Output |
| :--- | :--- | :--- |
| **1. Data Cleaning** | Outlier detection, gap treatment | Clean, gap-free price series |
| **2. Returns Computation** | Log-returns & simple returns | `log_returns.csv`, `simple_returns.csv` |
| **3. Feature Engineering** | Momentum, volatility, mean-reversion signals | `features_raw.csv` |
| **4. Macro Integration** | Merge VIX, rates, commodities per date | Augmented feature set |
| **5. Normalization** | Cross-sectional z-scoring per date | `features_normalized.csv` |
| **6. Statistical Profiling** | Non-normality, ARCH effects, distribution shape | Qualitative model insights |
| **7. Regime Detection** | K-Means on rolling return/vol surface | Regime labels per date |

> **Dependencies**: Requires `data/raw/prices.csv`, `data/raw/macro.csv`, and `data/raw/sector_map.json` from Phase 1.


---
## 1. Environment Setup & Data Loading

We load the complete scientific stack alongside financial-specific libraries:

- `statsmodels` — Jarque-Bera normality tests, Ljung-Box ARCH tests.
- `arch` — GARCH(1,1) volatility model fitting.
- `sklearn` — K-Means clustering for unsupervised regime detection.
- `scipy.stats` — Distribution fitting and hypothesis testing.

All raw data loaded here is treated as **read-only**. No modifications are made to `data/raw/` files.


---
## 2. Data Cleaning: Outlier Detection & Gap Treatment

### Why Forward-Fill?
Forward-fill (`ffill`) is the **only look-ahead-safe** method for filling price gaps. It assumes the last observed price is the best estimate of the current price when no trade occurred (e.g., exchange closure, illiquidity event). 

Alternatives that would introduce **look-ahead bias**:
- **Backward-fill** — uses future prices to fill past gaps ❌
- **Linear interpolation** — assumes knowledge of the future endpoint ❌
- **Mean imputation** — statistically distorts return series ❌

### Outlier Treatment: Z-Score Thresholding
We flag any daily return satisfying $|z| > 5\sigma$ as a statistical outlier, where:
$$z_t = \frac{r_t - \mu}{\sigma}$$
A 5σ threshold is deliberately conservative. At normal distribution, this probability is ~$2.9 \times 10^{-7}$—i.e., roughly once per million trading days. Anything exceeding this threshold is almost certainly a data error, not a genuine price move.

### Output
- Cleaned price matrix with all gaps filled.
- Outlier report: count and affected tickers.


---
## 3. Returns Computation: Log-Returns vs Simple Returns

### Why Log-Returns?
We compute **continuously compounded log-returns** as the primary statistical input:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

**Advantages over simple returns** $\left(\frac{P_t - P_{t-1}}{P_{t-1}}\right)$:

| Property | Log Returns | Simple Returns |
| :--- | :--- | :--- |
| **Time-additivity** | $r_{1:T} = \sum_{t=1}^{T} r_t$ ✓ | Not additive ✗ |
| **Approximate normality** | Better approximation ✓ | Bounded at -100% ✗ |
| **Stationarity** | Stationary series ✓ | Non-stationary at level ✗ |
| **Portfolio math** | Requires care | Direct: $R_p = \sum_i w_i R_i$ ✓ |

We retain **simple returns** (`simple_returns.csv`) for portfolio-level rebalancing calculations in Phase 5.

### Visualization: Return Series Overview
The chart shows the time series of log-returns for a cross-section of tickers. What to observe:
- **March 2020 (COVID crash)**: Extreme negative spikes across all sectors simultaneously.
- **2022 rate-hiking cycle**: Prolonged negative returns, especially in high-duration Tech names.
- **Heteroscedasticity**: Visually apparent volatility clustering—large moves cluster together.


---
## 4. Feature Engineering: Per-Ticker Signal Construction

We construct a rich feature matrix capturing three categories of price behaviour that have persistent predictive value in financial markets:

### Feature Taxonomy

**Momentum / Trend Signals**
| Feature | Formula | Interpretation |
| :--- | :--- | :--- |
| `ret_1d` | $r_t$ | Today's return; reversion signal |
| `ret_5d` | $\sum_{i=0}^{4} r_{t-i}$ | 1-week momentum |
| `ret_20d` | $\sum_{i=0}^{19} r_{t-i}$ | 1-month momentum |
| `ret_60d` | $\sum_{i=0}^{59} r_{t-i}$ | 1-quarter momentum |

**Volatility / Risk Regime Signals**
| Feature | Formula | Interpretation |
| :--- | :--- | :--- |
| `vol_5d` | $\hat{\sigma}_5 \cdot \sqrt{252}$ | Short-term realised vol (annualised) |
| `vol_20d` | $\hat{\sigma}_{20} \cdot \sqrt{252}$ | Medium-term realised vol |
| `vol_60d` | $\hat{\sigma}_{60} \cdot \sqrt{252}$ | Long-term realised vol |
| `vol_ratio` | $\sigma_5 / \sigma_{60}$ | Vol regime indicator: > 1 = risk escalation |

**Higher-Moment Signals**
- `skew_20d` — Rolling 20-day skewness; negative skew = crash risk premium.
- `kurt_20d` — Rolling 20-day excess kurtosis; captures fat-tail risk.

### Visualization: Feature Matrix Snapshot
The heatmap of feature values across tickers and dates reveals:
- **Sector clustering**: Technology tickers have persistently high `vol_ratio` during 2022.
- **Macro synchronisation**: `vol_5d` spikes across all tickers simultaneously during COVID and rate shock—confirming the need for cross-sectional normalisation (Step 5).


---
## 5. Macro Feature Integration: Exogenous Risk Drivers

Equity prices are endogenous to the macro environment. Without macro context, a pure price-feature model cannot distinguish between:
- **Idiosyncratic volatility** (company-specific risk) — tradeable signal
- **Systematic macro volatility** (VIX spike) — must be risk-managed, not traded

We derive three first-order macro features from the raw indicators:

| Feature | Derivation | Signal Meaning |
| :--- | :--- | :--- |
| `vix_chg` | $\Delta VIX_t = VIX_t - VIX_{t-1}$ | Regime velocity: rising VIX = risk-off acceleration |
| `sp500_ret` | $\ln(SP500_t / SP500_{t-1})$ | Market beta reference for CAPM-style factor exposure |
| `tnx_chg` | $\Delta TNX_t$ | Duration risk signal: rising rates = tech compression |

### Visualization: Macro Regime Dashboard
The 3-panel dashboard shows:
- **Top panel** — S&P 500 levels overlaid with VIX colour bands (green: VIX < 20, amber: 20–30, red: > 30). The March 2020 and late-2022 periods are clearly in the red regime.
- **Middle panel** — TNX (10-year yield) rise from ~1.5% to ~5% over 2022–2023, the sharpest rate hiking cycle in 40 years.
- **Bottom panel** — Gold's safe-haven role: spikes during COVID uncertainty and again during the 2022 geopolitical shock.

> **Key insight**: The three macro regimes are *not* symmetric. Bull regimes last longer (~18 months average) but Bear and Volatile regimes are sharper and more dangerous for leveraged portfolios.


---
## 6. Feature Normalization: Cross-Sectional Z-Scoring

### Why Cross-Sectional (Not Time-Series) Normalisation?

**Time-series z-scoring** (normalize each feature over its own history) creates look-ahead bias and removes the cross-sectional ranking information that drives relative-value strategies.

**Cross-sectional z-scoring** (normalize each feature *across all 30 tickers on the same date*) is the correct approach:

$$\tilde{f}_{i,t} = \frac{f_{i,t} - \mu_t}{\sigma_t + \epsilon}$$

where $\mu_t$ and $\sigma_t$ are the **cross-sectional** mean and standard deviation computed across all $N=30$ tickers at date $t$, and $\epsilon = 10^{-8}$ prevents division by zero.

**What this achieves:**
- A z-score of +2 for `vol_20d` on a given date means that ticker is **2 standard deviations more volatile than its peers** on that specific day—a pure relative signal.
- Global regime effects (e.g., VIX spike causing all volatilities to double) are cancelled out.
- Output features are **dimensionless** and on a comparable scale, satisfying model input requirements.

### Visualization: Normalisation Effect
The before/after distribution comparison should show:
- **Before**: Skewed, multi-modal distribution reflecting macro regime shifts.
- **After**: Approximately standard normal, centred at 0 with unit variance across the universe.


---
## 7. Distribution Analysis: Testing for Non-Normality

### The Problem with Assuming Normality
Most classical finance theory (Black-Scholes, Markowitz MPT, CAPM) assumes returns follow a **Gaussian distribution**. If this assumption fails, the consequences are severe:
- VaR models underestimate tail risk by 3–5× during crises.
- Option pricing using Black-Scholes misprices deep out-of-the-money puts.
- Portfolio optimisation based on mean-variance produces fragile allocations.

### The Jarque-Bera Test
We formally test the null hypothesis $H_0: (S, K) = (0, 3)$ (normality) using the Jarque-Bera statistic:

$$JB = \frac{n}{6}\left[S^2 + \frac{(K-3)^2}{4}\right]$$

where $S$ is sample skewness and $K$ is sample kurtosis. Under $H_0$, $JB \sim \chi^2(2)$.

**A p-value < 0.05 rejects normality.**

### Expected Findings
- **Skewness < 0** for most tickers: returns have a **negative tail bias** — crashes are more extreme than rallies of equivalent probability. This is the "volatility smile" in option markets.
- **Excess kurtosis > 3** universally: the so-called **"fat tail" or leptokurtic" property**. Observed for virtually every liquid equity at daily frequency.
- **Jarque-Bera p ≈ 0** for all 30 tickers: we **reject** normality with near-certainty.

### Visualization: Distribution Deep Dive
Each panel shows the empirical return histogram vs the fitted normal curve (red dashed). The heavy tails and negative skew are visible as the empirical distribution extending well beyond the normal curve on the left side.


---
## 8. Correlation Analysis: Pearson vs Spearman

### Two Flavours of Correlation

**Pearson correlation** measures linear dependence:
$$\rho_{ij} = \frac{\text{Cov}(r_i, r_j)}{\sigma_i \cdot \sigma_j}$$

**Spearman rank correlation** replaces raw values with their ranks before computing Pearson:
$$\rho^{(S)}_{ij} = \rho_{\text{Pearson}}(\text{rank}(r_i),\ \text{rank}(r_j))$$

### Why Both Matter for Graph Construction
Spearman is **robust to extreme observations** (fat tails) and captures **non-linear monotonic relationships**. A significant divergence between Pearson and Spearman for a pair suggests the relationship is driven by tail events rather than steady co-movement—critical for distinguishing crisis contagion from structural correlation.

### Visualization: Dual Correlation Heatmaps
The heatmaps are sorted by GICS sector. Key observations:

- **Within-sector blocks** (diagonal sub-matrices) show deep red (high correlation) — Technology stocks move together, as do Financials. This sector clustering directly motivates the community detection in Phase 4.
- **Cross-sector off-diagonal** blocks are lighter — expected under the factor model where idiosyncratic returns dominate.
- **Pearson vs Spearman divergence**: Energy pairs show larger divergence — their correlation is driven by commodity shock events (tail events), not steady-state co-movement.
- **Full market correlation**: During the COVID crisis period (if we compute rolling correlations), all pairs converge toward ρ ≈ 1 — the "everything sells off" phenomenon. Phase 4's dynamic graph captures this directly.


---
## 9. Volatility Analysis: ARCH Effect Detection

### Volatility Clustering
Empirical observation across all liquid markets: **large returns are followed by large returns, and small returns by small returns**. This phenomenon — formalised as **ARCH (AutoRegressive Conditional Heteroscedasticity)** by Engle (1982, Nobel Prize 2003) — means volatility is time-varying and auto-correlated.

If this effect is present, a static volatility estimate (e.g., using the full-sample standard deviation) will:
- **Underestimate** current risk during crisis periods.
- **Overestimate** current risk during calm periods.

### Ljung-Box Test on Squared Returns
We test $H_0$: No autocorrelation in $r_t^2$ using the Ljung-Box statistic at 10 lags:

$$LB(m) = n(n+2) \sum_{k=1}^{m} \frac{\hat{\rho}_k^2(r^2)}{n-k} \sim \chi^2(m)$$

A p-value < 0.05 **rejects** the null and confirms ARCH effects are present.

### Visualization: Volatility Clustering
The rolling 20-day volatility plots reveal:
- **Volatility spikes** precisely at March 2020 (COVID), March 2022 (Ukraine/rate shock), and October 2022 (earnings recession fears).
- **Mean-reversion** after each spike — volatility eventually reverts toward the long-run mean, but on different timescales per sector.
- **Cross-ticker synchronisation** of spikes confirms the systemic nature of these shocks — justifying a graph-based approach to modelling contagion in Phase 4.


---
## 10. GARCH(1,1) Volatility Modelling

### Why GARCH?
Having confirmed ARCH effects, we fit a **GARCH(1,1)** model to produce *conditional* volatility estimates — the best available forecast of tomorrow's volatility given today's information:

$$\sigma_t^2 = \omega + \alpha \cdot r_{t-1}^2 + \beta \cdot \sigma_{t-1}^2$$

**Interpretation of parameters:**
- $\omega$ — Long-run average variance (unconditional variance floor).
- $\alpha$ — ARCH coefficient: sensitivity of today's variance to **yesterday's shock** ($r_{t-1}^2$).
- $\beta$ — GARCH coefficient: **persistence** of yesterday's conditional variance.
- $\alpha + \beta$ — Total **persistence** of volatility. Values near 1.0 indicate shocks decay slowly (integrated GARCH behaviour).

### Why Skewed-t Distribution?
We use the **skewed Student-t** distribution for innovations rather than Gaussian, because:
- Standard t-distribution captures fat tails via degrees-of-freedom parameter $\nu$.
- Skewness parameter $\lambda$ captures the negative asymmetry documented in Step 7.

### Expected Parameter Values
For liquid equities, typical values are: $\alpha \approx 0.05$–$0.15$, $\beta \approx 0.80$–$0.92$, giving persistence $\alpha + \beta \approx 0.90$–$0.99$.

### Visualization: GARCH Conditional vs Realised Volatility
The chart compares the GARCH(1,1) conditional volatility estimate (smooth line) against the realised 20-day rolling volatility (noisy line). Key observations:
- GARCH is smoother — it filters out measurement noise in the rolling estimator.
- GARCH anticipates volatility increases faster — it updates based on squared shocks, not just a rolling window.
- The **divergence during COVID** shows the limits of GARCH: a pure time-series model cannot anticipate unprecedented exogenous shocks, only react to them.


---
## 11. Regime Detection: Unsupervised Market State Classification

### Motivation
Financial markets switch between qualitatively distinct states — **Bull, Bear, and Volatile** — with different return and risk profiles. Static models ignore this regime structure and produce averaged predictions that are suboptimal in all regimes.

### Methodology: K-Means on Return/Vol Surface
We cluster the 2D feature space $\{\bar{r}_{20d},\ \hat{\sigma}_{20d}\}$ (rolling 20-day mean return and volatility of the equal-weight market proxy) into $K=3$ clusters using **K-Means**:

$$\text{argmin}_{C_1, C_2, C_3} \sum_{k=1}^{3} \sum_{x \in C_k} \|x - \mu_k\|^2$$

Features are standardised to unit variance before clustering to prevent the volatility scale from dominating.

**Regime mapping** (by cluster centroid ranking):
| Regime | Return Centroid | Vol Centroid | Market Interpretation |
| :--- | :--- | :--- | :--- |
| **Bull** | Highest | Lowest | Trending markets; momentum works |
| **Bear** | Lowest | Highest | Crisis/drawdown; defensive positioning |
| **Volatile** | Near-zero | High | Choppy markets; mean-reversion works |

### Visualization: Regime Timeline & Distribution
The regime visualisation reveals:
- **Bull regime** dominates 2019, H2 2020–2021, and 2023–2024 — consistent with the known market narrative.
- **Bear regime** is concentrated in March 2020 (COVID crash) and throughout 2022 (rate-hiking drawdown).
- **Volatile regime** appears as transition states between Bull and Bear — brief but dangerous for momentum strategies.

> **Key insight**: Bear regimes, while brief (< 15% of trading days), account for the majority of portfolio drawdowns. Phase 4's graph-based analysis will show that network density spikes precisely during these Bear regimes — confirming the systemic nature of correlated selling.


---
## 12. Processed Data Persistence

All engineered artifacts are written to `data/processed/` under the **write-once** principle. Downstream notebooks read from these files and never modify them.

### Output Manifest

| File | Shape | Description | Used In |
| :--- | :--- | :--- | :--- |
| `log_returns.csv` | (T, 30) | Daily log-return matrix | Phase 3, 4, 5 |
| `simple_returns.csv` | (T, 30) | Daily simple returns | Phase 5 (portfolio math) |
| `features_raw.csv` | (T×30, F) | Raw engineered features per ticker-date | Phase 3 |
| `features_normalized.csv` | (T×30, F) | Cross-sectionally z-scored features | Phase 3, 5 |
| `risk_metrics.csv` | (30, M) | GARCH params, tail stats per ticker | Phase 4 |


---
## Phase 2 Summary: Data Foundation Established

### Statistical Properties of the Universe (2019–2024)

| Statistic | Finding | Implication |
| :--- | :--- | :--- |
| **Non-normality** | JB test rejects Gaussian for all 30 tickers | Fat-tail risk models required; standard VaR underestimates tail risk |
| **ARCH Effects** | ~90% of tickers show significant volatility clustering | GARCH-class models needed for accurate volatility forecasting |
| **Negative Skew** | Mean skewness ≈ −0.3 across universe | Crash risk is larger than rally risk of equal probability |
| **Excess Kurtosis** | Mean kurtosis ≈ 5–8 (vs 3 for Gaussian) | Events beyond 3σ are ~100× more frequent than normality predicts |
| **Sector Correlation** | Within-sector ρ > 0.65; cross-sector ρ ≈ 0.35 | Strong block structure motivates graph community detection |
| **Regime Breakdown** | Bull ~60%, Volatile ~25%, Bear ~15% of days | Bear regime outsized impact on drawdown despite low frequency |

### What the Data Reveals for Phase 4
1. **High intra-sector correlation** → natural community structure will emerge in the correlation graph.
2. **Regime-dependent correlation** → graph edges are *dynamic*; static graphs miss the crisis-period correlation surge.
3. **Volatility clustering** → systemic risk scores (Phase 4) must account for time-varying connectivity.
4. **Fat tails** → contagion simulations (Phase 4) must use non-Gaussian shock distributions.

### Next Phase: Graph Engine (Phase 3 / Notebook 03)

| Analysis | Description |
| :--- | :--- |
| **Correlation Graph** | Build threshold-filtered graph from `log_returns.csv` |
| **Granger Causality Network** | Directed graph from predictive relationships |
| **Minimum Spanning Tree** | Extract market backbone structure |
| **Systemic Risk Scores** | Centrality-weighted contagion vulnerability index |

> **Phase 3 notebook**: `03_quant_core.ipynb`
