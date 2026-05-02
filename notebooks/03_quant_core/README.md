# Quantitative Core: Statistical Hypothesis Testing

## Research Purpose
|
| **H1** | Returns are Normally distributed | Jarque-Bera, Student-t fit, QQ-plot | **REJECT** — fat tails present |
| **H2** | Volatility is constant over time | Ljung-Box on $r_t^2$, GARCH persistence | **REJECT** — ARCH effects confirmed |
| **H3** | Correlations are stable across regimes | Rolling correlation vs VIX, T-test | **REJECT** — correlation rises in stress |
| **H4** | Within-sector > cross-sector correlation | ANOVA, correlation block analysis | **FAIL TO REJECT** — sector structure valid |

> **Data contract**: All analysis reads from `data/processed/log_returns.csv` and `data/processed/features_normalized.csv` generated in Phase 2. No new data is downloaded here.

## Quantitative Logic & Methodology
## 1. Environment Setup & Data Loading

We import the statistical testing suite on top of the standard scientific stack:

- `scipy.stats` — Jarque-Bera normality test, t-test, ANOVA, Student-t fitting.
- `statsmodels` — Ljung-Box test for serial correlation in squared returns.
- `arch` — GARCH(1,1) model fitting for volatility persistence measurement.

The log-returns matrix `(T × 30)` is the primary input. All 30 tickers and their GICS sector assignments are loaded from the sector map.


## Key Mathematical Formulas
$$\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2$$

$$H_1: \bar{\rho}_{\text{within}} > \bar{\rho}_{\text{cross}} \quad \text{(sectors are structurally distinct)}$$

$$H_1: \rho_{\text{stress}} > \rho_{\text{calm}} \quad \text{(correlations rise in stress regimes)}$$

$$\text{VaR}^{t-\text{dist}}_{1\%} = \mu + t_{\nu,\ 0.01} \cdot \sigma$$

$$H_0: \text{Var}(r_t | r_{t-1}, r_{t-2}, ...) = \sigma^2 \quad (\text{constant, i.i.d.})$$

$$H_1: \text{Var}(r_t | r_{t-1}, r_{t-2}, ...) = \sigma_t^2 \quad (\text{time-varying, ARCH effects})$$

$$H_0: \rho_{\text{stress}} = \rho_{\text{calm}} \quad \text{(correlations are regime-invariant)}$$

$$H_0: \bar{\rho}_{\text{within}} = \bar{\rho}_{\text{cross}} \quad \text{(no sector differentiation)}$$

$$\text{VaR}^{\text{Historical}}_{1\%} = \text{1st percentile of empirical distribution}$$

$$\text{VaR}^{\text{Gaussian}}_{1\%} = \mu + 2.326\sigma$$

$$LB(m) = n(n+2) \sum_{k=1}^{m} \frac{\hat{\rho}_k^2}{n-k} \sim \chi^2(m)$$

$$F = \frac{\text{Between-group variance}}{\text{Within-group variance}} \sim F(k-1, N-k)$$

$$JB = \frac{n}{6}\left[ S^2 + \frac{(K-3)^2}{4} \right] \sim \chi^2(2)$$

