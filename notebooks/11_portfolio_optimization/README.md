# Phase 7: Portfolio Optimization

## Research Purpose


## Quantitative Logic & Methodology
# Phase 7: Portfolio Optimization
## AxiomAlpha — AI Quant Research System

### What We Are Doing
We are translating the raw, asset-level predictive signals (from Phase 5 ML and Phase 6 NLP) into a final set of portfolio weights. We construct three nested optimization strategies: 
1. A classical Markowitz baseline.
2. A constrained AxiomAlpha Base portfolio incorporating tail-risk limits.
3. The complete AxiomAlpha Full portfolio, which penalizes systemic network risk and incorporates regime overlays.

### Why Classical Optimization Is Insufficient
The classic Mean-Variance Optimization (Markowitz, 1952) is mathematically elegant but empirically brittle. 
- **The "Error Maximizer" Problem**: Classical models use historical returns for $\mu$, taking any backward-looking noise and maximizing exposure to it.
- **Normal Distribution Fallacy**: It assumes asset returns are normally distributed, completely ignoring the fat-tailed reality (leptokurtic distributions) of financial markets, leading to catastrophic underestimation of tail risk.
- **Static Covariance**: It assumes correlations are stable. In reality, correlations approach 1.0 during market crashes.

### AxiomAlpha's Mathematical Framework
To solve these structural flaws, we introduce the following mechanisms:

#### 1. Forward-Looking Expected Returns ($\mu$)
Instead of historical means, we use an ensemble:
- $\mu_{ML} = (P_{up} - 0.5) \times 2 \times \sigma \times M_{regime}$
- $\mu_{Sent} = Z_{sentiment} \times R_{base}$

#### 2. Robust Covariance ($\Sigma$)
We use Ledoit-Wolf shrinkage to pull sample covariance matrix $S$ toward a highly structured target $F$:
- $\Sigma_{LW} = \delta F + (1-\delta) S$

#### 3. Tail Risk Constraints (CVaR)
Conditional Value at Risk focuses purely on the extreme left tail (expected loss given we cross the VaR threshold):
- $CVaR_{\alpha} = -\mathbb{E}[r | r < -VaR_{\alpha}]$

#### 4. Systemic Risk Penalty
We penalize assets based on their PageRank/Eigenvector centrality in the financial network:
- $Penalty = \lambda_{sys} w^T S \Sigma S w$

### 1. Advanced Covariance Matrix Construction

#### What We Did
We computed multiple covariance matrices: the standard sample covariance, a Ledoit-Wolf shrunk covariance, and regime-specific matrices (Stress vs. Calm).

#### Why We Did It
The standard sample covariance matrix $S = \frac{1}{T-1} \sum (r_t - \mu)(r_t - \mu)^T$ suffers from severe estimation error when $N$ (assets) is large relative to $T$ (time). Extreme eigenvalues in $S$ cause the optimizer to take massive, highly leveraged long/short positions purely to exploit statistical noise.
Furthermore, correlations break down during crises. Relying on a single static matrix guarantees failure during regime shifts.

#### The Formulas Used
- **Ledoit-Wolf Shrinkage**: $\Sigma_{LW} = \delta F + (1-\delta) S$
  - $\delta$: The optimal shrinkage intensity, calculated analytically to minimize out-of-sample squared error.
- **Regime Blending**: $\Sigma_{Blend} = P_{Stress} \Sigma_{Stress} + P_{Calm} \Sigma_{Calm} + P_{Vol} \Sigma_{Volatile}$
  - Ensures the optimizer respects the elevated risk of current market conditions.

### 2. Forward-Looking Expected Returns Construction

#### What We Did
We discarded historical trailing means entirely. Instead, we constructed a composite expected return vector ($\mu_{final}$) by blending Machine Learning direction probabilities, NLP sentiment $z$-scores, and mathematical mean-reversion metrics.

#### Why We Did It
Historical return is the most dangerous input in portfolio construction. A stock that returned 50% last year does not have a 50% expected return next year; it likely has a negative expected return due to mean reversion. By using predictive models (ML + Sentiment), we shift the optimizer from "looking in the rear-view mirror" to "looking through the windshield."

#### The Formulas Used
The composite vector is a weighted sum of three distinct predictive signals:
1. **ML Component**: $\mu_{ML} = (P_{up} - 0.5) \times 2 \times \sigma_{realized} \times M_{regime}$
   - Centers the probability around 0, scales it by the asset's inherent volatility, and adjusts for the macro regime.
2. **Sentiment Component**: $\mu_{Sent} = Z_{sentiment} \times R_{base}$
   - Translates sentiment anomalies into expected basis point drifts.
3. **Reversion Component**: $\mu_{Rev} = - \frac{Price_{current} - MA_{20}}{MA_{20}} \times \beta_{reversion}$

$\mu_{final} = 0.5\mu_{ML} + 0.3\mu_{Sent} + 0.2\mu_{Rev}$

### 3. Strategy 1: Classical Markowitz (The Benchmark)

#### What We Did
We solved the classical Tangency Portfolio problem: maximizing the Sharpe ratio using historical returns and sample covariance, with the only constraint being full investment ($\sum w = 1$) and no short selling ($w \ge 0$).

#### Why We Did It
We must have a baseline to measure our system's added value. If AxiomAlpha cannot beat classical Markowitz on a risk-adjusted basis, the complexity is unwarranted.

#### The Optimization Formula
We frame the problem as minimizing the negative Sharpe Ratio via Sequential Least Squares Programming (SLSQP):
$$ \min_{w} - \frac{w^T \mu_{historical} - r_f}{\sqrt{w^T \Sigma_{sample} w}} $$
$$ s.t. \sum w_i = 1, \quad w_i \ge 0 $$

### 4. Strategy 2: AxiomAlpha Base Portfolio

#### What We Did
We solved a heavily constrained optimization problem using our robust inputs. We replaced historical returns with $\mu_{final}$ and sample covariance with $\Sigma_{LW}$. We also introduced strict upper bounds on asset weights to enforce diversification.

#### Why We Did It
By capping any single asset at 15%, we artificially bound the optimizer's "greed." Even if the ML model is 99% confident an asset will rise, we restrict exposure to prevent catastrophic ruin if the model is wrong.

#### The Optimization Formulas & Constraints
$$ \max_w \frac{w^T \mu_{final} - r_f}{\sqrt{w^T \Sigma_{LW} w}} $$
Subject to real-world risk management rules:
1. **Full Investment**: $\sum w_i = 1$
2. **Long Only**: $w_i \ge 0$
3. **Asset Concentration Limit**: $w_i \le 0.15$
4. **Tail Risk (CVaR) Approximation**: Implicitly managed via the shrunk covariance matrix, which drastically reduces the probability of simultaneous multi-asset drawdowns.

### 5. Strategy 3: AxiomAlpha Full Portfolio

#### What We Did
We added our proprietary Graph Theory network metrics and NLP overrides directly into the convex optimization objective function.

#### Why We Did It
During systemic crashes (e.g., 2008, 2020), assets do not behave independently—they cascade. Traditional covariance fails to capture structural "contagion." By penalizing assets with high eigenvector centrality (from our Phase 4 network graph), we actively de-weight the nodes most likely to drag the portfolio down in a flash crash.

#### The Complete AxiomAlpha Objective Function Formula
We minimize the negative Sharpe, but we add a penalty term for network risk, and a reward term for sentiment tilt:
$$ \max_w \left( \frac{w^T \mu_{final} - r_f}{\sqrt{w^T \Sigma_{blend} w}} - \lambda_{sys} \cdot w^T S \Sigma_{blend} S w + \gamma_{sent} \cdot w^T Z_{sentiment} \right) $$

Where:
- $S$: Diagonal matrix of systemic risk scores $\in [0,1]$.
- $\lambda_{sys}$: Hyperparameter scaling our fear of contagion (0.3).
- $\gamma_{sent}$: Hyperparameter scaling our tilt toward breaking news (0.1).

### 6. Monte Carlo Tail Risk Simulation

#### What We Did
We generated 1,000 simulated 1-year future paths for our optimized portfolios using Student-t distributed shocks instead of Gaussian (Normal) shocks.

#### Why We Did It
Backtests only show us the *one path* history actually took. To truly understand portfolio robustness, we must simulate thousands of alternative realities. Using Student-t shocks with low degrees of freedom ($\nu=5$) forces the simulation to generate frequent, massive outlier days ("Black Swans"). If AxiomAlpha survives the Student-t simulation, it is structurally sound.

#### The Simulation Formulas
1. Cholesky Decomposition of Covariance: $\Sigma = L L^T$
2. Generate $T$-distributed matrix $Z$: $z \sim t(\nu=5)$
3. Induce correlation: $R = \mu + Z L^T$

### 7. Global Risk Profile Deep Dive

#### What We Did
We evaluated the optimized portfolios against rigorous, professional-grade institutional metrics.

#### The Core Metrics & Formulas
1. **Conditional Value at Risk ($CVaR_{95}$)**:
   - The expected loss *given* that a loss has exceeded the $VaR_{95}$ threshold.
   - Formula: $CVaR_{\alpha} = -\frac{1}{\alpha} \int_{0}^{\alpha} VaR_{\gamma} d\gamma$
2. **Maximum Drawdown (MDD)**:
   - The largest peak-to-trough drop in portfolio value.
   - Formula: $MDD = \max \frac{Peak - Trough}{Peak}$
3. **Sortino Ratio**:
   - Like Sharpe, but only penalizes *downside* volatility.
   - Formula: $Sortino = \frac{\mu_p - r_f}{\sigma_{downside}}$

## Key Mathematical Formulas
$$ \min_{w} - \frac{w^T \mu_{historical} - r_f}{\sqrt{w^T \Sigma_{sample} w}} $$

$$ s.t. \sum w_i = 1, \quad w_i \ge 0 $$

$$ \max_w \left( \frac{w^T \mu_{final} - r_f}{\sqrt{w^T \Sigma_{blend} w}} - \lambda_{sys} \cdot w^T S \Sigma_{blend} S w + \gamma_{sent} \cdot w^T Z_{sentiment} \right) $$

$$ \max_w \frac{w^T \mu_{final} - r_f}{\sqrt{w^T \Sigma_{LW} w}} $$

