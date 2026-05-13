"""Reference implementations + regression tests for the finance formulas
used across NB03 / NB11 / NB12.

These are canonical (closed-form-verified) and intentionally redundant with
notebook code — if a notebook drifts away from the right formula, these
tests fail.

Includes a regression test for the `rf*252` bug that shipped in commit
e8274d1 (Sharpe was treating an annualized RFR as if it were daily).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from configs.settings import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


TPY = TRADING_DAYS_PER_YEAR  # alias


# --- Reference implementations ---------------------------------------------

_FLOAT_NOISE = 1e-12  # tolerance for "std is zero modulo float arithmetic"


def annualized_sharpe(daily_returns: np.ndarray, rf_annual: float, periods: int = TPY) -> float:
    """Sharpe = (annualized excess return) / (annualized vol).

    Daily excess return is `daily_returns - rf_annual/periods`. Multiply by
    `periods` to annualize the mean; multiply std by sqrt(periods).
    """
    daily_returns = np.asarray(daily_returns, dtype=float)
    excess = daily_returns - (rf_annual / periods)
    std = float(excess.std(ddof=1))
    if std < _FLOAT_NOISE:
        return 0.0
    return float((excess.mean() * periods) / (std * math.sqrt(periods)))


def annualized_sortino(daily_returns: np.ndarray, rf_annual: float, periods: int = TPY) -> float:
    """Sortino = annualized excess return / annualized downside deviation.

    Downside deviation uses only negative excess returns vs. zero.
    """
    daily_returns = np.asarray(daily_returns, dtype=float)
    excess = daily_returns - (rf_annual / periods)
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float('inf')
    dd = math.sqrt(float(np.mean(downside ** 2)))  # not ddof=1 — Sortino convention
    if dd < _FLOAT_NOISE:
        return 0.0
    return float((excess.mean() * periods) / (dd * math.sqrt(periods)))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Max drawdown as a non-positive fraction. -0.20 means a 20% peak-to-trough."""
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    return float(drawdowns.min())


def annualized_calmar(daily_returns: np.ndarray, equity_curve: np.ndarray,
                     periods: int = TPY) -> float:
    """Calmar = annualized return / abs(max drawdown). RFR-free."""
    ann_return = daily_returns.mean() * periods
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return float('inf')
    return ann_return / mdd


# --- Sharpe -----------------------------------------------------------------

class TestSharpe:

    def test_zero_returns_gives_zero_sharpe(self):
        daily = np.zeros(252)
        # mean=-rf/252, std=0 → handled to return 0
        assert annualized_sharpe(daily, rf_annual=0.04) == 0.0

    def test_constant_positive_returns_no_vol(self):
        # All returns = 1bp/day; std=0 → returns 0 by convention
        daily = np.full(252, 0.0001)
        assert annualized_sharpe(daily, rf_annual=0.0) == 0.0

    def test_matches_canonical_formula_on_sample(self):
        """The function should match the canonical formula computed on the
        same sample (sample-moments equivalence). Pinned to seed."""
        rng = np.random.default_rng(42)
        daily = rng.normal(0.0008, 0.012, 5000)
        rf = 0.04
        sharpe = annualized_sharpe(daily, rf_annual=rf)
        expected = (
            (daily.mean() * TPY - rf) / (daily.std(ddof=1) * math.sqrt(TPY))
        )
        assert sharpe == pytest.approx(expected, rel=1e-9)

    def test_in_realistic_range_for_realistic_inputs(self):
        """Realistic inputs (μ_daily ~ 0.0008, σ_daily ~ 0.012, rf=4%)
        should give a Sharpe somewhere in [0, 2]. Pinned to verify we're
        in the right ballpark, not numerically blowing up."""
        rng = np.random.default_rng(42)
        daily = rng.normal(0.0008, 0.012, 5000)
        sharpe = annualized_sharpe(daily, rf_annual=0.04)
        assert 0.0 < sharpe < 2.0, f"Sharpe out of plausible range: {sharpe}"

    def test_regression_rf_double_annualization_bug(self):
        """Regression for commit e8274d1.

        The bug was: `Sharpe = (mean_daily_excess) / (std_daily) * sqrt(252)`
        where mean_daily_excess used `rf_annual * 252` instead of
        `rf_annual / 252`. With rf_annual=0.04 and a noisy zero-mean daily
        series, the BUGGY Sharpe is enormously negative (~-200s); the
        CORRECT Sharpe is in a sane range. This test pins that gap.
        """
        rng = np.random.default_rng(0)
        daily = rng.normal(0.0, 0.01, 1000)  # zero-mean, 1% daily vol
        rf = 0.04

        correct = annualized_sharpe(daily, rf_annual=rf)

        # The buggy formula: subtract rf*252 from each daily return
        bad_excess = daily - (rf * TPY)
        buggy = (
            (bad_excess.mean() * TPY) / (bad_excess.std(ddof=1) * math.sqrt(TPY))
        )

        assert abs(correct) < 5.0, f"Correct Sharpe should be small-to-moderate, got {correct}"
        assert buggy < -50,        f"Buggy Sharpe should blow up negative, got {buggy}"
        # Gap must be enormous so a recurrence is unmistakable:
        assert (correct - buggy) > 50


# --- Sortino & Calmar -------------------------------------------------------

class TestSortino:

    def test_no_downside_gives_inf(self):
        daily = np.full(252, 0.001)  # all positive vs rf=0
        assert annualized_sortino(daily, rf_annual=0.0) == float('inf')

    def test_symmetric_returns_sortino_above_sharpe(self):
        """For symmetric returns, |downside dev| < total std, so Sortino > Sharpe."""
        rng = np.random.default_rng(1)
        daily = rng.normal(0.0008, 0.012, 2000)
        sharpe = annualized_sharpe(daily, rf_annual=0.04)
        sortino = annualized_sortino(daily, rf_annual=0.04)
        assert sortino > sharpe


class TestCalmar:

    def test_calmar_known_curve(self):
        # Build a synthetic equity curve with a known 20% drawdown
        prices = np.concatenate([
            np.linspace(100, 120, 100),  # +20%
            np.linspace(120, 96, 50),    # -20% from peak
            np.linspace(96, 130, 100),   # recover and beyond
        ])
        daily_returns = np.diff(np.log(prices))
        calmar = annualized_calmar(daily_returns, prices)
        # MDD is exactly -0.20 by construction
        assert abs(max_drawdown(prices) - (-0.20)) < 1e-9
        assert calmar > 0  # the curve ends higher


# --- Max drawdown ----------------------------------------------------------

class TestMaxDrawdown:

    def test_monotone_increasing_no_drawdown(self):
        prices = np.linspace(100, 200, 252)
        assert max_drawdown(prices) == 0.0

    def test_known_50pct_drawdown(self):
        prices = np.array([100., 200., 100.])  # peak at 200, trough at 100
        assert max_drawdown(prices) == pytest.approx(-0.50)


# --- Log-return identity ----------------------------------------------------

class TestLogReturns:

    def test_cum_log_return_equals_log_total_return(self):
        """sum(log_returns) == log(P_T / P_0). Pin this down — NB02 should hold it."""
        rng = np.random.default_rng(3)
        prices = 100 + rng.standard_normal(252).cumsum()
        prices = np.maximum(prices, 1.0)  # keep positive
        log_returns = np.diff(np.log(prices))
        cum = log_returns.sum()
        total = math.log(prices[-1] / prices[0])
        assert cum == pytest.approx(total, rel=1e-12)

    def test_simple_return_to_log_return_conversion(self):
        simple = 0.05
        log_eq = math.log(1 + simple)
        # Round-trip
        back = math.exp(log_eq) - 1
        assert back == pytest.approx(simple, rel=1e-12)


# --- Graph (MST) -----------------------------------------------------------

class TestMST:

    def test_minimum_spanning_tree_has_n_minus_1_edges(self):
        """A property NB04 produces; trivially worth pinning down."""
        import networkx as nx
        rng = np.random.default_rng(5)
        n = 30
        # Random complete graph weighted by abs(corr-like) similarities
        g = nx.complete_graph(n)
        for u, v in g.edges:
            g[u][v]['weight'] = float(rng.uniform(0.1, 1.0))
        mst = nx.minimum_spanning_tree(g, weight='weight')
        assert mst.number_of_edges() == n - 1


# --- Settings sanity -------------------------------------------------------

def test_risk_free_rate_is_annualized_not_daily():
    """If anyone ever sets RISK_FREE_RATE to ~0.0001 (a daily rate), this
    catches it. Annualized RFRs live in roughly 0% .. 20%."""
    assert 0.0 <= RISK_FREE_RATE <= 0.20


# --- Parity with src.quant_core --------------------------------------------
#
# The reference implementations above are the math-spec. The package in
# src/quant_core/__init__.py is what notebooks actually import. These
# parity tests pin the package to the spec — if a future refactor changes
# either one without updating the other, this fails.

class TestSrcQuantCoreParity:

    def test_sharpe_matches_reference(self):
        from src.quant_core import annualized_sharpe as pkg_sharpe
        rng = np.random.default_rng(7)
        daily = rng.normal(0.0008, 0.012, 5000)
        assert pkg_sharpe(daily, rf_annual=0.04) == pytest.approx(
            annualized_sharpe(daily, rf_annual=0.04), rel=1e-12
        )

    def test_sortino_matches_reference(self):
        from src.quant_core import annualized_sortino as pkg_sortino
        rng = np.random.default_rng(8)
        daily = rng.normal(0.0008, 0.012, 5000)
        assert pkg_sortino(daily, rf_annual=0.04) == pytest.approx(
            annualized_sortino(daily, rf_annual=0.04), rel=1e-12
        )

    def test_max_drawdown_matches_reference(self):
        from src.quant_core import max_drawdown as pkg_mdd
        prices = np.array([100., 200., 100., 150.])
        assert pkg_mdd(prices) == pytest.approx(max_drawdown(prices), rel=1e-12)

    def test_portfolio_metrics_keys_and_sane_values(self):
        from src.quant_core import portfolio_metrics
        rng = np.random.default_rng(11)
        daily = rng.normal(0.0008, 0.012, 5000)
        m = portfolio_metrics(daily, rf_annual=0.04)
        for k in ('ann_return', 'ann_vol', 'sharpe', 'sortino', 'calmar',
                  'max_dd', 'var_95', 'cvar_95', 'n_days'):
            assert k in m, f"missing key {k}"
        assert m['n_days'] == 5000
        assert -2.0 < m['sharpe'] < 5.0  # plausibility

    def test_portfolio_metrics_sortino_does_not_repeat_rf_252_bug(self):
        """If anyone re-introduces `rf*periods` in Sortino, this fails."""
        from src.quant_core import portfolio_metrics
        rng = np.random.default_rng(0)
        daily = rng.normal(0.0, 0.01, 1000)
        m = portfolio_metrics(daily, rf_annual=0.04)
        # Correct Sortino should be in a sensible range, not the -52.7 we saw
        # in the original NB12 bug.
        assert m['sortino'] > -10.0, f"Sortino blew up: {m['sortino']}"

    def test_compute_portfolio_returns_preserves_cash_overlay(self):
        """Regression for the strategy-tie bug: previously the helper did
        `w / w.sum()` which wiped out the regime overlay. With cash-aware
        normalization, two strategies with different total exposure should
        produce different returns."""
        from src.quant_core import compute_portfolio_returns
        idx = pd.date_range('2024-01-01', periods=20, freq='B')
        rets = pd.DataFrame({
            'A': np.full(20, 0.01),  # constant +1% daily
            'B': np.full(20, 0.01),
        }, index=idx)
        rebal = idx[0]
        w_full   = pd.Series([0.5, 0.5], index=['A', 'B'])           # sum=1.0
        w_half   = pd.Series([0.25, 0.25], index=['A', 'B'])         # sum=0.5 → 50% cash
        r_full = compute_portfolio_returns({rebal: w_full}, rets, rf_annual=0.0)
        r_half = compute_portfolio_returns({rebal: w_half}, rets, rf_annual=0.0)
        # full is invested at 100%, half is at 50% (other 50% earns 0 RFR)
        assert r_full.mean() == pytest.approx(0.01, rel=1e-9)
        assert r_half.mean() == pytest.approx(0.005, rel=1e-9)
        assert r_full.mean() != r_half.mean()  # the bug fix proof


# --- Cost model: linear + square-root market impact ------------------------

class TestSlippageModel:
    """Pins compute_portfolio_returns's two-term cost model:
        cost = transaction_cost * sum_i |Δw_i|              (linear)
             + slippage_coef    * sum_i |Δw_i|^1.5          (sqrt impact)
    """

    @staticmethod
    def _two_rebalance_setup():
        from src.quant_core import compute_portfolio_returns  # noqa: F401
        idx = pd.date_range('2024-01-01', periods=10, freq='B')
        rets = pd.DataFrame({
            'A': np.zeros(10),
            'B': np.zeros(10),
        }, index=idx)
        # Two rebalances → one turnover step.
        # Δw_A = |0.6 - 0.5| = 0.1; Δw_B = |0.4 - 0.5| = 0.1 → sum=0.2
        weights = {
            idx[0]: pd.Series([0.5, 0.5], index=['A', 'B']),
            idx[5]: pd.Series([0.6, 0.4], index=['A', 'B']),
        }
        return rets, weights

    def test_zero_costs_gives_zero_drag(self):
        from src.quant_core import compute_portfolio_returns
        rets, weights = self._two_rebalance_setup()
        r = compute_portfolio_returns(weights, rets,
                                     transaction_cost=0.0,
                                     slippage_coef=0.0,
                                     rf_annual=0.0)
        # Returns are all zero, no costs → every entry zero.
        assert (r == 0.0).all()

    def test_linear_cost_subtracts_only_turnover_times_bps(self):
        from src.quant_core import compute_portfolio_returns
        rets, weights = self._two_rebalance_setup()
        TC = 0.001  # 10 bps
        r = compute_portfolio_returns(weights, rets,
                                     transaction_cost=TC,
                                     slippage_coef=0.0,
                                     rf_annual=0.0)
        # Total drag across the run is exactly turnover * TC = 0.2 * 0.001 = 2e-4,
        # applied on the first day of the second period.
        assert r.sum() == pytest.approx(-0.2 * TC, rel=1e-9)

    def test_slippage_term_is_convex_in_per_asset_size(self):
        """slippage_coef * sum_i |Δw_i|^1.5. For Δw_A=Δw_B=0.1, the contribution
        is 2 * 0.1**1.5 = 2 * 0.0316... = 0.06324...
        """
        from src.quant_core import compute_portfolio_returns
        rets, weights = self._two_rebalance_setup()
        SLIP = 0.01
        r = compute_portfolio_returns(weights, rets,
                                     transaction_cost=0.0,
                                     slippage_coef=SLIP,
                                     rf_annual=0.0)
        expected = -SLIP * (0.1 ** 1.5 + 0.1 ** 1.5)
        assert r.sum() == pytest.approx(expected, rel=1e-9)

    def test_one_big_trade_beats_many_small_trades_under_slippage(self):
        """Convexity is the whole point of the sqrt model: a single 0.2 step
        costs more in impact than two 0.1 steps. (Linear costs would be equal.)
        """
        from src.quant_core import compute_portfolio_returns
        idx = pd.date_range('2024-01-01', periods=20, freq='B')
        rets = pd.DataFrame({'A': np.zeros(20)}, index=idx)
        SLIP = 0.01

        # Path 1: one big 0.2 rebalance.
        weights_big = {
            idx[0]:  pd.Series([1.0], index=['A']),
            idx[10]: pd.Series([0.8], index=['A']),
        }
        # Path 2: two 0.1 rebalances summing to the same 0.2 total turnover.
        weights_split = {
            idx[0]:  pd.Series([1.0], index=['A']),
            idx[5]:  pd.Series([0.9], index=['A']),
            idx[10]: pd.Series([0.8], index=['A']),
        }
        r_big   = compute_portfolio_returns(weights_big,   rets,
                                           transaction_cost=0.0,
                                           slippage_coef=SLIP, rf_annual=0.0)
        r_split = compute_portfolio_returns(weights_split, rets,
                                           transaction_cost=0.0,
                                           slippage_coef=SLIP, rf_annual=0.0)
        # 0.2^1.5 ≈ 0.0894;  2 * 0.1^1.5 ≈ 0.0632 — concentrated is more expensive.
        assert r_big.sum() < r_split.sum(), \
            f"sqrt impact should penalize concentration: big={r_big.sum()}, split={r_split.sum()}"


# --- Shorts toggle (OPT_LONG_ONLY=False codepath) --------------------------

class TestShortsEnabled:
    """Smoke-tests that compute_portfolio_returns handles a long/short weight
    series (some entries negative, sum=1.0) end-to-end without crashing,
    and that the turnover calc treats short legs symmetrically with longs.
    """

    def test_long_short_dot_product_matches_manual(self):
        from src.quant_core import compute_portfolio_returns
        idx = pd.date_range('2024-01-01', periods=5, freq='B')
        rets = pd.DataFrame({
            'A': np.full(5, 0.02),   # +2% daily
            'B': np.full(5, -0.01),  # -1% daily
        }, index=idx)
        # Long A, short B — sum=1.0 so no cash residual.
        w = pd.Series([1.2, -0.2], index=['A', 'B'])
        r = compute_portfolio_returns({idx[0]: w}, rets,
                                     transaction_cost=0.0,
                                     slippage_coef=0.0,
                                     rf_annual=0.0)
        # daily = 1.2 * 0.02 + (-0.2) * (-0.01) = 0.024 + 0.002 = 0.026
        assert r.mean() == pytest.approx(0.026, rel=1e-9)

    def test_short_leg_counts_toward_turnover(self):
        """Going from (1.2, -0.2) to (0.8, 0.2) is |Δ|=0.4+0.4=0.8 turnover,
        even though both sides cross zero."""
        from src.quant_core import compute_portfolio_returns
        idx = pd.date_range('2024-01-01', periods=10, freq='B')
        rets = pd.DataFrame({
            'A': np.zeros(10),
            'B': np.zeros(10),
        }, index=idx)
        TC = 0.001
        weights = {
            idx[0]: pd.Series([1.2, -0.2], index=['A', 'B']),
            idx[5]: pd.Series([0.8,  0.2], index=['A', 'B']),
        }
        r = compute_portfolio_returns(weights, rets,
                                     transaction_cost=TC,
                                     slippage_coef=0.0,
                                     rf_annual=0.0)
        assert r.sum() == pytest.approx(-0.8 * TC, rel=1e-9)
