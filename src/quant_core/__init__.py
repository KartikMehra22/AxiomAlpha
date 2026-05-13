"""Pure quant-core formulas extracted from NB03/NB11/NB12.

These are the canonical reference implementations; notebooks that need
Sharpe / Sortino / drawdown should import from here rather than re-defining
inline (which is how the `rf*252` Sharpe bug got into the repo originally).

`tests/test_finance_formulas.py` regression-tests every function below,
including the historical `rf*252` bug pattern.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from configs.settings import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR

__all__ = [
    "RISK_FREE_RATE",
    "TRADING_DAYS_PER_YEAR",
    "annualized_sharpe",
    "annualized_sortino",
    "annualized_calmar",
    "max_drawdown",
    "portfolio_metrics",
    "compute_portfolio_returns",
]


_FLOAT_NOISE = 1e-12  # treat std smaller than this as exactly zero


def _as_array(returns: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(returns), dtype=float)
    return arr[~np.isnan(arr)]


def annualized_sharpe(
    daily_returns: Iterable[float],
    rf_annual: float = RISK_FREE_RATE,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Sharpe = (annualized excess return) / (annualized vol).

    `rf_annual` is the ANNUALIZED risk-free rate. Daily excess subtracts
    `rf_annual / periods`. This function intentionally does NOT support a
    `rf*periods` shape — that historical bug is what
    `tests/test_finance_formulas.py` regression-tests against.
    """
    r = _as_array(daily_returns)
    if r.size == 0:
        return 0.0
    excess = r - (rf_annual / periods)
    std = float(excess.std(ddof=1))
    if std < _FLOAT_NOISE:
        return 0.0
    return float((excess.mean() * periods) / (std * math.sqrt(periods)))


def annualized_sortino(
    daily_returns: Iterable[float],
    rf_annual: float = RISK_FREE_RATE,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Sortino = annualized excess return / annualized downside deviation.

    Downside deviation: sqrt(mean(min(0, excess)**2)) — *not* ddof=1
    by Sortino convention.
    """
    r = _as_array(daily_returns)
    if r.size == 0:
        return 0.0
    excess = r - (rf_annual / periods)
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    dd = math.sqrt(float(np.mean(downside ** 2)))
    if dd < _FLOAT_NOISE:
        return 0.0
    return float((excess.mean() * periods) / (dd * math.sqrt(periods)))


def max_drawdown(equity_curve: Iterable[float]) -> float:
    """Max drawdown as a non-positive fraction. -0.20 = a 20% peak-to-trough.

    Returns NaN if the curve is empty or the running maximum is ever
    non-positive (which would produce inf/-inf in the percentage drop).
    Callers seeded from `cumprod(1 + r)` always start at 1.0, so this guard
    is defensive against malformed input rather than a hot path.
    """
    eq = np.asarray(list(equity_curve), dtype=float)
    if eq.size == 0:
        return float("nan")
    running_max = np.maximum.accumulate(eq)
    if not np.all(running_max > 0):
        return float("nan")
    return float(((eq - running_max) / running_max).min())


def annualized_calmar(
    daily_returns: Iterable[float],
    equity_curve: Iterable[float] | None = None,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calmar = annualized return / |max drawdown|. RFR-free by convention.

    If `equity_curve` is None, compounds it from `daily_returns`.
    """
    r = _as_array(daily_returns)
    if r.size == 0:
        return 0.0
    if equity_curve is None:
        equity_curve = np.cumprod(1.0 + r)
    mdd = abs(max_drawdown(equity_curve))
    if mdd < _FLOAT_NOISE:
        return float("inf")
    return float(r.mean() * periods / mdd)


def portfolio_metrics(
    daily_returns: Iterable[float],
    rf_annual: float = RISK_FREE_RATE,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """Canonical performance scorecard from a daily-return series.

    Returns a dict with snake_case keys: `ann_return`, `ann_vol`, `sharpe`,
    `sortino`, `calmar`, `max_dd`, `var_95`, `cvar_95`, `n_days`. Values
    are floats rounded to 4dp (or NaN where undefined).
    """
    r = _as_array(daily_returns)
    if r.size == 0:
        return {}
    ann_return = float(r.mean() * periods)
    ann_vol    = float(r.std(ddof=1) * math.sqrt(periods))
    sharpe     = annualized_sharpe(r, rf_annual=rf_annual, periods=periods)
    sortino    = annualized_sortino(r, rf_annual=rf_annual, periods=periods)
    cum = np.cumprod(1.0 + r)
    mdd = max_drawdown(cum)
    if mdd is not None and abs(mdd) > _FLOAT_NOISE:
        calmar = ann_return / abs(mdd)
    else:
        calmar = float("inf")
    var_95 = float(-np.percentile(r, 5))
    tail = r[r < -var_95]
    cvar_95 = float(-tail.mean()) if len(tail) else 0.0
    return {
        "ann_return": round(ann_return, 4),
        "ann_vol":    round(ann_vol, 4),
        "sharpe":     round(sharpe, 4),
        "sortino":    round(sortino, 4),
        "calmar":     round(calmar, 4),
        "max_dd":     round(float(mdd), 4),
        "var_95":     round(var_95, 4),
        "cvar_95":    round(cvar_95, 4),
        "n_days":     int(r.size),
    }


def compute_portfolio_returns(
    weights_dict: Mapping[pd.Timestamp, pd.Series],
    returns_df: pd.DataFrame,
    transaction_cost: float = 0.0,
    rf_annual: float = RISK_FREE_RATE,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Step a portfolio through `returns_df` using the given rebalance plan.

    `weights_dict` maps rebalance_date → ticker-indexed weight Series. If a
    weight series sums to <1, the residual is treated as cash earning the
    daily risk-free rate (so a regime-overlay strategy that holds cash is
    measured against fully-invested benchmarks fairly). If the series sums
    to >1 (numerical), it is renormalized to 1.

    Returns a daily-return Series.
    """
    daily_rf = rf_annual / periods
    dates = sorted(weights_dict.keys())
    out: list[pd.Series] = []
    prev_w: pd.Series | None = None
    for i, rebal in enumerate(dates):
        w = weights_dict[rebal]
        w_sum = float(w.sum())
        if w_sum > 1.0 + 1e-9:
            w = w / w_sum
            cash_pct = 0.0
        else:
            cash_pct = max(0.0, 1.0 - w_sum)
        next_rebal = dates[i + 1] if i + 1 < len(dates) else returns_df.index[-1]

        missing = [t for t in w.index if t not in returns_df.columns]
        if missing:
            raise KeyError(
                f"compute_portfolio_returns: weight tickers not in returns_df "
                f"on rebal date {rebal.date() if hasattr(rebal, 'date') else rebal}: "
                f"{missing}. Align the weight Series to the returns universe "
                f"(use .reindex(returns_df.columns).fillna(0)) before calling."
            )

        if prev_w is not None:
            common = w.index.intersection(prev_w.index)
            turnover = float(np.abs(
                w.reindex(common).fillna(0) - prev_w.reindex(common).fillna(0)
            ).sum())
            cost = turnover * transaction_cost
        else:
            cost = 0.0

        period = returns_df.loc[rebal:next_rebal, w.index].iloc[1:]
        if len(period) > 0:
            daily = period @ w
            if cash_pct > 0:
                daily = daily + cash_pct * daily_rf
            daily.iloc[0] -= cost
            out.append(daily)
        prev_w = w.copy()
    if out:
        return pd.concat(out).sort_index()
    return pd.Series(dtype=float)
