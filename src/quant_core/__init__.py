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
    dd = math.sqrt(float(np.mean(np.minimum(excess, 0.0) ** 2)))
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
        equity_curve = np.r_[1.0, np.cumprod(1.0 + r)]
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
    cum = np.r_[1.0, np.cumprod(1.0 + r)]
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
    slippage_coef: float = 0.0,
    rf_annual: float = RISK_FREE_RATE,
    periods: int = TRADING_DAYS_PER_YEAR,
    return_details: bool = False,
) -> pd.Series:
    """Step a portfolio through `returns_df` using the given rebalance plan.

    `weights_dict` maps rebalance_date → ticker-indexed weight Series. If a
    weight series sums to <1, the residual is treated as cash earning the
    daily risk-free rate (so a regime-overlay strategy that holds cash is
    measured against fully-invested benchmarks fairly). If the series sums
    to >1 (numerical), it is renormalized to 1.

    Cost model per rebalance:
      cost = transaction_cost * sum_i |Δw_i|
           + slippage_coef    * sum_i |Δw_i|^1.5
    The first term is the linear bid/ask + commissions; the second is a
    square-root market-impact term (Almgren-style), which becomes material
    on concentrated rebalances. Both are subtracted from the first day of
    the new holding period.

    Returns a daily-return Series.
    """
    if returns_df.index.has_duplicates or not returns_df.index.is_monotonic_increasing:
        raise ValueError("returns_df index must be sorted and unique")
    daily_rf = rf_annual / periods
    dates = sorted(pd.Timestamp(d) for d in weights_dict)
    if not dates:
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], name=returns_df.index.name))
        return (empty, pd.DataFrame(index=empty.index)) if return_details else empty
    holdings = None
    out = []
    details = []
    for i, rebal in enumerate(dates):
        target = weights_dict[rebal].astype(float)
        missing = target.index.difference(returns_df.columns)
        if len(missing):
            raise KeyError(f"weight tickers not in returns_df: {list(missing)}")
        target = target.reindex(returns_df.columns, fill_value=0.0)
        if not np.isfinite(target).all():
            raise ValueError("weights must be finite")
        if target.sum() > 1 + 1e-8:
            raise ValueError("net asset weights cannot exceed 1")
        if holdings is None:
            cost = 0.0  # initial allocation is provided as starting capital
        else:
            dw = (target - holdings).abs()
            cost = transaction_cost * dw.sum() + slippage_coef * (dw ** 1.5).sum()
        if not np.isfinite(cost) or cost < 0 or cost >= 1:
            raise ValueError("transaction costs must be finite and less than portfolio value")
        cash = 1.0 - target.sum()
        holdings = target.copy()
        end = dates[i + 1] if i + 1 < len(dates) else returns_df.index[-1]
        period = returns_df.loc[(returns_df.index > rebal) & (returns_df.index <= end)]
        for j, (date, asset_returns) in enumerate(period.iterrows()):
            if asset_returns.isna().any():
                raise ValueError(f"missing asset returns on {date}")
            gross = float(holdings @ asset_returns + cash * daily_rf)
            effective_cost = float(cost) if j == 0 else 0.0
            net = (1.0 - effective_cost) * (1.0 + gross) - 1.0
            out.append((date, net))
            contributions = holdings * asset_returns
            contributions = contributions.copy()
            contributions["Cash"] = cash * daily_rf
            contributions["Trading costs"] = -effective_cost * (1.0 + gross)
            details.append(contributions.rename(date))
            # Deduct costs proportionally from wealth; relative holdings drift
            # according to asset performance, not by resetting each day.
            growth = 1.0 + gross
            if growth <= 0:
                raise ValueError(f"portfolio wealth is non-positive on {date}")
            holdings = holdings * (1.0 + asset_returns) / growth
            cash = cash * (1.0 + daily_rf) / growth
    if not out:
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], name=returns_df.index.name))
        return (empty, pd.DataFrame(index=empty.index)) if return_details else empty
    result = pd.Series(dict(out), dtype=float).sort_index()
    result.index.name = returns_df.index.name
    if return_details:
        detail_frame = pd.DataFrame(details)
        detail_frame.index.name = returns_df.index.name
        return result, detail_frame
    return result
