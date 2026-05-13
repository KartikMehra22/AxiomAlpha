"""Pure data-processing primitives extracted from NB02.

Notebook code that wants to compute returns, rolling stats, or feature
engineering should import from here rather than re-implement inline. The
notebook keeps orchestration (loading, plotting, saving); the math lives
in this module so it can be unit-tested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "compute_log_returns",
    "compute_simple_returns",
    "rolling_volatility",
    "rolling_momentum",
    "rolling_drawdown",
    "engineer_features",
]


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns from a wide price DataFrame.

    Drops the first row (NaN). Index is preserved.

    Raises ValueError if any non-NaN price is zero or negative — those would
    silently produce -inf / NaN log returns and corrupt every downstream
    feature, so we fail loudly at the boundary instead.
    """
    bad = (prices <= 0) & prices.notna()
    if bool(bad.any().any()):
        bad_cols = bad.any()
        offenders = list(bad_cols[bad_cols].index)
        raise ValueError(
            f"compute_log_returns: {int(bad.sum().sum())} non-positive price(s) "
            f"in columns {offenders}; log returns would be -inf/NaN. Clean the "
            f"price feed before calling this function."
        )
    return np.log(prices / prices.shift(1)).dropna(how="all")


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns (P_t / P_{t-1} - 1) from a wide price DataFrame."""
    return prices.pct_change().dropna(how="all")


def rolling_volatility(
    returns: pd.DataFrame, window: int, min_periods: int | None = None
) -> pd.DataFrame:
    """Rolling sample-std of returns. ddof=1 by convention."""
    return returns.rolling(window, min_periods=min_periods or window).std()


def rolling_momentum(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling sum of log returns ≈ log of cumulative return over the window."""
    return returns.rolling(window).sum()


def rolling_drawdown(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Rolling max-drawdown computed inside a window of `window` days.

    Per ticker: cumulative product, rolling max, percentage off.
    """
    cum = (1.0 + returns).cumprod()
    peak = cum.rolling(window, min_periods=1).max()
    return (cum - peak) / peak


def engineer_features(
    returns: pd.DataFrame,
    return_lookbacks: tuple[int, ...] = (1, 5, 20, 60),
    vol_windows: tuple[int, ...] = (5, 20, 60),
) -> pd.DataFrame:
    """Build the canonical NB02 feature matrix in long format.

    Returns a DataFrame with columns:
        Date, ticker, ret_{n}d for n in return_lookbacks, vol_{n}d for
        n in vol_windows, mom_20d, mom_60d, drawdown.

    `returns` should be the daily log-returns DataFrame (wide format).
    """
    pieces: dict[str, pd.DataFrame] = {}
    for n in return_lookbacks:
        pieces[f"ret_{n}d"] = returns.rolling(n).sum() if n > 1 else returns
    for w in vol_windows:
        pieces[f"vol_{w}d"] = rolling_volatility(returns, w)
    pieces["mom_20d"] = rolling_momentum(returns, 20)
    pieces["mom_60d"] = rolling_momentum(returns, 60)
    pieces["drawdown"] = rolling_drawdown(returns, 60)

    out: pd.DataFrame | None = None
    for name, df in pieces.items():
        long = df.stack().rename(name).reset_index()
        long.columns = ["Date", "ticker", name]
        if out is None:
            out = long
        else:
            out = out.merge(long, on=["Date", "ticker"], how="outer")
    return (
        out.sort_values(["Date", "ticker"]).reset_index(drop=True)
        if out is not None else pd.DataFrame()
    )
