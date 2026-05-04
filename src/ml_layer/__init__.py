"""Pure ML helpers extracted from NB06/NB07.

Target construction and walk-forward evaluation utilities. Models
themselves stay in the notebooks (where their hyperparameters are
hand-tuned and visualized).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "make_direction_target",
    "make_volatility_target",
    "walk_forward_split",
]


def make_direction_target(
    long_features: pd.DataFrame,
    *,
    return_col: str = "ret_1d",
    ticker_col: str = "ticker",
    date_col: str = "Date",
) -> pd.Series:
    """Binary next-day direction target: 1 if next-day return > 0 else 0.

    Caller is responsible for sort-by-(ticker, Date) before calling.
    Returns `pd.NA` (not 0) at the last day per ticker, so callers can
    distinguish "down day" from "no-data day".
    """
    df = long_features.sort_values([ticker_col, date_col])
    next_ret = df.groupby(ticker_col)[return_col].shift(-1)
    target = (next_ret > 0).astype("Int64")
    target[next_ret.isna()] = pd.NA
    return target.rename("target_direction")


def make_volatility_target(
    long_features: pd.DataFrame,
    *,
    return_col: str = "ret_1d",
    horizon: int = 5,
    ticker_col: str = "ticker",
    date_col: str = "Date",
) -> pd.Series:
    """Realized volatility over the FORWARD `horizon` days (per ticker).

    At position t this is std(returns_{t+1}, ..., returns_{t+horizon}).
    The last `horizon` rows per ticker are NaN by construction.
    """
    df = long_features.sort_values([ticker_col, date_col]).copy()
    # rolling(horizon).std() at position t = std(s_{t-horizon+1..t}).
    # Shifting by -horizon means at position t we read the value that
    # was originally at position t+horizon, i.e. std(s_{t+1..t+horizon}).
    fwd = df.groupby(ticker_col)[return_col].transform(
        lambda s: s.rolling(horizon).std().shift(-horizon)
    )
    return fwd.rename(f"target_vol_{horizon}d")


def walk_forward_split(
    n_samples: int, n_splits: int = 5, min_train: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Walk-forward index splits — like sklearn TimeSeriesSplit but
    deterministic and explicit.

    Yields (train_idx, test_idx) tuples where train_idx ends strictly
    before test_idx starts.
    """
    if min_train is None:
        min_train = n_samples // (n_splits + 1)
    test_size = (n_samples - min_train) // n_splits
    if test_size < 1:
        raise ValueError(
            f"n_samples={n_samples} is too small for n_splits={n_splits}"
        )
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    train_end = min_train
    for _ in range(n_splits):
        train_idx = np.arange(0, train_end)
        test_idx  = np.arange(train_end, min(train_end + test_size, n_samples))
        if len(test_idx) == 0:
            break
        splits.append((train_idx, test_idx))
        train_end += test_size
    return splits
