"""Pure NLP helpers extracted from NB08/NB10.

Sentiment aggregation and information-coefficient computation. FinBERT
scoring + sentence-transformer embedding live in the notebooks (heavy
model loads — not appropriate for an importable module).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "aggregate_sentiment_per_day",
    "information_coefficient",
]


def aggregate_sentiment_per_day(
    news: pd.DataFrame,
    *,
    score_col: str = "sentiment_score",
    ticker_col: str = "ticker",
    date_col: str = "Date",
) -> pd.DataFrame:
    """Aggregate per-headline sentiment into per-(Date, ticker) features.

    Returns a long DataFrame with columns: `Date`, `ticker`,
    `sentiment_mean`, `sentiment_std`, `news_volume`.
    """
    if news.empty:
        return pd.DataFrame(columns=[date_col, ticker_col,
                                     "sentiment_mean", "sentiment_std",
                                     "news_volume"])
    grouped = (
        news.groupby([date_col, ticker_col])[score_col]
            .agg(sentiment_mean="mean", sentiment_std="std",
                 news_volume="size")
            .reset_index()
    )
    grouped["sentiment_std"] = grouped["sentiment_std"].fillna(0.0)
    return grouped


def information_coefficient(
    sentiment_features: pd.DataFrame,
    log_returns: pd.DataFrame,
    *,
    lag: int = 1,
    sentiment_col: str = "sentiment_mean",
    ticker_col: str = "ticker",
    date_col: str = "Date",
) -> float:
    """Average per-ticker Spearman IC between sentiment_t and return_{t+lag}.

    `sentiment_features` is a long DataFrame; `log_returns` is wide
    (Date index, ticker columns). Tickers missing from either are skipped.
    """
    if sentiment_features.empty or log_returns.empty:
        return float("nan")

    sf = sentiment_features.copy()
    sf[date_col] = pd.to_datetime(sf[date_col])

    ics: list[float] = []
    for ticker, group in sf.groupby(ticker_col):
        if ticker not in log_returns.columns:
            continue
        s = group.set_index(date_col)[sentiment_col]
        r = log_returns[ticker].shift(-lag)
        joined = pd.concat([s, r], axis=1, join="inner").dropna()
        if len(joined) < 10:
            continue
        ic = joined.iloc[:, 0].corr(joined.iloc[:, 1], method="spearman")
        if pd.notna(ic):
            ics.append(float(ic))
    if not ics:
        return float("nan")
    return float(np.mean(ics))
