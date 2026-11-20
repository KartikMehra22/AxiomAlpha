"""Point-in-time portfolio construction shared by the optimizer and backtest.

Signals are experimental out-of-sample probabilities, not expected returns.
Synthetic sentiment is intentionally excluded. No claim of trading performance
is implied by this research implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.graph_engine import build_mst, compute_centralities, correlation_matrix
from configs.settings import OPT_MAX_WEIGHT, OPT_LONG_ONLY, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


def trailing_signal(signals: pd.DataFrame, as_of, trading_index, tickers, max_age=10):
    """Return latest dated OOS signal for each ticker, never from the future.

    A missing or stale signal is an error; the caller decides whether to wait
    for warm-up or stop. Age is counted in observed trading dates.
    """
    frame = signals.copy()
    if 'training_end' not in frame.columns:
        raise ValueError('signals lack training-end provenance; regenerate notebook 07')
    frame['Date'] = pd.to_datetime(frame['Date'])
    frame['training_end'] = pd.to_datetime(frame['training_end'])
    as_of = pd.Timestamp(as_of)
    frame = frame.loc[(frame.Date <= as_of) & frame.ticker.isin(tickers)]
    frame = frame.sort_values('Date').drop_duplicates('ticker', keep='last').set_index('ticker')
    frame = frame.reindex(tickers)
    if frame['alpha_signal'].isna().any() or frame['Date'].isna().any():
        raise ValueError('missing out-of-sample signal for at least one ticker')
    if frame['training_end'].isna().any() or (frame['training_end'] >= frame['Date']).any():
        raise ValueError('invalid out-of-sample training provenance')
    calendar = pd.DatetimeIndex(trading_index).sort_values()
    signal_positions = calendar.get_indexer(pd.DatetimeIndex(frame['Date']))
    train_positions = calendar.get_indexer(pd.DatetimeIndex(frame['training_end']))
    if (signal_positions < 0).any() or (train_positions < 0).any() or ((signal_positions - train_positions) < 6).any():
        raise ValueError('invalid out-of-sample training provenance: five-day gap required')
    end = calendar.searchsorted(as_of, side='right')
    ages = np.array([end - calendar.searchsorted(d, side='right') for d in frame['Date']])
    if (ages > max_age).any():
        raise ValueError('out-of-sample signals are stale')
    prob = frame['alpha_signal'].astype(float)
    if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError('invalid direction probabilities')
    return prob, frame['Date']


def construct_portfolio(training_returns: pd.DataFrame, probabilities: pd.Series, *,
                        max_weight=OPT_MAX_WEIGHT, long_only=OPT_LONG_ONLY,
                        signal_scale=0.001, network_penalty=0.3):
    """Construct weights from past returns and dated probability signals.

    Mean estimates are daily historical log returns plus an experimental
    centered-probability tilt in daily-return units. The fixed tilt is a
    modeling assumption, not a calibrated expected return. Network penalty
    uses trailing-window MST centrality; a volatility regime sets cash.
    """
    r = training_returns.dropna(how='any').sort_index()
    tickers = list(training_returns.columns)
    if len(r) < 60 or r.shape[1] < 3:
        raise ValueError('at least 60 complete trading days and three assets are required')
    n = len(tickers)
    if n * max_weight < 1 - 1e-10:
        raise ValueError('maximum weight makes full investment infeasible')
    p = probabilities.reindex(tickers)
    if p.isna().any() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('invalid or missing model probabilities')
    graph = build_mst(correlation_matrix(r))
    central = compute_centralities(graph)
    ev = central['eigenvector'].fillna(central['degree']).reindex(tickers)
    span = float(ev.max() - ev.min())
    score = (ev - ev.min()) / span if span > 1e-12 else pd.Series(0.5, index=tickers)
    mkt_vol = r.mean(axis=1).rolling(20).std().dropna()
    low, high = mkt_vol.quantile([.33, .67])
    regime = 'Bull' if mkt_vol.iloc[-1] <= low else ('Bear' if mkt_vol.iloc[-1] >= high else 'Volatile')
    exposure = {'Bull': 1.0, 'Volatile': 0.7, 'Bear': 0.5}[regime]
    cov = r.cov().to_numpy()
    mu = r.mean().to_numpy() + signal_scale * (p.to_numpy() - 0.5)
    sc = score.to_numpy()
    def objective(w):
        risk = np.sqrt(max(float(w @ cov @ w), 0.0))
        sharpe_daily = (w @ mu - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR) / (risk + 1e-12)
        risk_penalty = network_penalty * ((w * sc) @ cov @ (w * sc))
        return -(sharpe_daily - risk_penalty)
    lo = 0.0 if long_only else -max_weight
    result = minimize(objective, np.full(n, 1 / n), method='SLSQP',
                      bounds=[(lo, max_weight)] * n,
                      constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}],
                      options={'maxiter': 300, 'ftol': 1e-9})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'portfolio optimization failed: {result.message}')
    weights = pd.Series(result.x * exposure, index=tickers, name='weight')
    return weights, {'regime': regime, 'cash_pct': 1 - exposure,
                     'signal_scale_daily': signal_scale, 'network_penalty': network_penalty,
                     'mean_daily': dict(zip(tickers, mu.tolist()))}


def audit_allocation(weights: pd.Series, max_weight=OPT_MAX_WEIGHT, long_only=OPT_LONG_ONLY):
    """Return validation errors for a proposed net allocation, without trading."""
    values = pd.to_numeric(weights, errors='coerce')
    errors = []
    if values.index.has_duplicates:
        return ['duplicate asset allocations']
    if values.empty or values.isna().any() or not np.isfinite(values).all():
        return ['missing or non-finite weights']
    if long_only and (values < -1e-8).any():
        errors.append('negative allocation in long-only portfolio')
    if (values.abs() > max_weight + 1e-8).any():
        errors.append('per-asset allocation exceeds limit')
    if values.sum() > 1 + 1e-8 or values.sum() < -1e-8:
        errors.append('net allocation outside zero to one')
    return errors
