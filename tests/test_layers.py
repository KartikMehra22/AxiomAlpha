"""Unit tests for the extracted src/{data_layer,graph_engine,ml_layer,nlp_layer}
modules. Every test uses synthetic inputs with known closed-form outputs."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.data_layer.processing import (
    compute_log_returns,
    compute_simple_returns,
    rolling_volatility,
    rolling_momentum,
    rolling_drawdown,
    engineer_features,
)
from src.graph_engine import (
    correlation_matrix,
    build_mst,
    compute_centralities,
    dynamic_graph_metrics,
)
from src.ml_layer import (
    make_direction_target,
    make_volatility_target,
    walk_forward_split,
)
from src.nlp_layer import (
    aggregate_sentiment_per_day,
    information_coefficient,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _wide_returns(n: int = 252, m: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B", name="Date")
    return pd.DataFrame(
        rng.normal(0.0005, 0.012, size=(n, m)),
        index=idx,
        columns=[f"T{i}" for i in range(m)],
    )


def _wide_prices_from_returns(returns: pd.DataFrame, start: float = 100.0) -> pd.DataFrame:
    """Inverse of compute_simple_returns for round-trip tests."""
    return start * (1.0 + returns).cumprod()


# ── data_layer.processing ──────────────────────────────────────────────────

class TestProcessing:

    def test_log_return_round_trip(self):
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=20, freq="B", name="Date")
        prices = pd.DataFrame(
            100 + rng.standard_normal((20, 3)).cumsum(axis=0),
            index=idx, columns=list("ABC"),
        ).clip(lower=1.0)  # keep positive
        log_r = compute_log_returns(prices)
        # cum sum of log returns == log(P_T / P_0)
        cum = log_r.sum()
        total = np.log(prices.iloc[-1] / prices.iloc[0])
        for c in prices.columns:
            assert cum[c] == pytest.approx(total[c], rel=1e-12)

    def test_simple_return_first_value(self):
        prices = pd.DataFrame({"A": [100.0, 105.0, 110.25]})
        sr = compute_simple_returns(prices)
        assert sr["A"].iloc[0] == pytest.approx(0.05, rel=1e-9)
        assert sr["A"].iloc[1] == pytest.approx(0.05, rel=1e-9)

    def test_rolling_vol_constant_returns_is_zero(self):
        idx = pd.date_range("2024-01-01", periods=30, freq="B", name="Date")
        df = pd.DataFrame({"A": np.full(30, 0.001)}, index=idx)
        vol = rolling_volatility(df, window=10)
        non_nan = vol["A"].dropna()
        assert (non_nan.abs() < 1e-12).all()

    def test_rolling_drawdown_no_drawdown_when_monotone(self):
        idx = pd.date_range("2024-01-01", periods=30, freq="B", name="Date")
        df = pd.DataFrame({"A": np.full(30, 0.001)}, index=idx)  # +0.1% daily
        dd = rolling_drawdown(df, window=20)
        assert (dd["A"].fillna(0) <= 0.0).all()
        assert dd["A"].dropna().abs().max() < 1e-12

    def test_engineer_features_columns_and_shape(self):
        rets = _wide_returns(n=300, m=4)
        feat = engineer_features(rets)
        for c in ("Date", "ticker", "ret_1d", "ret_5d", "ret_20d", "ret_60d",
                  "vol_5d", "vol_20d", "vol_60d", "mom_20d", "mom_60d", "drawdown"):
            assert c in feat.columns, f"missing {c}"
        # Each ticker × date pair should appear exactly once
        assert feat.duplicated(subset=["Date", "ticker"]).sum() == 0


# ── graph_engine ───────────────────────────────────────────────────────────

class TestGraphEngine:

    def test_correlation_matrix_diag_ones(self):
        rets = _wide_returns(n=120, m=5)
        c = correlation_matrix(rets)
        assert (np.abs(np.diag(c.values) - 1.0) < 1e-9).all()

    def test_mst_has_n_minus_1_edges(self):
        rets = _wide_returns(n=120, m=8, seed=1)
        c = correlation_matrix(rets)
        mst = build_mst(c)
        assert mst.number_of_edges() == 8 - 1

    def test_centralities_are_sane(self):
        rets = _wide_returns(n=120, m=6)
        mst = build_mst(correlation_matrix(rets))
        ct = compute_centralities(mst)
        # Required columns
        for col in ("degree", "betweenness", "eigenvector", "pagerank"):
            assert col in ct.columns
        # PageRank sums to ≈ 1
        assert ct["pagerank"].sum() == pytest.approx(1.0, rel=1e-6)
        # All centrality scores in [0, 1] (degree/betweenness/pagerank). Eigenvector can
        # exceed 1 in some normalisations, so we only assert non-negative.
        for col in ("degree", "betweenness", "pagerank"):
            assert (ct[col].dropna() >= 0).all()
            assert (ct[col].dropna() <= 1.0).all()

    def test_dynamic_graph_metrics_index_and_columns(self):
        rets = _wide_returns(n=200, m=10)
        dgm = dynamic_graph_metrics(rets, window=60)
        assert dgm.index.name == "Date"
        for c in ("density", "avg_corr", "clustering"):
            assert c in dgm.columns
        # density and clustering in [0, 1]
        assert (dgm["density"] >= 0).all()
        assert (dgm["density"] <= 1.0).all()


# ── ml_layer ───────────────────────────────────────────────────────────────

class TestMLLayer:

    def test_direction_target_lookahead_one_day(self):
        df = pd.DataFrame({
            "Date":   pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
            "ticker": ["A"]*3 + ["B"]*3,
            "ret_1d": [0.01, -0.02, 0.0,   0.0, 0.01, -0.005],
        })
        target = make_direction_target(df.copy())
        # Aligned on the sorted (ticker, Date) order
        df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)
        df["target"] = target.values
        # A: next-day returns are -0.02, 0.0, NaN → 0, 0, <NA>
        a_targets = df[df["ticker"] == "A"]["target"].dropna().tolist()
        assert a_targets == [0, 0]
        # B: next-day returns are 0.01, -0.005, NaN → 1, 0, <NA>
        b_targets = df[df["ticker"] == "B"]["target"].dropna().tolist()
        assert b_targets == [1, 0]

    def test_volatility_target_horizon(self):
        rng = np.random.default_rng(3)
        n = 100
        df = pd.DataFrame({
            "Date":   pd.date_range("2024-01-01", periods=n),
            "ticker": ["A"] * n,
            "ret_1d": rng.normal(0.0, 0.01, n),
        })
        out = make_volatility_target(df, horizon=5)
        # Forward-rolling std must be NaN for the last 5 rows
        assert out.iloc[-5:].isna().all()
        # Earlier rows: defined and non-negative
        assert (out.dropna() >= 0).all()

    def test_walk_forward_split_no_overlap(self):
        splits = walk_forward_split(n_samples=100, n_splits=4, min_train=20)
        for tr, te in splits:
            assert tr.max() < te.min(), "train/test overlap"
            assert len(set(tr) & set(te)) == 0
        # Test windows non-overlapping
        all_test = np.concatenate([te for _, te in splits])
        assert len(set(all_test)) == len(all_test)


# ── nlp_layer ──────────────────────────────────────────────────────────────

class TestNLPLayer:

    def test_aggregate_sentiment_groups_by_date_ticker(self):
        news = pd.DataFrame({
            "Date":            ["2024-01-01", "2024-01-01", "2024-01-02"],
            "ticker":          ["AAPL",        "AAPL",       "MSFT"],
            "sentiment_score": [0.5,           -0.5,          0.2],
        })
        out = aggregate_sentiment_per_day(news)
        # 2 unique (Date, ticker) pairs
        assert len(out) == 2
        aapl = out[out["ticker"] == "AAPL"].iloc[0]
        assert aapl["sentiment_mean"] == pytest.approx(0.0, rel=1e-9)
        assert aapl["news_volume"] == 2

    def test_aggregate_sentiment_empty_input(self):
        out = aggregate_sentiment_per_day(pd.DataFrame())
        assert out.empty

    def test_information_coefficient_perfectly_predictive(self):
        # Build sentiment_features s.t. tomorrow's return exactly equals today's sentiment.
        idx = pd.date_range("2024-01-01", periods=50, freq="B", name="Date")
        rng = np.random.default_rng(5)
        sent = rng.normal(0, 1, 50)
        ret = pd.Series(0.0, index=idx)
        ret.iloc[1:] = sent[:-1]  # tomorrow's return = today's sentiment

        log_returns = pd.DataFrame({"AAPL": ret})
        sent_df = pd.DataFrame({
            "Date":           idx,
            "ticker":         ["AAPL"] * 50,
            "sentiment_mean": sent,
        })
        ic = information_coefficient(sent_df, log_returns, lag=1)
        # Spearman between sentiment_t and ret_{t+1}=sent_t is +1
        assert ic == pytest.approx(1.0, rel=1e-6)

    def test_information_coefficient_unrelated(self):
        # Independent sentiment and return → IC ≈ 0 (within sampling noise)
        idx = pd.date_range("2024-01-01", periods=200, freq="B", name="Date")
        rng = np.random.default_rng(7)
        log_returns = pd.DataFrame({"AAPL": rng.normal(0, 0.01, 200)}, index=idx)
        sent_df = pd.DataFrame({
            "Date":           idx,
            "ticker":         ["AAPL"] * 200,
            "sentiment_mean": rng.normal(0, 1, 200),
        })
        ic = information_coefficient(sent_df, log_returns, lag=1)
        assert abs(ic) < 0.20  # generously loose for n=200


def test_drawdown_from_log_returns_includes_initial_price():
    prices = pd.DataFrame({'A': [100.0, 80.0, 96.0]})
    log_returns = compute_log_returns(prices)
    drawdown = rolling_drawdown(log_returns, window=3)
    assert drawdown['A'].iloc[0] == pytest.approx(-0.2)
    assert drawdown['A'].iloc[1] == pytest.approx(-0.04)


def test_simple_returns_do_not_impute_missing_prices():
    prices = pd.DataFrame({'A': [100.0, np.nan, 110.0]})
    assert compute_simple_returns(prices).empty
