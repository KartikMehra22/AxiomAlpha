import pytest
import pandas as pd
from src.io.loaders import (
    load_risk_metrics, load_regimes,
    load_mst_edges, load_dynamic_graph_metrics,
    load_centrality_scores, load_community_assignments,
    load_systemic_risk_scores, load_features_with_graph,
    load_granger_causality, load_leader_follower, load_graph_intelligence,
)

pytestmark = pytest.mark.requires_artifacts


def test_load_risk_metrics_has_date_and_ticker():
    df = load_risk_metrics()
    assert 'Date' in df.columns or df.index.name == 'Date'


def test_load_regimes_has_date_index_or_column():
    df = load_regimes()
    assert df.index.name == 'Date' or 'Date' in df.columns


def test_load_mst_edges_returns_dataframe():
    df = load_mst_edges()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_dynamic_graph_metrics_has_date_index():
    df = load_dynamic_graph_metrics()
    assert df.index.name == 'Date'


def test_load_centrality_scores_returns_dataframe():
    df = load_centrality_scores()
    assert isinstance(df, pd.DataFrame)


def test_load_features_with_graph_long_format():
    df = load_features_with_graph()
    assert 'Date' in df.columns
    assert 'ticker' in df.columns


def test_load_community_assignments_smoke():
    df = load_community_assignments()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_systemic_risk_scores_smoke():
    df = load_systemic_risk_scores()
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == 'ticker'
    assert len(df) > 0


def test_load_granger_causality_smoke():
    df = load_granger_causality()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_leader_follower_smoke():
    df = load_leader_follower()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_graph_intelligence_smoke():
    df = load_graph_intelligence()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
