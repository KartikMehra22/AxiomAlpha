"""Single source of truth for every pipeline read.

Every notebook imports from this module instead of using pd.read_csv directly.
Every loader validates schema and raises PipelineDataError on missing/malformed
inputs, with a message naming the producer notebook.
"""
from __future__ import annotations

import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd


class PipelineDataError(FileNotFoundError):
    """Raised when a pipeline artifact is missing or malformed."""


def _find_project_root() -> Path:
    """Walk up from this file until we find the AxiomAlpha directory."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if parent.name == 'AxiomAlpha':
            return parent
    raise RuntimeError(
        f"Could not locate AxiomAlpha project root from {here}"
    )


PROJECT_ROOT = _find_project_root()
RAW = PROJECT_ROOT / 'data' / 'raw'
PROC = PROJECT_ROOT / 'data' / 'processed'
OUTPUTS = PROJECT_ROOT / 'data' / 'outputs'


def _require(path: Path, *, producer_nb: str) -> Path:
    """Return path if it exists, else raise PipelineDataError naming producer."""
    if not path.exists():
        raise PipelineDataError(
            f"Missing pipeline artifact: {path.name}\n"
            f"  Expected at: {path}\n"
            f"  Producer:    {producer_nb}\n"
            f"  Fix: run {producer_nb} first (or `python run_all.py`)."
        )
    return path


def _validate(
    df: pd.DataFrame,
    *,
    required_cols: list[str],
    name: str,
    monotonic_index: bool = False,
) -> pd.DataFrame:
    """Schema validation; raises PipelineDataError on failure."""
    have = set(df.columns)
    if df.index.name:
        have.add(df.index.name)
    missing = set(required_cols) - have
    if missing:
        raise PipelineDataError(
            f"{name}: missing columns {sorted(missing)}. "
            f"Got columns: {list(df.columns)}, index: {df.index.name}"
        )
    if monotonic_index and not df.index.is_monotonic_increasing:
        raise PipelineDataError(
            f"{name}: index is not monotonically increasing"
        )
    return df


def load_prices() -> pd.DataFrame:
    """Daily close prices, wide format (Date index, ticker columns). Producer: NB01."""
    p = _require(RAW / 'prices.csv', producer_nb='NB01_data_pipeline')
    df = pd.read_csv(p, index_col='Date', parse_dates=['Date'])
    df.index = df.index.normalize()  # date-only, drop tz/time
    return _validate(df, required_cols=[], name='prices', monotonic_index=True)


def load_macro() -> pd.DataFrame:
    """Daily macro factors (VIX, SP500, TNX). Producer: NB01."""
    p = _require(RAW / 'macro.csv', producer_nb='NB01_data_pipeline')
    df = pd.read_csv(p, index_col='Date', parse_dates=['Date'])
    df.index = df.index.normalize()
    return _validate(df, required_cols=[], name='macro', monotonic_index=True)


def load_sector_map() -> dict[str, str]:
    """Ticker → GICS sector. Producer: NB01."""
    p = _require(RAW / 'sector_map.json', producer_nb='NB01_data_pipeline')
    with open(p, encoding='utf-8') as f:
        sm = json.load(f)
    if not isinstance(sm, dict):
        raise PipelineDataError(f"sector_map.json: expected dict, got {type(sm)}")
    return sm


def _load_wide_csv(path: Path, *, producer_nb: str, name: str) -> pd.DataFrame:
    """Wide-format CSV: Date as index, no other index columns."""
    p = _require(path, producer_nb=producer_nb)
    df = pd.read_csv(p, index_col='Date', parse_dates=['Date'])
    df.index = df.index.normalize()
    return _validate(df, required_cols=[], name=name, monotonic_index=True)


def _load_long_csv(
    path: Path, *, producer_nb: str, name: str,
    required_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Long-format CSV: Date and ticker as columns (not index)."""
    p = _require(path, producer_nb=producer_nb)
    df = pd.read_csv(p, parse_dates=['Date'])
    df['Date'] = df['Date'].dt.normalize()
    return _validate(
        df,
        required_cols=['Date', 'ticker'] + (required_cols or []),
        name=name,
    )


def load_log_returns() -> pd.DataFrame:
    return _load_wide_csv(PROC / 'log_returns.csv',
                          producer_nb='NB02_data_processing', name='log_returns')


def load_simple_returns() -> pd.DataFrame:
    return _load_wide_csv(PROC / 'simple_returns.csv',
                          producer_nb='NB02_data_processing', name='simple_returns')


def load_features_raw() -> pd.DataFrame:
    return _load_long_csv(PROC / 'features_raw.csv',
                          producer_nb='NB02_data_processing', name='features_raw')


def load_features_normalized() -> pd.DataFrame:
    return _load_long_csv(PROC / 'features_normalized.csv',
                          producer_nb='NB02_data_processing', name='features_normalized')


def load_feature_scaler():
    """Sklearn StandardScaler fitted on training features. Producer: NB02."""
    p = _require(PROC / 'feature_scaler.pkl',
                 producer_nb='NB02_data_processing')
    return joblib.load(p)


def load_risk_metrics() -> pd.DataFrame:
    return _load_long_csv(PROC / 'risk_metrics.csv',
                          producer_nb='NB03_quant_core', name='risk_metrics')


def load_regimes() -> pd.DataFrame:
    return _load_wide_csv(PROC / 'regimes.csv',
                          producer_nb='NB03_quant_core', name='regimes')


def load_mst_edges() -> pd.DataFrame:
    p = _require(PROC / 'mst_edges.csv', producer_nb='NB04_graph_engine')
    return pd.read_csv(p)


def load_dynamic_graph_metrics() -> pd.DataFrame:
    p = _require(PROC / 'dynamic_graph_metrics.csv',
                 producer_nb='NB04_graph_engine')
    df = pd.read_csv(p)
    # TODO(Task 17): NB04 will emit 'Date' (capitalized) — remove the
    # case-insensitive rename below once that lands.
    date_col = next((c for c in df.columns if c.lower() == 'date'), None)
    if date_col is None:
        raise PipelineDataError(
            "dynamic_graph_metrics: no date column found. "
            f"Got columns: {list(df.columns)}"
        )
    if date_col != 'Date':
        df = df.rename(columns={date_col: 'Date'})
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    df = df.set_index('Date')
    df.index.name = 'Date'
    return df


def load_centrality_scores() -> pd.DataFrame:
    p = _require(PROC / 'centrality_scores.csv',
                 producer_nb='NB05_graph_analysis')
    return pd.read_csv(p)


def load_community_assignments() -> pd.DataFrame:
    p = _require(PROC / 'community_assignments.csv',
                 producer_nb='NB05_graph_analysis')
    return pd.read_csv(p)


def load_systemic_risk_scores() -> pd.DataFrame:
    return _load_wide_csv(PROC / 'systemic_risk_scores.csv',
                          producer_nb='NB05_graph_analysis',
                          name='systemic_risk_scores')


def load_features_with_graph() -> pd.DataFrame:
    return _load_long_csv(PROC / 'features_with_graph.csv',
                          producer_nb='NB05_graph_analysis',
                          name='features_with_graph')


def load_granger_causality() -> pd.DataFrame:
    p = _require(PROC / 'granger_causality.csv',
                 producer_nb='NB05_graph_analysis')
    return pd.read_csv(p)


def load_leader_follower() -> pd.DataFrame:
    p = _require(PROC / 'leader_follower.csv',
                 producer_nb='NB05_graph_analysis')
    return pd.read_csv(p)


def load_graph_intelligence() -> pd.DataFrame:
    p = _require(PROC / 'graph_intelligence.csv',
                 producer_nb='NB05_graph_analysis')
    return pd.read_csv(p)
