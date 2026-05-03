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


def load_ml_signals() -> pd.DataFrame:
    return _load_long_csv(PROC / 'ml_signals.csv',
                          producer_nb='NB06_ml_models', name='ml_signals')


def load_models() -> tuple:
    """Returns (model_direction, model_volatility, model_regime). Producer: NB06."""
    paths = [PROC / f'model_{kind}.pkl' for kind in ('direction', 'volatility', 'regime')]
    for p in paths:
        _require(p, producer_nb='NB06_ml_models')
    return tuple(joblib.load(p) for p in paths)


def load_master_signals() -> pd.DataFrame:
    return _load_long_csv(
        PROC / 'master_signals.csv',
        producer_nb='NB07_ml_analysis', name='master_signals',
        required_cols=['alpha_signal'],
    )


def load_signal_explanations() -> dict:
    p = _require(PROC / 'signal_explanations.json',
                 producer_nb='NB07_ml_analysis')
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def load_synthetic_news() -> pd.DataFrame:
    p = _require(RAW / 'synthetic_news.csv', producer_nb='NB08_nlp_sentiment')
    # parse_dates only if Date column exists (handles both schemas)
    df = pd.read_csv(p)
    date_col = next((c for c in df.columns if c.lower() == 'date'), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
        if date_col != 'Date':
            df = df.rename(columns={date_col: 'Date'})
    return df


def load_sentiment_features() -> pd.DataFrame:
    return _load_long_csv(PROC / 'sentiment_features.csv',
                          producer_nb='NB08_nlp_sentiment', name='sentiment_features')


def load_embeddings() -> np.ndarray:
    p = _require(PROC / 'embeddings.npy', producer_nb='NB09_rag_setup')
    arr = np.load(p)
    if arr.ndim != 2:
        raise PipelineDataError(f"embeddings.npy: expected 2D, got {arr.shape}")
    return arr


def load_faiss_index():
    """Returns a faiss.Index instance. Producer: NB09."""
    import faiss
    p = _require(PROC / 'faiss_index.bin', producer_nb='NB09_rag_setup')
    return faiss.read_index(str(p))


def load_rag_documents() -> list[dict]:
    p = _require(PROC / 'rag_documents.json', producer_nb='NB09_rag_setup')
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def load_optimal_weights() -> pd.DataFrame:
    return _load_long_csv(
        PROC / 'optimal_weights.csv',
        producer_nb='NB11_portfolio_optimization', name='optimal_weights',
        required_cols=['weight'],
    )


def load_equity_curves() -> pd.DataFrame:
    return _load_wide_csv(PROC / 'equity_curves.csv',
                          producer_nb='NB12_backtesting', name='equity_curves')


def load_performance_metrics() -> dict:
    p = _require(PROC / 'performance_metrics.csv',
                 producer_nb='NB12_backtesting')
    df = pd.read_csv(p)
    if {'metric', 'value'}.issubset(df.columns):
        return dict(zip(df['metric'], df['value']))
    if df.shape[0] == 1:
        return df.iloc[0].to_dict()
    return df.to_dict(orient='list')


def load_attribution_results() -> pd.DataFrame:
    p = _require(PROC / 'attribution_results.csv',
                 producer_nb='NB12_backtesting')
    return pd.read_csv(p)


def load_regime_performance() -> pd.DataFrame:
    p = _require(PROC / 'regime_performance.csv',
                 producer_nb='NB12_backtesting')
    return pd.read_csv(p)


def load_backtest_results() -> pd.DataFrame:
    p = _require(PROC / 'backtest_results.csv',
                 producer_nb='NB12_backtesting')
    df = pd.read_csv(p)
    date_col = next((c for c in df.columns if c.lower() == 'date'), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
        if date_col != 'Date':
            df = df.rename(columns={date_col: 'Date'})
    return df


def load_pipeline_output() -> dict:
    p = _require(PROC / 'pipeline_output.json',
                 producer_nb='NB14_agent_orchestration')
    with open(p, encoding='utf-8') as f:
        return json.load(f)
