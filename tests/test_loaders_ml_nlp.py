import pytest
import numpy as np
import pandas as pd
from src.io.loaders import (
    load_ml_signals, load_models, load_master_signals,
    load_signal_explanations,
    load_synthetic_news, load_sentiment_features,
    load_embeddings, load_faiss_index, load_rag_documents,
)

pytestmark = pytest.mark.requires_artifacts


def test_load_ml_signals_long_format():
    df = load_ml_signals()
    assert 'Date' in df.columns and 'ticker' in df.columns


def test_load_models_returns_three_models():
    m_dir, m_vol, m_reg = load_models()
    for m in (m_dir, m_vol, m_reg):
        assert hasattr(m, 'predict')


def test_load_master_signals_has_required_columns():
    df = load_master_signals()
    needed = {'Date', 'ticker', 'alpha_signal'}
    assert needed.issubset(df.columns), f"missing: {needed - set(df.columns)}"


def test_load_synthetic_news_has_text_column():
    df = load_synthetic_news()
    assert any(c in df.columns for c in ['headline', 'text', 'title'])


def test_load_sentiment_features_long_format():
    df = load_sentiment_features()
    assert 'Date' in df.columns and 'ticker' in df.columns


def test_load_embeddings_returns_2d_array():
    arr = load_embeddings()
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    assert arr.shape[1] in (128, 256, 384, 512, 768), \
        f"unexpected embedding dim {arr.shape[1]}"
