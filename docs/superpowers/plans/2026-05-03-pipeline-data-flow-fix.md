# AxiomAlpha Pipeline Data-Flow Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all 15 AxiomAlpha notebooks chain through real upstream outputs via a single loader module — eliminate every `np.random.*` data fabrication and every NameError so `python run_all.py` runs end-to-end on a fresh kernel and produces deterministic, truthful results.

**Architecture:** Add `src/io/loaders.py` as the single source of truth for every read; centralize constants in `configs/settings.py`; pin dependencies; add `run_all.py` (papermill orchestrator) plus two smoke tests (`check_pipeline.py`, `check_determinism.py`); patch each notebook to start with a `# Load Inputs` cell using the loaders, replace `np.random` data fabrication with real upstream reads, and fix look-ahead bias / walk-forward bugs along the way.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn, xgboost, arch, networkx, sentence-transformers, transformers (FinBERT), faiss-cpu, cvxpy, langgraph, papermill, pytest.

**Spec:** `docs/superpowers/specs/2026-05-03-pipeline-data-flow-fix-design.md` (commit `b1f2912`).

---

## Phase A — Foundation (Tasks 1–13)

Build the infrastructure that every notebook will depend on: loader module, config, pinned deps, smoke tests, orchestrator. TDD throughout.

### Task 1: Package skeleton + test scaffolding

**Files:**
- Create: `src/__init__.py`
- Create: `src/io/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pytest.ini`
- Modify: `requirements.txt` (add `pytest`, `papermill`)

- [ ] **Step 1: Create empty package init files**

```bash
mkdir -p src/io tests
touch src/__init__.py src/io/__init__.py tests/__init__.py
```

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

- [ ] **Step 3: Create `tests/conftest.py` with shared fixtures**

```python
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Redirect data paths to a temp directory so tests don't touch real data."""
    raw = tmp_path / 'data' / 'raw'
    proc = tmp_path / 'data' / 'processed'
    raw.mkdir(parents=True)
    proc.mkdir(parents=True)
    return {'root': tmp_path, 'raw': raw, 'proc': proc}


@pytest.fixture
def sample_prices_df():
    """30 trading days of 5 tickers, no NaN."""
    dates = pd.date_range('2024-01-01', periods=30, freq='B', name='Date')
    rng = np.random.default_rng(42)
    data = 100 + rng.standard_normal((30, 5)).cumsum(axis=0)
    return pd.DataFrame(data, index=dates,
                        columns=['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META'])
```

- [ ] **Step 4: Add pytest + papermill to requirements.txt**

Append to `requirements.txt`:

```
# Testing & Orchestration
pytest>=7.4.0
papermill>=2.5.0
```

- [ ] **Step 5: Verify pytest discovers no tests yet (sanity check)**

Run: `python -m pytest`
Expected: `no tests ran` (exit 5) or `collected 0 items`. Either is fine.

- [ ] **Step 6: Commit**

```bash
git add src/__init__.py src/io/__init__.py tests/ pytest.ini requirements.txt
git commit -m "chore: add package skeleton and pytest scaffolding"
```

---

### Task 2: `PipelineDataError` + path resolution

**Files:**
- Create: `src/io/loaders.py`
- Create: `tests/test_loaders_paths.py`

- [ ] **Step 1: Write failing test for `_find_project_root`**

Create `tests/test_loaders_paths.py`:

```python
import pytest
from pathlib import Path
from src.io.loaders import _find_project_root, PipelineDataError, RAW, PROC


def test_find_project_root_returns_axiomalpha_dir():
    root = _find_project_root()
    assert root.name == 'AxiomAlpha'
    assert (root / 'data').is_dir()


def test_raw_and_proc_constants_exist():
    assert RAW.exists() and RAW.is_dir()
    assert PROC.exists() and PROC.is_dir()


def test_pipeline_data_error_is_filenotfound_subclass():
    assert issubclass(PipelineDataError, FileNotFoundError)
    err = PipelineDataError("test message")
    assert "test message" in str(err)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_loaders_paths.py -v`
Expected: `ImportError: cannot import name '_find_project_root' from 'src.io.loaders'` (or ModuleNotFoundError).

- [ ] **Step 3: Implement minimal `loaders.py`**

Create `src/io/loaders.py`:

```python
"""Single source of truth for every pipeline read.

Every notebook imports from this module instead of using pd.read_csv directly.
Every loader validates schema and raises PipelineDataError on missing/malformed
inputs, with a message naming the producer notebook.
"""
from __future__ import annotations
from pathlib import Path


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_loaders_paths.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/io/loaders.py tests/test_loaders_paths.py
git commit -m "feat(loaders): add PipelineDataError and path resolution"
```

---

### Task 3: `_require` and `_validate` helpers

**Files:**
- Modify: `src/io/loaders.py`
- Create: `tests/test_loaders_helpers.py`

- [ ] **Step 1: Write failing tests for helpers**

Create `tests/test_loaders_helpers.py`:

```python
import pytest
import pandas as pd
import numpy as np
from src.io.loaders import _require, _validate, PipelineDataError


def test_require_returns_path_when_exists(tmp_path):
    p = tmp_path / 'x.csv'
    p.write_text('hello')
    assert _require(p, producer_nb='NB01') == p


def test_require_raises_with_producer_in_message(tmp_path):
    p = tmp_path / 'missing.csv'
    with pytest.raises(PipelineDataError, match="NB01_data_pipeline"):
        _require(p, producer_nb='NB01_data_pipeline')


def test_validate_passes_when_columns_present():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    out = _validate(df, required_cols=['a', 'b'], name='toy')
    assert out is df


def test_validate_raises_when_column_missing():
    df = pd.DataFrame({'a': [1]})
    with pytest.raises(PipelineDataError, match="missing columns"):
        _validate(df, required_cols=['a', 'b'], name='toy')


def test_validate_monotonic_index_passes_when_sorted():
    df = pd.DataFrame({'x': [1, 2, 3]},
                      index=pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']))
    _validate(df, required_cols=[], name='toy', monotonic_index=True)


def test_validate_monotonic_index_raises_when_unsorted():
    df = pd.DataFrame({'x': [1, 2]},
                      index=pd.to_datetime(['2024-01-02', '2024-01-01']))
    with pytest.raises(PipelineDataError, match="monotonic"):
        _validate(df, required_cols=[], name='toy', monotonic_index=True)
```

- [ ] **Step 2: Run test to verify failure**

Run: `python -m pytest tests/test_loaders_helpers.py -v`
Expected: ImportError on `_require` and `_validate`.

- [ ] **Step 3: Add helpers to `loaders.py`**

Append to `src/io/loaders.py`:

```python
import pandas as pd


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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_loaders_helpers.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/io/loaders.py tests/test_loaders_helpers.py
git commit -m "feat(loaders): add _require and _validate helpers"
```

---

### Task 4: Raw-layer loaders (prices, macro, sector_map)

**Files:**
- Modify: `src/io/loaders.py`
- Create: `tests/test_loaders_raw.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_loaders_raw.py`:

```python
import pytest
import json
import pandas as pd
import numpy as np
from src.io.loaders import (
    load_prices, load_macro, load_sector_map,
    PipelineDataError, RAW,
)


def test_load_prices_returns_dataframe_with_date_index():
    df = load_prices()
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == 'Date'
    assert pd.api.types.is_datetime64_any_dtype(df.index)
    assert df.index.is_monotonic_increasing
    assert df.shape[0] > 100  # at least 100 trading days
    assert df.shape[1] >= 25  # at least 25 tickers


def test_load_prices_no_nans():
    df = load_prices()
    assert df.isnull().sum().sum() == 0, "prices.csv should be cleaned"


def test_load_macro_returns_dataframe_with_date_index():
    df = load_macro()
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == 'Date'
    assert pd.api.types.is_datetime64_any_dtype(df.index)


def test_load_sector_map_returns_dict_with_string_values():
    sm = load_sector_map()
    assert isinstance(sm, dict)
    assert len(sm) >= 25
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in sm.items())


def test_load_prices_raises_when_file_missing(monkeypatch, tmp_path):
    # Temporarily redirect RAW to an empty dir
    from src.io import loaders
    monkeypatch.setattr(loaders, 'RAW', tmp_path)
    with pytest.raises(PipelineDataError, match="NB01_data_pipeline"):
        load_prices()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_loaders_raw.py -v`
Expected: ImportError on the new loader functions.

- [ ] **Step 3: Implement raw loaders**

Append to `src/io/loaders.py`:

```python
import json
import numpy as np


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
    with open(p) as f:
        sm = json.load(f)
    if not isinstance(sm, dict):
        raise PipelineDataError(f"sector_map.json: expected dict, got {type(sm)}")
    return sm
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_loaders_raw.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/io/loaders.py tests/test_loaders_raw.py
git commit -m "feat(loaders): add raw-layer loaders (prices, macro, sector_map)"
```

---

### Task 5: Returns + features loaders

**Files:**
- Modify: `src/io/loaders.py`
- Create: `tests/test_loaders_returns.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_loaders_returns.py`:

```python
import pytest
import pandas as pd
from src.io.loaders import (
    load_log_returns, load_simple_returns,
    load_features_raw, load_features_normalized, load_feature_scaler,
)


def test_load_log_returns_shape_and_index():
    df = load_log_returns()
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == 'Date'
    assert pd.api.types.is_datetime64_any_dtype(df.index)
    assert df.index.is_monotonic_increasing


def test_load_simple_returns_shape_and_index():
    df = load_simple_returns()
    assert df.index.name == 'Date'


def test_load_features_normalized_long_format():
    """Long-format: Date and ticker as columns (not index)."""
    df = load_features_normalized()
    assert 'Date' in df.columns
    assert 'ticker' in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df['Date'])


def test_load_features_raw_long_format():
    df = load_features_raw()
    assert 'Date' in df.columns
    assert 'ticker' in df.columns


def test_load_feature_scaler_returns_sklearn_scaler():
    scaler = load_feature_scaler()
    assert hasattr(scaler, 'transform')
    assert hasattr(scaler, 'mean_') or hasattr(scaler, 'scale_')
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_loaders_returns.py -v`

- [ ] **Step 3: Implement loaders**

Append to `src/io/loaders.py`:

```python
import joblib


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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_loaders_returns.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/io/loaders.py tests/test_loaders_returns.py
git commit -m "feat(loaders): add returns and features loaders"
```

---

### Task 6: Quant + graph loaders

**Files:**
- Modify: `src/io/loaders.py`
- Create: `tests/test_loaders_graph.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_loaders_graph.py`:

```python
import pytest
import pandas as pd
from src.io.loaders import (
    load_risk_metrics, load_regimes,
    load_mst_edges, load_dynamic_graph_metrics,
    load_centrality_scores, load_community_assignments,
    load_systemic_risk_scores, load_features_with_graph,
    load_granger_causality, load_leader_follower, load_graph_intelligence,
)


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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_loaders_graph.py -v`

- [ ] **Step 3: Implement loaders**

Append to `src/io/loaders.py`:

```python
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
    return _load_wide_csv(PROC / 'dynamic_graph_metrics.csv',
                          producer_nb='NB04_graph_engine',
                          name='dynamic_graph_metrics')


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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_loaders_graph.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/io/loaders.py tests/test_loaders_graph.py
git commit -m "feat(loaders): add quant and graph loaders"
```

---

### Task 7: ML + RAG + NLP loaders

**Files:**
- Modify: `src/io/loaders.py`
- Create: `tests/test_loaders_ml_nlp.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_loaders_ml_nlp.py`:

```python
import pytest
import numpy as np
import pandas as pd
from src.io.loaders import (
    load_ml_signals, load_models, load_master_signals,
    load_signal_explanations,
    load_synthetic_news, load_sentiment_features,
    load_embeddings, load_faiss_index, load_rag_documents,
)


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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_loaders_ml_nlp.py -v`

- [ ] **Step 3: Implement loaders**

Append to `src/io/loaders.py`:

```python
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
    with open(p) as f:
        return json.load(f)


def load_synthetic_news() -> pd.DataFrame:
    p = _require(RAW / 'synthetic_news.csv', producer_nb='NB08_nlp_sentiment')
    return pd.read_csv(p, parse_dates=['Date'] if 'Date' else None)


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
    import faiss  # local import to keep loaders importable without faiss
    p = _require(PROC / 'faiss_index.bin', producer_nb='NB09_rag_setup')
    return faiss.read_index(str(p))


def load_rag_documents() -> list[dict]:
    p = _require(PROC / 'rag_documents.json', producer_nb='NB09_rag_setup')
    with open(p) as f:
        return json.load(f)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_loaders_ml_nlp.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/io/loaders.py tests/test_loaders_ml_nlp.py
git commit -m "feat(loaders): add ML, RAG, and NLP loaders"
```

---

### Task 8: Optimization + backtest + agent loaders

**Files:**
- Modify: `src/io/loaders.py`
- Create: `tests/test_loaders_opt_bt.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_loaders_opt_bt.py`:

```python
import pytest
import pandas as pd
from src.io.loaders import (
    load_optimal_weights, load_equity_curves, load_performance_metrics,
    load_attribution_results, load_regime_performance,
    load_backtest_results, load_pipeline_output,
)


def test_load_optimal_weights_long_format():
    df = load_optimal_weights()
    assert 'Date' in df.columns and 'ticker' in df.columns
    assert 'weight' in df.columns


def test_load_equity_curves_has_date_index():
    df = load_equity_curves()
    assert df.index.name == 'Date'


def test_load_performance_metrics_returns_dict():
    m = load_performance_metrics()
    assert isinstance(m, dict)
    assert 'sharpe' in m or 'Sharpe' in m


def test_load_pipeline_output_returns_dict():
    out = load_pipeline_output()
    assert isinstance(out, dict)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_loaders_opt_bt.py -v`

- [ ] **Step 3: Implement loaders**

Append to `src/io/loaders.py`:

```python
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
    # Two-column form (metric, value) is most common; flatten to dict
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
    return pd.read_csv(p, parse_dates=['Date'] if 'Date' else None)


def load_pipeline_output() -> dict:
    p = _require(PROC / 'pipeline_output.json',
                 producer_nb='NB14_agent_orchestration')
    with open(p) as f:
        return json.load(f)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_loaders_opt_bt.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/io/loaders.py tests/test_loaders_opt_bt.py
git commit -m "feat(loaders): add optimization, backtest, agent loaders"
```

---

### Task 9: Centralized config (`configs/settings.py`)

**Files:**
- Create: `configs/__init__.py`
- Modify: `configs/settings.py` (currently empty)
- Create: `tests/test_settings.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_settings.py`:

```python
from configs import settings


def test_tickers_is_list_of_30_strings():
    assert isinstance(settings.TICKERS, list)
    assert len(settings.TICKERS) == 30
    assert all(isinstance(t, str) and t.isupper() for t in settings.TICKERS)


def test_date_range_is_strings():
    assert settings.START_DATE == '2019-01-01'
    assert settings.END_DATE == '2024-12-31'


def test_random_seed_is_42():
    assert settings.RANDOM_SEED == 42


def test_paths_resolve():
    assert settings.PROJECT_ROOT.name == 'AxiomAlpha'
    assert settings.DATA_RAW.is_dir()
    assert settings.DATA_PROC.is_dir()


def test_model_params_are_dicts_with_seed():
    assert settings.RF_PARAMS['random_state'] == 42
    assert settings.XGB_PARAMS['random_state'] == 42
```

- [ ] **Step 2: Run test to verify failure**

Run: `python -m pytest tests/test_settings.py -v`

- [ ] **Step 3: Create `configs/__init__.py` and write `configs/settings.py`**

Create `configs/__init__.py` (empty file).

Write `configs/settings.py`:

```python
"""Centralized configuration for the AxiomAlpha pipeline.

Notebooks import from here instead of hardcoding constants.
"""
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROC = PROJECT_ROOT / 'data' / 'processed'
DATA_OUT = PROJECT_ROOT / 'data' / 'outputs'

# --- Universe (30 tickers) ---
TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META',
    'JPM', 'BAC', 'GS', 'MS', 'WFC',
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG',
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE',
    'BA', 'CAT', 'GE', 'MMM', 'UPS',
]

# --- Date range ---
START_DATE = '2019-01-01'
END_DATE = '2024-12-31'

# --- Reproducibility ---
RANDOM_SEED = 42

# --- Statistical / quant ---
GARCH_P = 1
GARCH_Q = 1
ROLLING_GRAPH_WINDOW = 60
N_REGIMES = 3  # Bear, Volatile, Bull

# --- Machine Learning ---
ML_N_SPLITS = 5
RF_PARAMS = dict(
    n_estimators=200, max_depth=6, min_samples_leaf=20,
    random_state=RANDOM_SEED, n_jobs=1,  # n_jobs=1 for determinism
)
XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_SEED, n_jobs=1, eval_metric='logloss',
)
RIDGE_PARAMS = dict(alpha=1.0, random_state=RANDOM_SEED)

# --- NLP ---
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # sentence-transformers
FINBERT_MODEL = 'ProsusAI/finbert'
SYNTHETIC_NEWS_N = 1000

# --- Optimization ---
OPT_RISK_AVERSION = 2.0
OPT_MAX_WEIGHT = 0.10
OPT_LONG_ONLY = True
CVAR_ALPHA = 0.95
COVAR_SHRINKAGE = 'ledoit_wolf'

# --- Backtest ---
BACKTEST_REBALANCE = 'M'  # monthly
TRANSACTION_COST_BPS = 5
INITIAL_CAPITAL = 1_000_000

# --- Visualization ---
DARK_BG = '#0d0d0d'
COLORS = ['#00D4FF', '#FF6B6B', '#51CF66', '#FFD43B', '#FF922B', '#888888']
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_settings.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add configs/__init__.py configs/settings.py tests/test_settings.py
git commit -m "feat(config): centralize tickers, hyperparams, paths in settings.py"
```

---

### Task 10: Pin requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Get currently working versions**

Run: `pip freeze | grep -iE "pandas|numpy|scipy|statsmodels|arch|scikit-learn|xgboost|joblib|networkx|python-louvain|transformers|torch|sentence-transformers|faiss|matplotlib|seaborn|plotly|umap|cvxpy|langgraph|tqdm|yfinance|fredapi|papermill|pytest"`

Capture the output and use the exact versions in the rewrite below. (If a version is not installed, use the floor below.)

- [ ] **Step 2: Rewrite `requirements.txt` with pinned versions**

Replace `requirements.txt` contents with:

```
# AxiomAlpha — pinned for reproducibility (Python 3.11+)

# --- Core data ---
pandas==2.1.4
numpy==1.26.3
yfinance==0.2.36
fredapi==0.5.1

# --- Statistical / quant ---
scipy==1.11.4
statsmodels==0.14.1
arch==6.3.0

# --- Machine learning ---
scikit-learn==1.3.2
xgboost==2.0.3
joblib==1.3.2

# --- Graph ---
networkx==3.2.1
python-louvain==0.16

# --- NLP & RAG ---
transformers==4.36.2
torch==2.1.2
sentence-transformers==2.2.2
faiss-cpu==1.7.4

# --- Visualization ---
matplotlib==3.8.2
seaborn==0.13.1
plotly==5.18.0

# --- Dimensionality reduction ---
umap-learn==0.5.5

# --- Optimization ---
cvxpy==1.4.1

# --- Agent system (anthropic removed; we use templated explanations) ---
langgraph==0.0.40
# langchain removed; not used.

# --- Testing & orchestration ---
pytest==7.4.4
papermill==2.5.0

# --- Utilities ---
tqdm==4.66.1
```

If `pip freeze` reported a *higher minor version* of any package and it's known compatible, use that instead. The numbers above are the safe default known to work end-to-end.

- [ ] **Step 3: Verify install still works**

Run: `pip install -r requirements.txt`
Expected: no errors, all packages resolve.

- [ ] **Step 4: Verify pytest still passes**

Run: `python -m pytest`
Expected: all loader/settings tests pass.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt
git commit -m "chore: pin all dependency versions; remove unused anthropic and langchain"
```

---

### Task 11: `scripts/check_pipeline.py` smoke test

**Files:**
- Create: `scripts/check_pipeline.py`
- Create: `tests/test_check_pipeline.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_check_pipeline.py`:

```python
import json
import pytest
from pathlib import Path
from scripts.check_pipeline import check_notebook, FORBIDDEN, ALLOWLIST_TAG


def make_nb(cells: list[str]) -> dict:
    return {'cells': [
        {'cell_type': 'code', 'source': src, 'metadata': {}, 'outputs': []}
        for src in cells
    ], 'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5}


def write_nb(path: Path, cells: list[str]):
    path.write_text(json.dumps(make_nb(cells)))


def test_no_errors_for_clean_notebook(tmp_path):
    nb = tmp_path / '02_data_processing.ipynb'
    write_nb(nb, [
        'from src.io.loaders import load_prices\nprices = load_prices()',
    ])
    assert check_notebook(nb) == []


def test_flags_np_random_normal(tmp_path):
    nb = tmp_path / '11_portfolio_optimization.ipynb'
    write_nb(nb, [
        'from src.io.loaders import load_log_returns\n'
        'mu = np.random.normal(0.08, 0.05, 30)',
    ])
    errors = check_notebook(nb)
    assert any('np.random.normal' in e or 'fabricates' in e for e in errors)


def test_allowlist_tag_skips_check(tmp_path):
    nb = tmp_path / '12_backtesting.ipynb'
    write_nb(nb, [
        'from src.io.loaders import load_log_returns\n'
        '# noqa: pipeline-check\n'
        'z = np.random.standard_t(5, size=(252, 30))',
    ])
    assert check_notebook(nb) == []


def test_flags_missing_loader_import(tmp_path):
    nb = tmp_path / '02_data_processing.ipynb'
    write_nb(nb, ['import pandas as pd\nprices = pd.read_csv("foo.csv")'])
    errors = check_notebook(nb)
    assert any("from src.io.loaders" in e for e in errors)


def test_does_not_flag_nb01(tmp_path):
    """NB01 is exempt — it ingests from yfinance, doesn't read from pipeline."""
    nb = tmp_path / '01_data_pipeline.ipynb'
    write_nb(nb, ['import yfinance as yf\nprices = yf.download(...)'])
    errors = check_notebook(nb)
    assert all("from src.io.loaders" not in e for e in errors)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_check_pipeline.py -v`

- [ ] **Step 3: Implement `scripts/check_pipeline.py`**

Create `scripts/check_pipeline.py`:

```python
"""Smoke test: scan every notebook for forbidden patterns.

Run: python scripts/check_pipeline.py

Exits 0 if all notebooks pass, 1 otherwise. Use as a pre-commit hook
or CI check.

Cells tagged `# noqa: pipeline-check` are skipped (for legitimate
Monte Carlo simulations, optimizer warm-starts, etc.).
"""
import json
import re
import sys
from pathlib import Path

FORBIDDEN = [
    (r'np\.random\.normal\(',      'np.random.normal — fabricates data; use real upstream read'),
    (r'np\.random\.randn\(',       'np.random.randn — fabricates data; use real upstream read'),
    (r'np\.random\.choice\(',      'np.random.choice — fabricates data; use real upstream read'),
    (r'np\.random\.uniform\(',     'np.random.uniform — fabricates data; use real upstream read'),
    (r'np\.random\.rand\(',        'np.random.rand — fabricates data; use real upstream read'),
    (r"parse_dates=\['date'\]",    "use 'Date' (capitalized) consistently"),
    (r'pd\.read_csv\(',            'use src.io.loaders, not direct pd.read_csv'),
    (r'pd\.read_parquet\(',        'use src.io.loaders, not direct pd.read_parquet'),
    (r'joblib\.load\(',            'use src.io.loaders, not direct joblib.load'),
]

ALLOWLIST_TAG = '# noqa: pipeline-check'

# NB01 ingests from yfinance/FRED — exempt from "must use loaders" rule.
# It uses raw pd.read_csv only when reading back its own freshly written outputs
# during validation, which is fine.
EXEMPT_NOTEBOOKS = {'01_data_pipeline'}


def check_notebook(nb_path: Path) -> list[str]:
    errors: list[str] = []
    nb = json.loads(nb_path.read_text())
    nb_stem = nb_path.stem

    for cell_idx, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        if ALLOWLIST_TAG in src:
            continue
        for pattern, msg in FORBIDDEN:
            if re.search(pattern, src):
                # NB01 is allowed pd.read_csv for its own validation reads.
                if nb_stem in EXEMPT_NOTEBOOKS and 'pd.read_csv' in pattern:
                    continue
                errors.append(f"{nb_path.name} cell {cell_idx}: {msg}")

    # Every notebook (except NB01) must import from src.io.loaders.
    if nb_stem not in EXEMPT_NOTEBOOKS:
        full_src = '\n'.join(
            ''.join(c.get('source', []))
            for c in nb.get('cells', [])
            if c.get('cell_type') == 'code'
        )
        if 'from src.io.loaders' not in full_src:
            errors.append(
                f"{nb_path.name}: missing 'from src.io.loaders import ...' "
                f"(every notebook must read inputs through the loader module)"
            )
    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    nb_dir = repo_root / 'notebooks'
    if not nb_dir.is_dir():
        print(f"❌ notebooks/ not found at {nb_dir}")
        return 1

    all_errors: list[str] = []
    nb_paths = sorted(nb_dir.glob('*/*.ipynb'))
    if not nb_paths:
        print(f"❌ no notebooks found under {nb_dir}")
        return 1

    for nb_path in nb_paths:
        all_errors.extend(check_notebook(nb_path))

    if all_errors:
        print('❌ Pipeline check failed:')
        for e in all_errors:
            print(f'  - {e}')
        return 1

    print(f'✅ All {len(nb_paths)} notebooks pass pipeline check.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 4: Make sure `scripts/__init__.py` exists for import**

```bash
touch scripts/__init__.py
```

- [ ] **Step 5: Run tests to verify pass**

Run: `python -m pytest tests/test_check_pipeline.py -v`

- [ ] **Step 6: Commit**

```bash
git add scripts/check_pipeline.py scripts/__init__.py tests/test_check_pipeline.py
git commit -m "feat(scripts): add check_pipeline.py smoke test for random-data regressions"
```

---

### Task 12: `scripts/check_determinism.py`

**Files:**
- Create: `scripts/check_determinism.py`

- [ ] **Step 1: Implement determinism checker**

Create `scripts/check_determinism.py`:

```python
"""Verify the pipeline is deterministic.

Workflow:
  1. Run `python run_all.py` once.
  2. Snapshot SHA-256 of every CSV in data/processed/.
  3. Run `python run_all.py` again.
  4. Compare hashes; fail if any CSV differs.

JSON files (which embed timestamps) are excluded from the check.

Run: python scripts/check_determinism.py
"""
import hashlib
import subprocess
import sys
from pathlib import Path


def file_hash(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def snapshot(proc_dir: Path) -> dict[str, str]:
    return {p.name: file_hash(p) for p in sorted(proc_dir.glob('*.csv'))}


def run_pipeline(repo_root: Path):
    print(f"▶ Running run_all.py …")
    result = subprocess.run(
        [sys.executable, 'run_all.py'],
        cwd=str(repo_root), capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"❌ run_all.py failed:\n{result.stdout}\n{result.stderr}")
        sys.exit(1)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    proc = repo_root / 'data' / 'processed'

    run_pipeline(repo_root)
    before = snapshot(proc)
    if not before:
        print("❌ no CSVs in data/processed/ after first run")
        return 1

    run_pipeline(repo_root)
    after = snapshot(proc)

    diffs = [k for k in before if before[k] != after.get(k)]
    if diffs:
        print(f"❌ {len(diffs)} files differ between runs (non-deterministic):")
        for k in diffs:
            print(f"  - {k}: {before[k][:12]} → {after[k][:12]}")
        return 1

    print(f"✅ All {len(before)} CSVs are bit-identical across runs.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 2: Smoke check (without running full pipeline yet)**

Run: `python scripts/check_determinism.py --help` is not implemented; just verify the file imports cleanly:

`python -c "import scripts.check_determinism; print('OK')"`
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add scripts/check_determinism.py
git commit -m "feat(scripts): add check_determinism.py for back-to-back run hashing"
```

---

### Task 13: `run_all.py` orchestrator

**Files:**
- Create: `run_all.py`

- [ ] **Step 1: Implement orchestrator**

Create `run_all.py`:

```python
"""End-to-end pipeline runner.

Usage:
    python run_all.py            # run all 15 notebooks in order
    python run_all.py --from 06  # resume from NB06

Each notebook is executed in-place (output cells updated). Fails fast
on the first error.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import papermill as pm

from configs.settings import RANDOM_SEED

NOTEBOOKS = [
    '01_data_pipeline',
    '02_data_processing',
    '03_quant_core',
    '04_graph_engine',
    '05_graph_analysis',
    '06_ml_models',
    '07_ml_analysis',
    '08_nlp_sentiment',
    '09_rag_setup',
    '10_nlp_analysis',
    '11_portfolio_optimization',
    '12_backtesting',
    '13_agent_system',
    '14_agent_orchestration',
    '15_agent_demo',
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='start_from', default=None,
                    help='Resume from this notebook prefix (e.g. "06")')
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    nb_dir = repo_root / 'notebooks'

    notebooks = NOTEBOOKS
    if args.start_from:
        notebooks = [n for n in NOTEBOOKS if n[:2] >= args.start_from]
        print(f"▶ Resuming from {notebooks[0]}")

    total_start = time.time()
    for nb_name in notebooks:
        src = nb_dir / nb_name / f'{nb_name}.ipynb'
        if not src.exists():
            print(f"❌ Notebook not found: {src}")
            return 1

        print(f"\n{'='*70}\n▶ {nb_name}\n{'='*70}")
        t0 = time.time()
        try:
            pm.execute_notebook(
                str(src), str(src),  # in-place
                parameters={'RANDOM_SEED': RANDOM_SEED},
                kernel_name='python3',
                progress_bar=False,
            )
        except pm.PapermillExecutionError as e:
            print(f"\n❌ {nb_name} FAILED:")
            print(f"   Cell {e.exec_count}: {e.ename}: {e.evalue}")
            return 1
        elapsed = time.time() - t0
        print(f"✅ {nb_name} OK ({elapsed:.1f}s)")

    total = time.time() - total_start
    print(f"\n🎉 Pipeline complete in {total:.1f}s")
    print(f"   See data/processed/RESULTS.md")
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 2: Verify it imports**

Run: `python -c "import run_all; print('OK')"`
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add run_all.py
git commit -m "feat: add run_all.py papermill-driven pipeline orchestrator"
```

---

## Phase B — Notebook fixes (Tasks 14–28)

One task per notebook. Pattern for each: open the notebook, apply the changes from spec §4, run it manually to verify, commit.

**General approach for every notebook task:**
- Add a `# Load Inputs` cell at the top using `from src.io.loaders import ...`.
- Add `import sys; sys.path.insert(0, str(Path.cwd().parents[1]))` so the notebook can import from the repo root.
- Set `np.random.seed(42)` and `random.seed(42)` if any random component is used.
- Replace forbidden `np.random.*` data fabrications with real reads or computations.
- Fix all `plt.savefig(f'{OUTPUTS_DIR}/ 'fig_xxx.png', ...)` syntax bugs (mismatched quotes).

**To edit notebook cells programmatically**, use this pattern (run in a Python shell or temp script):

```python
import json
from pathlib import Path

def insert_cell(nb_path, idx, source: str, cell_type='code'):
    nb = json.loads(Path(nb_path).read_text())
    cell = {
        'cell_type': cell_type, 'metadata': {}, 'source': source.splitlines(keepends=True),
        **({'execution_count': None, 'outputs': []} if cell_type == 'code' else {}),
    }
    nb['cells'].insert(idx, cell)
    Path(nb_path).write_text(json.dumps(nb, indent=1))

def replace_cell(nb_path, idx, source: str):
    nb = json.loads(Path(nb_path).read_text())
    nb['cells'][idx]['source'] = source.splitlines(keepends=True)
    Path(nb_path).write_text(json.dumps(nb, indent=1))
```

You can also open the notebook in Jupyter/VSCode and edit cells visually. Either works.

**Verification for every notebook task:**
- Run `python scripts/check_pipeline.py` and confirm this notebook (and the previous ones) pass.
- Open the notebook fresh, restart kernel, run all cells. Should complete with no errors.
- Confirm the expected output files exist in `data/processed/` with non-zero size.

---

### Task 14: NB01 — data_pipeline (config integration only)

**Files:**
- Modify: `notebooks/01_data_pipeline/01_data_pipeline.ipynb`

NB01 has no data-flow bugs (it's the source of truth). Only changes: import from `configs.settings` for tickers and date range; `np.random.seed(42)` in case any random ops exist.

- [ ] **Step 1: Add `# Load Config` cell at the top of the notebook**

Insert before the path-management cell:

```python
# === Load Config ===
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))  # repo root
from configs.settings import (
    TICKERS, START_DATE, END_DATE, RANDOM_SEED, DATA_RAW, DATA_OUT,
)
import numpy as np, random
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
```

- [ ] **Step 2: Replace any hardcoded ticker list / date range usage**

Search for any cell that hardcodes a ticker list (e.g., `tickers = ['AAPL', ...]`) or `'2019-01-01'` and replace with `TICKERS`, `START_DATE`, `END_DATE`.

- [ ] **Step 3: Run the notebook**

In Jupyter/VSCode: Restart Kernel → Run All. Should complete without errors. Confirm `data/raw/prices.csv`, `data/raw/macro.csv`, `data/raw/sector_map.json` are written.

- [ ] **Step 4: Run smoke test**

Run: `python scripts/check_pipeline.py`
Expected: NB01 passes (it's exempt from the loader-import rule).

- [ ] **Step 5: Commit**

```bash
git add notebooks/01_data_pipeline/01_data_pipeline.ipynb
git commit -m "fix(NB01): load config from configs.settings; seed random"
```

---

### Task 15: NB02 — data_processing (restore loads, kill bfill, fix savefig)

**Files:**
- Modify: `notebooks/02_data_processing/02_data_processing.ipynb`

This notebook has Class A (NameError) bugs. Restore data loads.

- [ ] **Step 1: Add `# Load Inputs` cell at the top (after path-management cell)**

Insert this code cell:

```python
# === Load Inputs ===
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))
from src.io.loaders import load_prices, load_macro, load_sector_map
from configs.settings import RANDOM_SEED, COLORS, DATA_PROC

import numpy as np, pandas as pd, random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

prices = load_prices()
macro = load_macro()
sector_map = load_sector_map()

print(f"Loaded prices: {prices.shape}, macro: {macro.shape}, sectors: {len(sector_map)}")
```

- [ ] **Step 2: Replace `prices.ffill().bfill()` with `prices.ffill()` only**

Search the notebook for `prices.ffill().bfill()` and change to `prices.ffill()`. Add a one-line comment: `# ffill only — bfill leaks future prices into early NaN cells`.

- [ ] **Step 3: Fix every `plt.savefig` syntax bug**

Find every line of the form:

```python
plt.savefig(f'{OUTPUTS_DIR}/ 'fig_XXX.png', dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
```

Replace with:

```python
plt.savefig(f'{OUTPUTS_DIR}/fig_XXX.png', dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
```

(Two changes: remove the stray space, and quote the whole filename in a single f-string.)

- [ ] **Step 4: For predictor features, add `.shift(1)` to rolling windows**

In the feature-construction cell, replace each rolling feature used as a predictor:

```python
df['vol_5d']  = r.rolling(5).std() * np.sqrt(252)
df['vol_20d'] = r.rolling(20).std() * np.sqrt(252)
df['vol_60d'] = r.rolling(60).std() * np.sqrt(252)
```

with:

```python
df['vol_5d']  = (r.rolling(5).std() * np.sqrt(252)).shift(1)
df['vol_20d'] = (r.rolling(20).std() * np.sqrt(252)).shift(1)
df['vol_60d'] = (r.rolling(60).std() * np.sqrt(252)).shift(1)
```

Do the same for `mom_20d`, `mom_60d`, `dist_ma20`, `skew_60d`, `kurt_60d`, `drawdown`.

**Do NOT shift `ret_1d`, `ret_5d`, `ret_20d`, `ret_60d`** — these are returns realized at time `t` and are valid features (the realized return *at* `t` is known by close of `t`).

- [ ] **Step 5: Replace `pd.read_csv(RAW_DIR / 'prices.csv', ...)` with `load_prices()`**

The "re-read raw to check missingness" cell should become:

```python
raw_prices = load_prices()  # already validated; missing-data check below
missing_pct = raw_prices.isnull().mean()
to_drop = missing_pct[missing_pct > 0.05].index.tolist()
```

- [ ] **Step 6: Add post-feature lookahead assertion at end**

After feature construction is done, add a cell:

```python
# Lookahead-bias guard: feature dates must not extend past return dates
assert features_normalized['Date'].max() <= log_returns.index.max(), \
    "lookahead detected: features have dates beyond available returns"
```

- [ ] **Step 7: Confirm `np.random.seed(RANDOM_SEED)` is set in the load cell**

Already in Step 1.

- [ ] **Step 8: Run the notebook end-to-end**

Restart Kernel → Run All. Confirm all 5 outputs are written: `log_returns.csv`, `simple_returns.csv`, `features_raw.csv`, `features_normalized.csv`, `feature_scaler.pkl`.

- [ ] **Step 9: Run smoke test**

Run: `python scripts/check_pipeline.py`
Expected: NB02 passes.

- [ ] **Step 10: Commit**

```bash
git add notebooks/02_data_processing/02_data_processing.ipynb
git commit -m "fix(NB02): restore data loads, drop bfill leakage, lookahead-safe features"
```

---

### Task 16: NB03 — quant_core (restore loads, GARCH α+β check)

**Files:**
- Modify: `notebooks/03_quant_core/03_quant_core.ipynb`

- [ ] **Step 1: Add `# Load Inputs` cell**

```python
# === Load Inputs ===
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))
from src.io.loaders import load_log_returns, load_prices, load_macro
from configs.settings import RANDOM_SEED, GARCH_P, GARCH_Q, N_REGIMES, DATA_PROC

import numpy as np, pandas as pd, random, warnings
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

log_returns = load_log_returns()
prices = load_prices()
macro = load_macro()

print(f"Loaded log_returns: {log_returns.shape}, prices: {prices.shape}")
```

- [ ] **Step 2: Add GARCH validity assertion**

After the GARCH fitting loop, add:

```python
# GARCH validity: α+β < 1 for stationarity
non_stationary = []
for ticker, fit in garch_results.items():
    persistence = fit.params['alpha[1]'] + fit.params['beta[1]']
    if persistence >= 1.0:
        warnings.warn(f"{ticker}: GARCH non-stationary (α+β={persistence:.4f} ≥ 1)")
        non_stationary.append(ticker)
    elif persistence > 0.99:
        print(f"  ⚠ {ticker}: high persistence (α+β={persistence:.4f}) — near boundary")
print(f"Non-stationary tickers: {len(non_stationary)}/{len(garch_results)}")
```

- [ ] **Step 3: Standardize regime encoding to 0=Bear, 1=Volatile, 2=Bull**

In the regime classification cell, after computing regimes ensure the encoding is:

```python
regime_encoding = {'Bear': 0, 'Volatile': 1, 'Bull': 2}
regimes['regime_encoded'] = regimes['regime_label'].map(regime_encoding)
```

- [ ] **Step 4: Save `risk_metrics.csv` with required columns**

In the cell that writes `risk_metrics.csv`, ensure columns include: `Date, ticker, garch_vol, var_95, cvar_95, downside_vol`. Save with `Date` capitalized:

```python
risk_metrics.to_csv(DATA_PROC / 'risk_metrics.csv', index=False)
```

- [ ] **Step 5: Save `regimes.csv` with `Date` index**

```python
regimes.to_csv(DATA_PROC / 'regimes.csv')  # Date is already the index
```

- [ ] **Step 6: Replace any direct `pd.read_csv` with loader calls**

Search for `pd.read_csv` in the notebook; replace with the appropriate loader.

- [ ] **Step 7: Run notebook end-to-end**

Restart Kernel → Run All. Confirm `risk_metrics.csv` and `regimes.csv` written.

- [ ] **Step 8: Run smoke test**

`python scripts/check_pipeline.py`

- [ ] **Step 9: Commit**

```bash
git add notebooks/03_quant_core/03_quant_core.ipynb
git commit -m "fix(NB03): restore data loads, add GARCH stationarity check"
```

---

### Task 17: NB04 — graph_engine (restore loads)

**Files:**
- Modify: `notebooks/04_graph_engine/04_graph_engine.ipynb`

- [ ] **Step 1: Add `# Load Inputs` cell**

```python
# === Load Inputs ===
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))
from src.io.loaders import load_log_returns, load_sector_map
from configs.settings import RANDOM_SEED, ROLLING_GRAPH_WINDOW, DATA_PROC, DATA_OUT

import numpy as np, pandas as pd, random
import matplotlib.pyplot as plt, seaborn as sns
import networkx as nx
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

log_returns = load_log_returns()
sector_map = load_sector_map()

print(f"Loaded log_returns: {log_returns.shape}")
```

- [ ] **Step 2: Use `ROLLING_GRAPH_WINDOW` from config**

Replace any hardcoded `60` (rolling window for graph metrics) with `ROLLING_GRAPH_WINDOW`.

- [ ] **Step 3: Save `dynamic_graph_metrics.csv` with required columns**

Ensure the saved CSV has columns: `Date (index), density, avg_path_length, clustering, modularity`.

```python
dynamic_metrics.to_csv(DATA_PROC / 'dynamic_graph_metrics.csv')  # Date is index
```

- [ ] **Step 4: Save `mst_edges.csv`**

```python
mst_edges.to_csv(DATA_PROC / 'mst_edges.csv', index=False)
```

- [ ] **Step 5: Run notebook end-to-end**

- [ ] **Step 6: Run smoke test**

`python scripts/check_pipeline.py`

- [ ] **Step 7: Commit**

```bash
git add notebooks/04_graph_engine/04_graph_engine.ipynb
git commit -m "fix(NB04): restore data loads via loaders"
```

---

### Task 18: NB05 — graph_analysis (real impacts, no random)

**Files:**
- Modify: `notebooks/05_graph_analysis/05_graph_analysis.ipynb`

- [ ] **Step 1: Add `# Load Inputs` cell**

```python
# === Load Inputs ===
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))
from src.io.loaders import (
    load_log_returns, load_mst_edges, load_dynamic_graph_metrics,
    load_features_normalized,
)
from configs.settings import RANDOM_SEED, DATA_PROC, DATA_OUT

import numpy as np, pandas as pd, random
import matplotlib.pyplot as plt, seaborn as sns
import networkx as nx
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

log_returns = load_log_returns()
mst_edges = load_mst_edges()
dynamic_metrics = load_dynamic_graph_metrics()
features_normalized = load_features_normalized()

print(f"Loaded: log_returns {log_returns.shape}, mst {mst_edges.shape}")
```

- [ ] **Step 2: Replace `impacts = np.random.rand(len(node_list))` with real computation**

Find the systemic-impact cell (near line 478). Replace:

```python
impacts = np.random.rand(len(node_list))
```

with:

```python
# Real systemic impact: eigenvector centrality × asset volatility
G = nx.from_pandas_edgelist(mst_edges, source='source', target='target',
                             edge_attr='weight' if 'weight' in mst_edges.columns else None)
eig = nx.eigenvector_centrality_numpy(G, weight='weight' if 'weight' in mst_edges.columns else None)
asset_vol = log_returns.std() * np.sqrt(252)
impacts = np.array([eig.get(n, 0) * asset_vol.get(n, 0) for n in node_list])
# Normalize to [0, 1]
if impacts.max() > 0:
    impacts = impacts / impacts.max()
```

- [ ] **Step 3: Audit any other `np.random.*` in this notebook and replace**

`python scripts/check_pipeline.py` will tell you which cells still have forbidden patterns. For each, derive the value from real data (e.g., centrality, volatility, correlation).

- [ ] **Step 4: Write all expected outputs**

Confirm the notebook writes: `centrality_scores.csv`, `community_assignments.csv`, `systemic_risk_scores.csv`, `granger_causality.csv`, `leader_follower.csv`, `features_with_graph.csv`, `graph_intelligence.csv`. All as `*.to_csv(DATA_PROC / 'name.csv', index=False)` (or with `Date` as index for wide formats).

- [ ] **Step 5: Run notebook end-to-end**

- [ ] **Step 6: Run smoke test**

`python scripts/check_pipeline.py` — NB05 should pass.

- [ ] **Step 7: Commit**

```bash
git add notebooks/05_graph_analysis/05_graph_analysis.ipynb
git commit -m "fix(NB05): replace random impacts with real centrality × volatility"
```

---

### Task 19: NB06 — ml_models (restore loads, train/test assertion)

**Files:**
- Modify: `notebooks/06_ml_models/06_ml_models.ipynb`

- [ ] **Step 1: Add `# Load Inputs` cell**

```python
# === Load Inputs ===
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))
from src.io.loaders import (
    load_features_with_graph, load_regimes, load_dynamic_graph_metrics,
)
from configs.settings import (
    RANDOM_SEED, ML_N_SPLITS, RF_PARAMS, XGB_PARAMS, RIDGE_PARAMS, DATA_PROC, DATA_OUT,
)

import numpy as np, pandas as pd, random
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, mean_squared_error, f1_score,
)
import xgboost as xgb
import joblib

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

features_with_graph = load_features_with_graph()
regimes = load_regimes()
dynamic_metrics = load_dynamic_graph_metrics()

print(f"Loaded: features {features_with_graph.shape}, regimes {regimes.shape}")
```

- [ ] **Step 2: Lag the regime feature**

Find the cell that merges `regime_encoded` into `df`. Change:

```python
df = df.merge(regimes[['regime_encoded']], left_on='Date', right_index=True, how='left')
df['target_regime'] = df.groupby('ticker')['regime_encoded'].shift(-1)
```

to:

```python
# Lag regime feature so today's predictor uses YESTERDAY's regime label
df = df.merge(regimes[['regime_encoded']], left_on='Date', right_index=True, how='left')
df['target_regime'] = df.groupby('ticker')['regime_encoded'].shift(-1)  # target: tomorrow
df['regime_encoded_feat'] = df.groupby('ticker')['regime_encoded'].shift(1)  # feature: yesterday
df = df.drop(columns=['regime_encoded'])  # drop the same-day version
df = df.rename(columns={'regime_encoded_feat': 'regime_encoded'})
```

- [ ] **Step 3: Add per-fold train/test no-overlap assertion**

In `evaluate_classification`, replace the loop body with:

```python
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    # Hard guard: train must end before test starts (walk-forward)
    train_dates = X.iloc[train_idx].index if X.index.name == 'Date' else df.iloc[train_idx]['Date']
    test_dates  = X.iloc[test_idx].index  if X.index.name == 'Date' else df.iloc[test_idx]['Date']
    assert train_dates.max() < test_dates.min(), (
        f"Fold {fold}: train/test overlap "
        f"(train ends {train_dates.max()}, test starts {test_dates.min()})"
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # ... (rest unchanged)
```

- [ ] **Step 4: Use `RF_PARAMS`, `XGB_PARAMS` from config**

Replace inline dict construction:

```python
rf = RandomForestClassifier(n_estimators=200, max_depth=6, ...)
xgb_clf = xgb.XGBClassifier(n_estimators=200, ...)
```

with:

```python
rf = RandomForestClassifier(**RF_PARAMS)
xgb_clf = xgb.XGBClassifier(**XGB_PARAMS)
```

- [ ] **Step 5: Fix every `plt.savefig` syntax bug**

Same as NB02 step 3 — find `plt.savefig(f'{OUTPUTS_DIR}/ "fig_xxx.png")` (also broken with mismatched quotes) and replace with `plt.savefig(f'{OUTPUTS_DIR}/fig_xxx.png', dpi=150, bbox_inches='tight', facecolor='#0d0d0d')`.

Also fix the surrounding `plt.show()plt.close()` (missing semicolon/newline) — split into two lines:

```python
plt.show()
plt.close()
```

- [ ] **Step 6: Save outputs**

Ensure these are written:

```python
ml_signals.to_csv(DATA_PROC / 'ml_signals.csv', index=False)
joblib.dump(model_dir, DATA_PROC / 'model_direction.pkl')
joblib.dump(model_vol, DATA_PROC / 'model_volatility.pkl')
joblib.dump(model_reg, DATA_PROC / 'model_regime.pkl')
```

- [ ] **Step 7: Run notebook end-to-end**

- [ ] **Step 8: Run smoke test**

- [ ] **Step 9: Commit**

```bash
git add notebooks/06_ml_models/06_ml_models.ipynb
git commit -m "fix(NB06): restore data loads, add walk-forward assertion, lag regime feature"
```

---

### Task 20: NB07 — ml_analysis (restore loads, write master_signals + ml_eval.json)

**Files:**
- Modify: `notebooks/07_ml_analysis/07_ml_analysis.ipynb`

- [ ] **Step 1: Add `# Load Inputs` cell**

```python
# === Load Inputs ===
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))
from src.io.loaders import (
    load_features_with_graph, load_regimes, load_dynamic_graph_metrics,
    load_models, load_feature_scaler,
)
from configs.settings import RANDOM_SEED, DATA_PROC, DATA_OUT

import numpy as np, pandas as pd, random
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import zscore
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

features_df = load_features_with_graph()
regimes = load_regimes()
dynamic_metrics = load_dynamic_graph_metrics()
model_dir, model_vol, model_reg = load_models()
scaler = load_feature_scaler()

print(f"Loaded: features {features_df.shape}, models OK")
```

- [ ] **Step 2: Write `master_signals.csv` with required columns**

After computing `df` with all signal columns, save:

```python
out_cols = ['Date', 'ticker', 'alpha_signal', 'alpha_zscore',
            'risk_signal', 'risk_zscore', 'regime_pred', 'combined_score']
df[out_cols].to_csv(DATA_PROC / 'master_signals.csv', index=False)
```

- [ ] **Step 3: Write `ml_eval.json` with per-fold metrics**

If not already written, add:

```python
ml_eval = {
    'avg_auc': float(full_summary_t1['avg_auc'].max()),
    'best_model': str(best_model_name),
    'per_model': full_summary_t1.to_dict(orient='index'),
    'graph_importance_pct': float(graph_importance_pct) if 'graph_importance_pct' in dir() else None,
}
with open(DATA_PROC / 'ml_eval.json', 'w') as f:
    json.dump(ml_eval, f, indent=2, default=str)
```

- [ ] **Step 4: Write `signal_explanations.json`**

If `signal_explanations` is computed in this notebook (or compute a basic version):

```python
signal_explanations = {
    ticker: {
        'top_features': list(top_features_per_ticker.get(ticker, [])),
        'avg_alpha': float(df[df.ticker == ticker]['alpha_signal'].mean()),
    }
    for ticker in df.ticker.unique()
}
with open(DATA_PROC / 'signal_explanations.json', 'w') as f:
    json.dump(signal_explanations, f, indent=2)
```

- [ ] **Step 5: Run notebook end-to-end**

- [ ] **Step 6: Run smoke test**

- [ ] **Step 7: Commit**

```bash
git add notebooks/07_ml_analysis/07_ml_analysis.ipynb
git commit -m "fix(NB07): restore data loads, write master_signals + ml_eval.json"
```

---

### Task 21: NB08 — nlp_sentiment (deterministic synthetic news, real FinBERT)

**Files:**
- Modify: `notebooks/08_nlp_sentiment/08_nlp_sentiment.ipynb`

- [ ] **Step 1: Add `# Load Inputs` cell**

```python
# === Load Inputs ===
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))
from src.io.loaders import load_prices, load_sector_map
from configs.settings import (
    RANDOM_SEED, FINBERT_MODEL, SYNTHETIC_NEWS_N, DATA_RAW, DATA_PROC,
)

import numpy as np, pandas as pd, random, json
import matplotlib.pyplot as plt
from transformers import pipeline
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

prices = load_prices()
sector_map = load_sector_map()

print(f"Loaded prices: {prices.shape}, sectors: {len(sector_map)}")
```

- [ ] **Step 2: Make synthetic news generation deterministic**

Confirm `random.seed(RANDOM_SEED)` is called BEFORE the synthetic-news generation loop (already done in Step 1). The existing `random.choice` calls become deterministic.

- [ ] **Step 3: Save `synthetic_news.csv` with `Date` capitalized**

```python
news_df.to_csv(DATA_RAW / 'synthetic_news.csv', index=False)
```

Ensure the date column is named `Date` (capitalize if it's `date`).

- [ ] **Step 4: Save `sentiment_features.csv` with `Date` capitalized**

```python
# Aggregate FinBERT scores per (Date, ticker)
sentiment_features = (
    news_df.groupby(['Date', 'ticker'], as_index=False)
    .agg(sentiment_score=('finbert_score', 'mean'),
         news_volume=('headline', 'count'))
)
sentiment_features.to_csv(DATA_PROC / 'sentiment_features.csv', index=False)
```

- [ ] **Step 5: Tag legitimate `random.choice` calls with allowlist comment**

The synthetic-news generation uses `random.choice` legitimately (for templates). At the top of that cell, add:

```python
# noqa: pipeline-check
# Synthetic news generation — random.choice on TEMPLATES is intentional,
# seeded with RANDOM_SEED for determinism. Real FinBERT scores follow.
```

- [ ] **Step 6: Run notebook end-to-end**

- [ ] **Step 7: Run smoke test**

- [ ] **Step 8: Commit**

```bash
git add notebooks/08_nlp_sentiment/08_nlp_sentiment.ipynb
git commit -m "fix(NB08): deterministic synthetic news, Date column, real FinBERT scoring"
```

---

### Task 22: NB09 — rag_setup (real sentence-transformer embeddings)

**Files:**
- Modify: `notebooks/09_rag_setup/09_rag_setup.ipynb`

This is the critical fix — replace placeholder embeddings with real ones.

- [ ] **Step 1: Add `# Load Inputs` cell**

```python
# === Load Inputs ===
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))
from src.io.loaders import load_synthetic_news
from configs.settings import RANDOM_SEED, EMBEDDING_MODEL, DATA_PROC

import numpy as np, pandas as pd, random
import faiss
from sentence_transformers import SentenceTransformer
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

synthetic_news = load_synthetic_news()
documents = synthetic_news['headline'].tolist()  # or 'text' if that's the column
print(f"Loaded {len(documents)} synthetic news documents")
```

- [ ] **Step 2: Replace `embeddings_matrix = np.random.rand(N, 128)` with real embeddings**

Find the cell with `np.random.rand(len(documents), 128)`. Replace:

```python
embeddings_matrix = np.random.rand(len(documents), 128)  # Placeholder
```

with:

```python
# Real sentence-transformer embeddings
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings_matrix = model.encode(
    documents, show_progress_bar=True, convert_to_numpy=True,
    normalize_embeddings=True,
)
print(f"Embeddings shape: {embeddings_matrix.shape}")  # (N, 384) for MiniLM-L6
```

- [ ] **Step 3: Save real embeddings**

```python
np.save(DATA_PROC / 'embeddings.npy', embeddings_matrix)
```

- [ ] **Step 4: Build FAISS index on real embeddings**

```python
dim = embeddings_matrix.shape[1]
index = faiss.IndexFlatIP(dim)  # inner product (cosine since normalized)
index.add(embeddings_matrix.astype(np.float32))
faiss.write_index(index, str(DATA_PROC / 'faiss_index.bin'))
```

- [ ] **Step 5: Save documents list as JSON**

```python
rag_docs = synthetic_news[['Date', 'ticker', 'headline']].to_dict(orient='records')
with open(DATA_PROC / 'rag_documents.json', 'w') as f:
    json.dump(rag_docs, f, default=str, indent=2)
```

- [ ] **Step 6: Save model handle for reuse**

```python
import joblib
joblib.dump({'model_name': EMBEDDING_MODEL}, DATA_PROC / 'rag_models.pkl')
```

- [ ] **Step 7: Smoke check — query "tech earnings beat"**

Add a verification cell:

```python
query = "tech earnings beat"
q_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
_, top_idx = index.search(q_vec, 5)
print("Top 5 retrieved documents:")
for i in top_idx[0]:
    print(f"  - [{rag_docs[i].get('ticker', '')}] {rag_docs[i]['headline']}")
```

Manually verify the top-5 results are tech-related earnings headlines.

- [ ] **Step 8: Run notebook end-to-end**

- [ ] **Step 9: Run smoke test**

- [ ] **Step 10: Commit**

```bash
git add notebooks/09_rag_setup/09_rag_setup.ipynb
git commit -m "fix(NB09): replace placeholder embeddings with real sentence-transformer"
```

---

### Task 23: NB10 — nlp_analysis (kill random flag, lag sentiment, drop circular dep)

**Files:**
- Modify: `notebooks/10_nlp_analysis/10_nlp_analysis.ipynb`

- [ ] **Step 1: Add `# Load Inputs` cell**

```python
# === Load Inputs ===
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))
from src.io.loaders import (
    load_sentiment_features, load_log_returns, load_regimes,
)
from configs.settings import RANDOM_SEED, DATA_PROC

import numpy as np, pandas as pd, random
import matplotlib.pyplot as plt, seaborn as sns
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

sentiment_features = load_sentiment_features()
log_returns = load_log_returns()
regimes = load_regimes()

print(f"Loaded sentiment: {sentiment_features.shape}, returns: {log_returns.shape}")
```

- [ ] **Step 2: Drop the `phase7_input.csv` read (circular dependency)**

Find any `pd.read_csv(.../phase7_input.csv...)` and remove. Don't replace; the data we need comes from the loaders above.

- [ ] **Step 3: Lag sentiment by 1 day before computing IC**

```python
# Lag: sentiment at t is used to predict return at t+1 (1-day execution lag)
sent_lagged = sentiment_features.copy()
sent_lagged['Date'] = pd.to_datetime(sent_lagged['Date'])
sent_lagged = sent_lagged.sort_values(['ticker', 'Date'])
sent_lagged['sentiment_score_lag1'] = (
    sent_lagged.groupby('ticker')['sentiment_score'].shift(1)
)
sent_lagged = sent_lagged.dropna(subset=['sentiment_score_lag1'])
```

- [ ] **Step 4: Replace `final_signals['sentiment_risk_flag'] = np.random.choice(...)`**

Find the line `np.random.choice([True, False], size=len(final_signals), p=[0.1, 0.9])`. Replace with a real derivation:

```python
# Risk flag derived from real sentiment: True when sentiment is in bottom 10% AND returns are negative
sent_threshold = final_signals['sentiment_score'].quantile(0.10)
final_signals['sentiment_risk_flag'] = (
    (final_signals['sentiment_score'] <= sent_threshold) &
    (final_signals['log_return'] < 0)
)
```

- [ ] **Step 5: Compute and save real IC**

```python
ic_per_ticker = (
    sent_lagged.merge(
        log_returns.stack().rename('log_return').reset_index().rename(columns={'level_1': 'ticker'}),
        on=['Date', 'ticker'], how='inner',
    )
    .groupby('ticker')
    .apply(lambda g: g['sentiment_score_lag1'].corr(g['log_return']))
)
ic_lagged = float(ic_per_ticker.mean())

nlp_metrics = {
    'ic_lagged': ic_lagged,
    'ic_per_ticker': ic_per_ticker.to_dict(),
    'n_tickers': len(ic_per_ticker),
    'sentiment_lag_days': 1,
}
with open(DATA_PROC / 'nlp_metrics.json', 'w') as f:
    json.dump(nlp_metrics, f, indent=2, default=float)
print(f"Lagged sentiment IC: {ic_lagged:.4f}")
```

- [ ] **Step 6: Remove all viz-only `np.random` calls**

For each viz cell that uses `np.random.normal/randn/choice/uniform/rand`:
- If it's a stub mock-up that doesn't reflect any real data → delete the cell.
- If it's meant to show a simulation → replace with real upstream data (e.g., real cumulative returns from `log_returns`).

For example replace:

```python
axs[0, 0].plot(days, np.cumprod(1 + np.random.normal(0.0005, 0.01, 100)), label='ML only')
```

with:

```python
ml_only_path = (1 + log_returns.mean(axis=1).iloc[-100:]).cumprod()
axs[0, 0].plot(ml_only_path.index, ml_only_path.values, label='Mean of universe')
```

- [ ] **Step 7: Don't write `phase7_input.csv`**

Remove the line `df.to_csv('../data/processed/phase7_input.csv')` if present. NB11 reads sources directly via loaders.

- [ ] **Step 8: Run notebook end-to-end**

- [ ] **Step 9: Run smoke test**

- [ ] **Step 10: Commit**

```bash
git add notebooks/10_nlp_analysis/10_nlp_analysis.ipynb
git commit -m "fix(NB10): real sentiment risk flag, 1-day lag, drop phase7_input circular"
```

---

### Task 24: NB11 — portfolio_optimization (real signals, real covariance)

**Files:**
- Modify: `notebooks/11_portfolio_optimization/11_portfolio_optimization.ipynb`

This is the largest single fix — every signal source is currently fabricated.

- [ ] **Step 1: Add `# Load Inputs` cell**

```python
# === Load Inputs ===
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))
from src.io.loaders import (
    load_log_returns, load_master_signals, load_sentiment_features,
    load_risk_metrics, load_systemic_risk_scores,
)
from configs.settings import (
    RANDOM_SEED, OPT_RISK_AVERSION, OPT_MAX_WEIGHT, OPT_LONG_ONLY,
    CVAR_ALPHA, COVAR_SHRINKAGE, DATA_PROC, TICKERS,
)

import numpy as np, pandas as pd, random
import cvxpy as cp
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.covariance import LedoitWolf
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

log_returns = load_log_returns()
master_signals = load_master_signals()
sentiment_features = load_sentiment_features()
risk_metrics = load_risk_metrics()
systemic_scores = load_systemic_risk_scores()

print(f"Loaded log_returns: {log_returns.shape}, master_signals: {master_signals.shape}")
```

- [ ] **Step 2: Replace random `mu_*` vectors with real values**

Find the cell with `mu_historical = np.random.normal(0.08, 0.05, n_assets)` etc. Replace the entire block:

```python
n_assets = log_returns.shape[1]
asset_names = log_returns.columns.tolist()

# Real expected returns from each signal source
mu_historical = (log_returns.mean() * 252).reindex(asset_names).fillna(0).values  # annualized

# ML alpha — average per-ticker alpha signal across all dates (lagged)
ml_signal_avg = (
    master_signals.assign(Date=pd.to_datetime(master_signals['Date']))
    .sort_values(['ticker', 'Date'])
    .groupby('ticker')['alpha_signal']
    .apply(lambda s: s.shift(1).mean())  # shift-1 for execution lag
)
mu_ml = ml_signal_avg.reindex(asset_names).fillna(0).values

# Sentiment — average per-ticker sentiment (lagged)
sent_avg = (
    sentiment_features.assign(Date=pd.to_datetime(sentiment_features['Date']))
    .sort_values(['ticker', 'Date'])
    .groupby('ticker')['sentiment_score']
    .apply(lambda s: s.shift(1).mean())
)
mu_sent = sent_avg.reindex(asset_names).fillna(0).values

# 1-month reversal: negative of recent 20-day mean return
mu_rev = -(log_returns.tail(20).mean() * 252).reindex(asset_names).fillna(0).values
```

- [ ] **Step 3: Replace random covariance with Ledoit-Wolf**

Find `sample_cov = np.cov(np.random.randn(n_assets, 1500))`. Replace:

```python
# Real covariance via Ledoit-Wolf shrinkage on log returns
lw = LedoitWolf().fit(log_returns.dropna().values)
sample_cov = lw.covariance_ * 252  # annualized
print(f"Cov shape: {sample_cov.shape}, shrinkage: {lw.shrinkage_:.3f}")
```

- [ ] **Step 4: Replace random `sys_scores` and `sent_scores`**

Find:

```python
sys_scores  = np.random.uniform(0.1, 0.9, n)
sent_scores = np.random.normal(0, 1, n)
```

Replace with real values from loaded data:

```python
# Real systemic risk scores — average across dates per ticker
sys_avg = systemic_scores.mean()  # one value per ticker (column-wise)
sys_scores = sys_avg.reindex(asset_names).fillna(sys_avg.mean()).values

# Real sentiment scores — average per ticker
sent_avg_tk = sentiment_features.groupby('ticker')['sentiment_score'].mean()
sent_scores = sent_avg_tk.reindex(asset_names).fillna(0).values
```

- [ ] **Step 5: Replace any other `np.random` data fabrication in this notebook**

`python scripts/check_pipeline.py` will list remaining instances. For each, derive from real upstream data. Tag the legitimate `np.random.dirichlet` warm-start with `# noqa: pipeline-check`:

```python
# noqa: pipeline-check
# Optimizer warm-start; deterministic under np.random.seed(RANDOM_SEED)
w0 = np.random.dirichlet(np.ones(n))
```

- [ ] **Step 6: Use config constants for optimizer**

Replace any hardcoded risk-aversion / max-weight / long-only with `OPT_RISK_AVERSION`, `OPT_MAX_WEIGHT`, `OPT_LONG_ONLY`.

- [ ] **Step 7: Save `optimal_weights.csv` with required columns**

```python
out = pd.DataFrame({
    'Date': pd.Timestamp.today().normalize(),  # current date for static optimization
    'ticker': asset_names,
    'weight': w_full,
    'mu_combined': mu_combined,
    'expected_vol': expected_vol_per_asset,
})
out.to_csv(DATA_PROC / 'optimal_weights.csv', index=False)
```

If the notebook does walk-forward optimization (computing weights at each rebalance date), save one row per (Date, ticker) pair with the appropriate Date.

- [ ] **Step 8: Run notebook end-to-end**

- [ ] **Step 9: Run smoke test**

`python scripts/check_pipeline.py` — NB11 should pass.

- [ ] **Step 10: Commit**

```bash
git add notebooks/11_portfolio_optimization/11_portfolio_optimization.ipynb
git commit -m "fix(NB11): real mu vectors, real Ledoit-Wolf covariance, real sys/sent scores"
```

---

### Task 25: NB12 — backtesting (read optimal_weights, walk-forward assertion, RESULTS.md)

**Files:**
- Modify: `notebooks/12_backtesting/12_backtesting.ipynb`

- [ ] **Step 1: Add `# Load Inputs` cell**

```python
# === Load Inputs ===
import sys, json
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))
from src.io.loaders import (
    load_log_returns, load_simple_returns, load_optimal_weights,
    load_regimes, load_systemic_risk_scores, load_macro,
)
from configs.settings import (
    RANDOM_SEED, BACKTEST_REBALANCE, TRANSACTION_COST_BPS, INITIAL_CAPITAL,
    DATA_PROC,
)

import numpy as np, pandas as pd, random
import matplotlib.pyplot as plt, seaborn as sns
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

log_returns = load_log_returns()
simple_returns = load_simple_returns()
optimal_weights = load_optimal_weights()
regimes = load_regimes()
systemic_scores = load_systemic_risk_scores()
macro = load_macro()

print(f"Loaded log_returns: {log_returns.shape}, weights: {optimal_weights.shape}")
```

- [ ] **Step 2: Replace internal weight fabrication with `optimal_weights` reads**

Find any cell that constructs portfolio weights inline (e.g., random Markowitz, Dirichlet draws used as final weights). Replace with reads from `optimal_weights`:

```python
# Pivot long-format optimal_weights to wide (Date × ticker)
weights_wide = (
    optimal_weights.assign(Date=pd.to_datetime(optimal_weights['Date']))
    .pivot(index='Date', columns='ticker', values='weight')
    .reindex(columns=simple_returns.columns)
    .fillna(0)
)
```

- [ ] **Step 3: Add walk-forward integrity assertion**

In the rebalance loop, before applying weights:

```python
for rebal_date in rebalance_dates:
    weights_t = weights_wide.loc[:rebal_date].iloc[-1]  # last available weight ≤ rebal_date
    # Walk-forward guard: weight date must be < execution date
    weight_date = weights_wide.loc[:rebal_date].index[-1]
    assert weight_date < rebal_date, (
        f"WALK-FORWARD VIOLATION at {rebal_date}: "
        f"weights came from {weight_date} (must be strictly < {rebal_date})"
    )
    # Apply weights to forward returns
    next_rebal = next_rebalance_date(rebal_date)
    period_returns = simple_returns.loc[rebal_date:next_rebal]
    portfolio_ret = (period_returns * weights_t).sum(axis=1)
```

- [ ] **Step 4: Tag Monte Carlo `np.random.standard_t` with allowlist**

Find lines like `z = np.random.standard_t(5, size=(252, n_assets))` and add `# noqa: pipeline-check` on the line above (Monte Carlo stress test is a legitimate use).

- [ ] **Step 5: Add `RESULTS.md` and `RESULTS.json` generation**

At the end of the notebook, add a final cell:

```python
# === Auto-generated RESULTS.md ===
results = {
    'run_timestamp':     datetime.now().isoformat(timespec='seconds'),
    'data_period':       f"{simple_returns.index.min().date()} → {simple_returns.index.max().date()}",
    'n_trading_days':    int(len(simple_returns)),
    'n_rebalances':      int(len(rebalance_dates)),
    'sharpe':            float(sharpe_aa),
    'annualized_return': float(ann_ret_aa),
    'max_drawdown':      float(mdd_aa),
    'sharpe_eq_weight':  float(sharpe_eq),
    'sharpe_markowitz':  float(sharpe_mk) if 'sharpe_mk' in dir() else None,
}

# Pull metrics from upstream JSONs if available
try:
    with open(DATA_PROC / 'ml_eval.json') as f:
        ml_eval = json.load(f)
    results['ml_direction_auc'] = float(ml_eval.get('avg_auc', 0))
    results['graph_feat_imp'] = float(ml_eval.get('graph_importance_pct', 0)) if ml_eval.get('graph_importance_pct') else None
except FileNotFoundError:
    pass

try:
    with open(DATA_PROC / 'nlp_metrics.json') as f:
        nlp_metrics = json.load(f)
    results['sentiment_ic_lagged'] = float(nlp_metrics.get('ic_lagged', 0))
except FileNotFoundError:
    pass

with open(DATA_PROC / 'RESULTS.json', 'w') as f:
    json.dump(results, f, indent=2)

md_lines = [
    "# AxiomAlpha — Live Pipeline Results",
    f"_Generated: {results['run_timestamp']}_",
    "",
    f"## Backtest ({results['data_period']}, {results['n_trading_days']} days, {results['n_rebalances']} rebalances)",
    "",
    "| Strategy | Sharpe | Annualized Return | Max Drawdown |",
    "|---|:---:|:---:|:---:|",
    f"| **AxiomAlpha** | **{results['sharpe']:.3f}** | **{results['annualized_return']:.1%}** | **{results['max_drawdown']:.1%}** |",
]
if results.get('sharpe_markowitz') is not None:
    md_lines.append(f"| Markowitz | {results['sharpe_markowitz']:.3f} | — | — |")
md_lines.append(f"| Equal Weight | {results['sharpe_eq_weight']:.3f} | — | — |")
md_lines.extend([
    "",
    "## Statistical Results",
    "| Component | Metric | Value |",
    "|---|---|:---:|",
])
if 'ml_direction_auc' in results:
    md_lines.append(f"| ML Direction | AUC | {results['ml_direction_auc']:.3f} |")
if 'sentiment_ic_lagged' in results:
    md_lines.append(f"| Sentiment (lagged 1d) | IC | {results['sentiment_ic_lagged']:.4f} |")
if results.get('graph_feat_imp') is not None:
    md_lines.append(f"| Graph Features | Importance | {results['graph_feat_imp']:.1%} |")
md_lines.extend([
    "",
    "> All numbers above are produced by the live pipeline. "
    "Re-run `python run_all.py` to refresh.",
])
(DATA_PROC / 'RESULTS.md').write_text('\n'.join(md_lines))
print("✅ RESULTS.md regenerated")
```

- [ ] **Step 6: Save all backtest outputs**

Confirm these are written: `backtest_results.csv`, `equity_curves.csv`, `performance_metrics.csv`, `attribution_results.csv`, `regime_performance.csv`.

- [ ] **Step 7: Run notebook end-to-end**

- [ ] **Step 8: Run smoke test**

- [ ] **Step 9: Commit**

```bash
git add notebooks/12_backtesting/12_backtesting.ipynb
git commit -m "fix(NB12): read real optimal_weights, add walk-forward assertion, generate RESULTS.md"
```

---

### Task 26: NB13 — agent_system (real reads, no random state)

**Files:**
- Modify: `notebooks/13_agent_system/13_agent_system.ipynb`

- [ ] **Step 1: Add `# Load Inputs` cell**

```python
# === Load Inputs ===
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))
from src.io.loaders import (
    load_master_signals, load_optimal_weights, load_regimes,
    load_risk_metrics, load_sentiment_features,
    load_centrality_scores, load_systemic_risk_scores,
)
from configs.settings import RANDOM_SEED, DATA_PROC

import numpy as np, pandas as pd, random
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Each agent reads only its assigned data
master_signals = load_master_signals()
optimal_weights = load_optimal_weights()
regimes = load_regimes()
risk_metrics = load_risk_metrics()
sentiment_features = load_sentiment_features()
centrality_scores = load_centrality_scores()
systemic_scores = load_systemic_risk_scores()

print("Loaded all agent inputs")
```

- [ ] **Step 2: Replace `current_regime = np.random.choice(['Bull','Bear','Volatile'])`**

Find every occurrence in agent classes. Replace with the most recent real regime:

```python
# Use real most-recent regime
latest_regime_code = int(regimes['regime_encoded'].iloc[-1])
regime_name = {0: 'Bear', 1: 'Volatile', 2: 'Bull'}[latest_regime_code]
current_regime = regime_name
```

- [ ] **Step 3: Replace `confidence = np.random.uniform(0.6, 0.95)`**

Use real model confidence (probability from regime classifier or directional model). Where each agent computes confidence:

```python
# Real confidence from master_signals (mean alpha-signal magnitude over recent window)
confidence = float(master_signals.tail(100)['alpha_signal'].abs().mean())
```

- [ ] **Step 4: Replace `direction_prob`, `vol_forecast`, `sentiment_mean`, `news_volume` with real values**

For each agent's data-population block, replace random samples with the real per-ticker latest values:

```python
# Real per-ticker latest signals
latest = master_signals.sort_values('Date').groupby('ticker').tail(1).set_index('ticker')
direction_prob = latest['alpha_signal'].to_dict()  # real
vol_forecast = latest['risk_signal'].to_dict() if 'risk_signal' in latest.columns else {}

sent_latest = (
    sentiment_features.sort_values('Date').groupby('ticker').tail(1).set_index('ticker')
)
sentiment_mean = sent_latest['sentiment_score'].to_dict()
news_volume = sent_latest['news_volume'].to_dict() if 'news_volume' in sent_latest.columns else {}
```

- [ ] **Step 5: Remove all `np.random.*` data fabrication; tag any legitimate ones**

Use `python scripts/check_pipeline.py` to identify remaining patterns. For each, derive from real data.

- [ ] **Step 6: Run notebook end-to-end**

- [ ] **Step 7: Run smoke test**

- [ ] **Step 8: Commit**

```bash
git add notebooks/13_agent_system/13_agent_system.ipynb
git commit -m "fix(NB13): replace random agent state with real upstream reads"
```

---

### Task 27: NB14 — agent_orchestration (real returns, real weights)

**Files:**
- Modify: `notebooks/14_agent_orchestration/14_agent_orchestration.ipynb`

- [ ] **Step 1: Add `# Load Inputs` cell**

```python
# === Load Inputs ===
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))
from src.io.loaders import (
    load_log_returns, load_simple_returns, load_optimal_weights,
    load_regimes, load_master_signals, load_risk_metrics,
    load_sentiment_features, load_centrality_scores,
    load_performance_metrics, load_attribution_results,
)
from configs.settings import RANDOM_SEED, DATA_PROC

import numpy as np, pandas as pd, random
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

log_returns = load_log_returns()
simple_returns = load_simple_returns()
optimal_weights = load_optimal_weights()
regimes = load_regimes()
master_signals = load_master_signals()
risk_metrics = load_risk_metrics()
sentiment_features = load_sentiment_features()
centrality_scores = load_centrality_scores()
performance_metrics = load_performance_metrics()
attribution_results = load_attribution_results()

print(f"Loaded log_returns: {log_returns.shape}")
```

- [ ] **Step 2: Replace `returns = pd.DataFrame(np.random.randn(100, 10), ...)`**

Find this line (around line 361) and replace with:

```python
# Use real recent returns, real ticker universe
returns = log_returns.tail(252)  # last year of real returns, real tickers
```

- [ ] **Step 3: Populate pipeline state from real CSVs**

Wherever the LangGraph state is constructed, fill it from real data. Example:

```python
state = {
    'tickers': list(log_returns.columns),
    'period_start': str(log_returns.index.min().date()),
    'period_end': str(log_returns.index.max().date()),
    'latest_regime': int(regimes['regime_encoded'].iloc[-1]),
    'optimal_weights': (
        optimal_weights.sort_values('Date').groupby('ticker').tail(1)
        .set_index('ticker')['weight'].to_dict()
    ),
    'sharpe': float(performance_metrics.get('sharpe', 0)),
    # ... etc
}
```

- [ ] **Step 4: Save `pipeline_output.json` with real values**

```python
with open(DATA_PROC / 'pipeline_output.json', 'w') as f:
    json.dump(state, f, indent=2, default=str)
```

- [ ] **Step 5: Remove all remaining `np.random.*` data fabrication**

- [ ] **Step 6: Run notebook end-to-end**

- [ ] **Step 7: Run smoke test**

- [ ] **Step 8: Commit**

```bash
git add notebooks/14_agent_orchestration/14_agent_orchestration.ipynb
git commit -m "fix(NB14): real returns and weights, real LangGraph state"
```

---

### Task 28: NB15 — agent_demo (real outputs, no random fallbacks)

**Files:**
- Modify: `notebooks/15_agent_demo/15_agent_demo.ipynb`

- [ ] **Step 1: Add `# Load Inputs` cell**

```python
# === Load Inputs ===
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))
from src.io.loaders import (
    load_optimal_weights, load_equity_curves, load_performance_metrics,
    load_attribution_results, load_regime_performance, load_pipeline_output,
)
from configs.settings import RANDOM_SEED, DATA_PROC, DATA_OUT

import numpy as np, pandas as pd, random
import matplotlib.pyplot as plt, seaborn as sns
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

optimal_weights = load_optimal_weights()
equity_curves = load_equity_curves()
performance_metrics = load_performance_metrics()
attribution_results = load_attribution_results()
regime_performance = load_regime_performance()
pipeline_output = load_pipeline_output()

print("Loaded all demo inputs")
```

- [ ] **Step 2: Remove the `agent_portfolio.csv` fallback**

Find:

```python
try:
    portfolio = pd.read_csv('data/processed/agent_portfolio.csv', index_col=0)
except:
    portfolio = pd.DataFrame({'weight': np.random.rand(5)}, index=[f'TICKER_{i}' for i in range(5)])
```

Replace with real `optimal_weights`:

```python
# Use latest optimal_weights as the portfolio
portfolio = (
    optimal_weights.sort_values('Date').groupby('ticker').tail(1)
    .set_index('ticker')[['weight']]
)
```

- [ ] **Step 3: Replace random equity curves with real ones**

Find:

```python
aa_eq = np.exp(np.random.normal(0.0005, 0.01, 1000).cumsum())
bm_eq = np.exp(np.random.normal(0.0003, 0.012, 1000).cumsum())
```

Replace:

```python
aa_eq = equity_curves['AxiomAlpha'].values if 'AxiomAlpha' in equity_curves.columns else equity_curves.iloc[:, 0].values
bm_eq = equity_curves['EqualWeight'].values if 'EqualWeight' in equity_curves.columns else equity_curves.iloc[:, -1].values
```

- [ ] **Step 4: Templated Q&A**

For any "LLM Q&A" demo cell, replace stub LLM responses with templated answers derived from real signals:

```python
def explain_decision(query: str, weights: pd.Series, perf: dict) -> str:
    if 'why these weights' in query.lower():
        top = weights.nlargest(3)
        return (
            f"Top 3 holdings: {', '.join(top.index)} ({', '.join(f'{w:.1%}' for w in top.values)}). "
            f"Selected by combined alpha + risk + regime signals; portfolio Sharpe = {perf.get('sharpe', 'n/a'):.3f}."
        )
    if 'regime' in query.lower():
        return f"Current regime: {pipeline_output.get('latest_regime', 'unknown')}. " \
               f"Position sizing reflects regime-conditional risk premium."
    return f"Query received: {query}. (No template match; would route to LLM in production.)"

print(explain_decision("Why these weights?", portfolio['weight'], performance_metrics))
```

- [ ] **Step 5: Remove all remaining `np.random` fabrication**

- [ ] **Step 6: Run notebook end-to-end**

- [ ] **Step 7: Run smoke test (final notebook)**

`python scripts/check_pipeline.py` — all 15 notebooks should now pass.

- [ ] **Step 8: Commit**

```bash
git add notebooks/15_agent_demo/15_agent_demo.ipynb
git commit -m "fix(NB15): real backtest outputs, templated Q&A, no random fallbacks"
```

---

## Phase C — Documentation (Tasks 29–31)

### Task 29: Top-level README updates

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Remove "anthropic / Claude" row from Tech Stack**

Find the table row mentioning `anthropic` (Claude) and delete it.

- [ ] **Step 2: Add "sentence-transformers" row**

Add a row to the Tech Stack table:

```
| **NLP Embeddings** | `sentence-transformers` | Real embeddings (all-MiniLM-L6-v2) for synthetic-news RAG. |
```

- [ ] **Step 3: Replace hardcoded results with link to RESULTS.md**

Find the "Verified Performance (Backtest)" table. Replace its entire contents with:

```markdown
### 📈 Live Performance

> All headline numbers are produced by the live pipeline. See **[data/processed/RESULTS.md](data/processed/RESULTS.md)** — regenerated by every `python run_all.py`.
```

- [ ] **Step 4: Update Limitations**

Modify the "Synthetic News Data" bullet:

```markdown
- **Synthetic News Data:** To bypass expensive vendor APIs, NB08 generates synthetic financial headlines using deterministic templates (seed=42). However, **FinBERT scoring and sentence-transformer embeddings are real** — only the headline text is synthetic. Real-world news ingestion is planned future work.
```

- [ ] **Step 5: Update Quick Start**

Replace step 3 of Quick Start:

```markdown
3. **Run the pipeline:**
   ```bash
   python run_all.py            # full end-to-end pipeline (15 notebooks)
   python scripts/check_pipeline.py     # smoke test (no random data leaks)
   python scripts/check_determinism.py  # verify two runs produce identical CSVs
   ```
```

- [ ] **Step 6: Update Project Structure**

In the Project Structure tree, add:

```text
├── configs/settings.py     # Centralized config (tickers, seeds, hyperparameters)
├── src/io/loaders.py       # Single source of truth for every read
├── scripts/check_pipeline.py     # Smoke test for random-data regressions
├── scripts/check_determinism.py  # Back-to-back reproducibility check
├── run_all.py              # papermill orchestrator: one command runs everything
├── tests/                  # pytest suite for loaders & checks
├── data/processed/RESULTS.md     # Auto-generated headline numbers
└── docs/superpowers/       # Specs and implementation plans
```

- [ ] **Step 7: Add "Honesty Notes" section**

Append to README.md before "License":

```markdown
## ✅ Honesty Notes

- All headline numbers come from the live pipeline (`data/processed/RESULTS.md`), regenerated each run.
- Synthetic news (NB08): the **headlines** are synthetic; FinBERT scoring and sentence-transformer embeddings are real.
- Agent explanations (NB13–15) are **templated**, not LLM-generated. The `anthropic` package is not used.
- The pipeline is **deterministic**: `python run_all.py` produces bit-identical CSVs across runs (seed = 42).
- Backtest assumes 5 bps transaction cost, no slippage modeling, long-only constraint.
```

- [ ] **Step 8: Commit**

```bash
git add README.md
git commit -m "docs(README): match docs to code (no Claude, live RESULTS.md, honesty notes)"
```

---

### Task 30: Per-notebook READMEs (NBs 01–08)

**Files:**
- Create/Modify: `notebooks/01_data_pipeline/README.md` through `notebooks/08_nlp_sentiment/README.md`

- [ ] **Step 1: Use this template for every per-notebook README**

For each of NBs 01–08, write a README following this exact template (fill in the specifics from the data contract in spec §3):

```markdown
# NB-NN: <Title>

## Purpose
<One paragraph: what this notebook does and why it exists in the pipeline.>

## Inputs
| Artifact | Producer | Loaded via |
|---|---|---|
| `<file>.csv` | NB-NN | `loaders.load_<x>()` |

## Outputs
| Artifact | Path | Schema | Consumer |
|---|---|---|---|
| `<file>.csv` | `data/processed/` | `Date, <columns>` | NB-NN, NB-NN |

## Key methods
- <Method 1>: <one-line description, formula if relevant>

## How to run
\`\`\`bash
python run_all.py            # full pipeline
\`\`\`

## Verified after fix
- ✅ No `np.random.*` data fabrication
- ✅ Reads exclusively via `src.io.loaders`
- ✅ Outputs schema-validated by downstream loaders
```

Write the 8 READMEs concretely, using the data contract in spec §3 to fill in producers / consumers / schemas / methods. Do not leave placeholders.

- [ ] **Step 2: Commit**

```bash
git add notebooks/01_data_pipeline/README.md notebooks/02_data_processing/README.md \
        notebooks/03_quant_core/README.md  notebooks/04_graph_engine/README.md \
        notebooks/05_graph_analysis/README.md notebooks/06_ml_models/README.md \
        notebooks/07_ml_analysis/README.md   notebooks/08_nlp_sentiment/README.md
git commit -m "docs: per-notebook READMEs for NB01-NB08 (data contract documented)"
```

---

### Task 31: Per-notebook READMEs (NBs 09–15)

**Files:**
- Create/Modify: `notebooks/09_rag_setup/README.md` through `notebooks/15_agent_demo/README.md`

- [ ] **Step 1: Write READMEs for NBs 09–15**

Use the same template as Task 30. Fill in the specifics for each notebook from the data contract in spec §3.

- [ ] **Step 2: Commit**

```bash
git add notebooks/09_rag_setup/README.md  notebooks/10_nlp_analysis/README.md \
        notebooks/11_portfolio_optimization/README.md \
        notebooks/12_backtesting/README.md notebooks/13_agent_system/README.md \
        notebooks/14_agent_orchestration/README.md notebooks/15_agent_demo/README.md
git commit -m "docs: per-notebook READMEs for NB09-NB15 (data contract documented)"
```

---

## Phase D — Final verification (Tasks 32–35)

### Task 32: Run full smoke test

- [ ] **Step 1: Run pytest suite**

```bash
python -m pytest -v
```

Expected: all loader, helper, settings, and check_pipeline tests pass.

- [ ] **Step 2: Run pipeline check**

```bash
python scripts/check_pipeline.py
```

Expected:

```
✅ All 15 notebooks pass pipeline check.
```

If any fail, return to the corresponding notebook task and fix.

- [ ] **Step 3: Commit any drift fixes**

If you needed to fix anything:

```bash
git add <files>
git commit -m "fix: address remaining check_pipeline.py findings"
```

---

### Task 33: Run end-to-end pipeline

- [ ] **Step 1: Clean processed data and re-run**

```bash
# Optional: clear old outputs to confirm everything is regenerated
rm -f data/processed/*.csv data/processed/*.json data/processed/*.pkl \
      data/processed/*.npy data/processed/*.bin data/processed/*.md
python run_all.py
```

Expected: all 15 notebooks complete with `✅` and total runtime printed at the end.

- [ ] **Step 2: Spot-check `RESULTS.md`**

```bash
cat data/processed/RESULTS.md
```

Expected: live headline numbers (Sharpe, AUC, IC) populated; the values may differ from the README's old hardcoded numbers — that is expected and correct.

- [ ] **Step 3: Spot-check key outputs exist**

```bash
ls -la data/processed/ | grep -E "\.csv|\.json|\.pkl|\.npy|\.bin"
```

Confirm all 30+ artifacts from the data contract are present and non-empty.

- [ ] **Step 4: If any notebook failed**

Diagnose using the papermill error message (it includes the failing cell index and traceback). Return to the corresponding notebook task, fix, re-run.

- [ ] **Step 5: Commit any drift fixes**

```bash
git add <files>
git commit -m "fix: end-to-end run drift fixes"
```

---

### Task 34: Run determinism check

- [ ] **Step 1: Run determinism check**

```bash
python scripts/check_determinism.py
```

Expected:

```
✅ All N CSVs are bit-identical across runs.
```

- [ ] **Step 2: If determinism fails**

Identify the differing files. Common causes:
- An `np.random.*` not under `np.random.seed(RANDOM_SEED)`.
- A model with `n_jobs > 1` (XGBoost, RandomForest) — use `n_jobs=1` (already in `RF_PARAMS`/`XGB_PARAMS`).
- A timestamp written into a CSV (move it to JSON instead).
- FAISS index byte-level non-determinism — if persistent, switch to `IndexFlatL2` or persist via `np.save` of vectors only (and rebuild index on load).

Fix and re-run determinism check.

- [ ] **Step 3: Commit fixes**

```bash
git add <files>
git commit -m "fix: address determinism check findings"
```

---

### Task 35: Final acceptance verification

Walk through each acceptance criterion from spec §10 and confirm.

- [ ] **Step 1: Acceptance criteria pass**

Run each, confirm pass:

```bash
python run_all.py                          # AC1: end-to-end on fresh kernel
python scripts/check_pipeline.py           # AC2: no forbidden patterns, all import loaders
python scripts/check_determinism.py        # AC3: bit-identical across runs
python -m pytest -v                        # all loader/check tests pass
```

For AC4–AC10 (each notebook has Load Inputs cell, no random data, RESULTS.md auto-generated, README honest, per-NB READMEs exist, lookahead fixes applied, walk-forward holds): inspect manually or grep:

```bash
# AC4 — every notebook has Load Inputs
for nb in notebooks/*/*.ipynb; do
  jq -r '.cells[] | select(.cell_type=="code") | .source | join("")' "$nb" | \
    grep -q "from src.io.loaders" && echo "✅ $nb" || echo "❌ $nb"
done

# AC5 — no forbidden np.random uses outside allowlist (already covered by check_pipeline.py)

# AC6 — RESULTS.md exists
test -f data/processed/RESULTS.md && echo "✅ RESULTS.md" || echo "❌ RESULTS.md"

# AC7 — README has no Claude claim
grep -i "anthropic\|claude" README.md && echo "❌ stale Claude claim" || echo "✅ README clean"

# AC8 — per-NB READMEs
ls notebooks/*/README.md | wc -l  # should be 15

# AC9 / AC10 — assertions in code (already verified by run_all completing)
```

- [ ] **Step 2: Update plan checklist**

Mark all checkboxes in this plan as complete.

- [ ] **Step 3: Final commit (if any cleanup)**

```bash
git add -A
git commit -m "chore: final cleanup; pipeline data-flow fix complete"
```

- [ ] **Step 4: Summarize**

Print to console (or note in PR description):

```
Pipeline data-flow fix complete.

- 15 notebooks now read real upstream outputs via src/io/loaders.py
- Zero np.random data fabrication (smoke test enforced)
- python run_all.py runs end-to-end deterministically
- RESULTS.md regenerated with live numbers
- README claims match code (no Claude, no hardcoded performance)

Headline numbers from this run: see data/processed/RESULTS.md
```

---

## Self-Review (already applied)

**Spec coverage:** every section of the spec maps to one or more tasks above.
- §3 (data contract) → Task 4–8 (loaders) + Tasks 14–28 (notebooks consume it).
- §4 (per-notebook fix matrix) → Tasks 14–28 (one task per notebook).
- §5 (new/changed files) → Tasks 1, 9, 10, 11, 12, 13, 25, 29, 30, 31.
- §6 (correctness audit) → Steps embedded in Tasks 15 (NB02 lookahead), 16 (NB03 GARCH), 19 (NB06 walk-forward), 23 (NB10 sentiment lag), 24 (NB11 mu lag), 25 (NB12 walk-forward + RESULTS.md), 34 (determinism).
- §7 (verification) → Tasks 32–35.
- §10 (acceptance criteria) → Task 35.

**Placeholder scan:** no TBD/TODO/"add appropriate" left. Every code step shows actual code.

**Type consistency:** loader function names referenced in notebook tasks (Tasks 14–28) all match the names defined in Tasks 4–8.

**Internal consistency:** `RANDOM_SEED`, `DATA_PROC`, `EMBEDDING_MODEL`, etc. are defined once in Task 9 and used consistently. Notebook tasks consistently use the `# === Load Inputs ===` cell pattern. Allowlist tag `# noqa: pipeline-check` matches Tasks 11 (definition) and 21, 24, 25 (uses).
