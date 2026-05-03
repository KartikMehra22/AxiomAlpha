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
