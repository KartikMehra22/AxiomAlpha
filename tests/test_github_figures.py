import base64
import json
from pathlib import Path


def test_notebook_previews_match_generated_images():
    root = Path(__file__).resolve().parents[1]
    for folder, filenames in {
        '04_graph_engine': ['stock_network.png', 'network_correlation.png'],
        '07_ml_analysis': ['ml_probabilities.png'],
        '10_nlp_analysis': ['synthetic_sentiment.png'],
        '12_backtesting': ['backtest_equity.png', 'backtest_drawdown.png'],
    }.items():
        notebook = json.loads((root / 'notebooks' / folder / f'{folder}.ipynb').read_text())
        previews = [c for c in notebook['cells'] if c.get('id', '').startswith('github-preview-image-')]
        assert len(previews) == len(filenames)
        for cell, filename in zip(previews, filenames):
            png = base64.b64decode(cell['outputs'][0]['data']['image/png'])
            assert png == (root / 'docs' / 'figures' / filename).read_bytes()
