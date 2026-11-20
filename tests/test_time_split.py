import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def test_notebook_split_keeps_all_tickers_together_and_embargoes_labels():
    path = next((Path(__file__).resolve().parents[1] / 'notebooks' / '06_ml_models').glob('*.ipynb'))
    notebook = json.loads(path.read_text())
    tree = ast.parse(''.join(notebook['cells'][8]['source']))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DateTimeSplit')
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    env = {'pd': pd, 'np': np, 'TimeSeriesSplit': TimeSeriesSplit}
    exec(compile(module, str(path), 'exec'), env)
    dates = pd.date_range('2024-01-01', periods=60, freq='B')
    X = pd.DataFrame({'x': range(120)}, index=dates.repeat(2))
    for train, test in env['DateTimeSplit'](n_splits=3, gap=5).split(X):
        train_dates, test_dates = X.index[train].unique(), X.index[test].unique()
        assert train_dates.max() < test_dates.min()
        assert dates.get_loc(test_dates.min()) - dates.get_loc(train_dates.max()) == 6
        assert len(train) % 2 == 0 and len(test) % 2 == 0
