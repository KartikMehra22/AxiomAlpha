from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.requires_artifacts
def test_graph_dashboard_renders_with_ticker_identifiers():
    path = Path(__file__).resolve().parents[1] / 'pages' / '3_Graph.py'
    app = AppTest.from_file(str(path), default_timeout=45).run()
    assert not app.exception, [e.message for e in app.exception]
