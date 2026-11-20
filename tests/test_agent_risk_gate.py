import json
from pathlib import Path
import pandas as pd


def _nodes():
    path = next((Path(__file__).resolve().parents[1] / 'notebooks' / '14_agent_orchestration').glob('*.ipynb'))
    notebook = json.loads(path.read_text())
    namespace = {}
    exec(''.join(notebook['cells'][7]['source']), namespace)
    return namespace


def test_risk_gate_blocks_bad_allocation_and_missing_metrics():
    nodes = _nodes()
    nodes['load_performance_metrics'] = lambda: {'AxiomAlpha': {'max_dd': -0.2}}
    nodes['load_optimal_weights'] = lambda: pd.DataFrame({'Date': ['2024-12-30'], 'ticker': ['A'], 'weight': [.15]})
    state = nodes['node_risk']({})
    assert state['risk_alert']
    assert nodes['check_risk'](state) == 'block'
    assert nodes['node_strategy'](state)['strategy_decision']['status'] == 'blocked'
    nodes['load_performance_metrics'] = lambda: {}
    state = nodes['node_risk']({})
    assert state['risk_alert']


def test_risk_gate_blocks_allocation_older_than_market_data():
    nodes = _nodes()
    nodes['load_performance_metrics'] = lambda: {'AxiomAlpha': {'max_dd': -0.2}}
    nodes['load_optimal_weights'] = lambda: pd.DataFrame({'Date': ['2024-12-29'], 'ticker': ['A'], 'weight': [.05]})
    state = nodes['node_risk']({'data_report': {'last_date': '2024-12-30'}})
    assert state['risk_alert']
    assert 'allocation date' in state['risk_audit']['alert']
