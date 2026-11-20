"""Render GitHub-friendly figures from saved AxiomAlpha research artifacts.

These are historical research results, not live market data or forecasts.
Run after the pipeline to refresh the README and notebook previews.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
OUTPUT = ROOT / 'docs' / 'figures'
OUTPUT.mkdir(parents=True, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUTPUT / name, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(OUTPUT / name)


# MST is the saved network snapshot from NB04; edge weights are distances.
edges = pd.read_csv(DATA / 'mst_edges.csv')
graph = nx.from_pandas_edgelist(edges, 'source', 'target', edge_attr='weight')
pos = nx.spring_layout(graph, seed=42, weight=None)
fig, ax = plt.subplots(figsize=(11, 8))
nx.draw_networkx_edges(graph, pos, ax=ax, alpha=.5, width=1.2)
nx.draw_networkx_nodes(graph, pos, ax=ax, node_color='#3182bd', node_size=600)
nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_color='white')
ax.set_title('Stock correlation network — minimum spanning tree')
ax.text(.5, -.03, 'Saved network snapshot; edges represent correlation-derived distances',
        transform=ax.transAxes, ha='center', fontsize=9, color='#555')
ax.axis('off')
save(fig, 'stock_network.png')

network = pd.read_csv(DATA / 'dynamic_graph_metrics.csv', parse_dates=['Date'])
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(network['Date'], network['avg_corr'], color='#2274a5', lw=1.5)
ax.set(title='Rolling market-network correlation', ylabel='Average pairwise correlation', xlabel='')
save(fig, 'network_correlation.png')

signals = pd.read_csv(DATA / 'master_signals.csv', parse_dates=['Date'])
latest_date = signals['Date'].max()
latest = signals[signals['Date'] == latest_date].sort_values('alpha_signal')
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(latest['ticker'], latest['alpha_signal'], color='#459d7f')
ax.axvline(.5, color='#d24d57', ls='--', label='0.5')
ax.set(title=f'Walk-forward model probabilities — {latest_date:%Y-%m-%d}', xlabel='Probability of positive next-day return')
ax.legend()
save(fig, 'ml_probabilities.png')

returns = pd.read_csv(DATA / 'backtest_results.csv', parse_dates=['Date']).set_index('Date')
curves = 100 * (1 + returns).cumprod()
fig, ax = plt.subplots(figsize=(11, 5))
for col in curves:
    ax.plot(curves.index, curves[col], label=col.replace('_', ' '), lw=1.6)
ax.set(title='Historical backtest: portfolio value', ylabel='Value of $100', xlabel='')
ax.legend()
ax.text(.01, .02, 'Research backtest; not a prediction of future returns', transform=ax.transAxes, fontsize=8, color='#555')
save(fig, 'backtest_equity.png')

fig, ax = plt.subplots(figsize=(11, 4))
for col in curves:
    peak = curves[col].cummax().clip(lower=100)
    ax.plot(curves.index, (curves[col] / peak - 1) * 100, label=col.replace('_', ' '), lw=1.3)
ax.set(title='Historical backtest: drawdown', ylabel='Drawdown (%)', xlabel='')
ax.legend()
save(fig, 'backtest_drawdown.png')

metrics = pd.read_json(DATA / 'nlp_metrics.json', typ='series')
per_ticker = pd.Series(metrics['ic_per_ticker']).sort_values()
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(per_ticker.index, per_ticker.values, color=['#d65b5b' if x < 0 else '#459d7f' for x in per_ticker.values])
ax.axvline(0, color='#555', lw=.8)
ax.set(title='Synthetic-news sentiment correlation by ticker', xlabel='Pearson correlation with next trading-day return')
ax.text(.5, -.08, 'Synthetic headlines are partly conditioned on same-day returns; this is not real-news alpha',
        transform=ax.transAxes, ha='center', fontsize=8, color='#555')
save(fig, 'synthetic_sentiment.png')

# GitHub displays saved notebook outputs. Embed the same PNGs as a preview,
# and keep executable display cells so the image is reproducible locally.
import base64
import json

previews = {
    '04_graph_engine': [('Stock correlation network', 'stock_network.png'),
                        ('Rolling network correlation', 'network_correlation.png')],
    '07_ml_analysis': [('Walk-forward model probabilities', 'ml_probabilities.png')],
    '10_nlp_analysis': [('Synthetic-news sentiment correlation', 'synthetic_sentiment.png')],
    '12_backtesting': [('Historical equity curves', 'backtest_equity.png'),
                       ('Historical drawdowns', 'backtest_drawdown.png')],
}
for folder, charts in previews.items():
    notebook_path = ROOT / 'notebooks' / folder / f'{folder}.ipynb'
    notebook = json.loads(notebook_path.read_text())
    # Preserve all authored cells and replace only previously generated previews.
    notebook['cells'] = [c for c in notebook['cells'] if not c.get('id', '').startswith('github-preview-')]
    preview_cells = []
    for number, (title, filename) in enumerate(charts):
        encoded = base64.b64encode((OUTPUT / filename).read_bytes()).decode('ascii')
        preview_cells.append({
            'cell_type': 'markdown', 'id': f'github-preview-title-{number}', 'metadata': {},
            'source': [f'### {title}\n', '\n', 'Saved preview generated from the current processed artifacts. Re-run `python scripts/render_github_figures.py` after updating results.\n'],
        })
        preview_cells.append({
            'cell_type': 'code', 'id': f'github-preview-image-{number}', 'metadata': {},
            'execution_count': None,
            'source': ['from IPython.display import Image, display\n', f'display(Image(filename="../../docs/figures/{filename}"))\n'],
            'outputs': [{'output_type': 'display_data', 'metadata': {},
                         'data': {'image/png': encoded,
                                  'text/plain': ['<IPython.core.display.Image object>']}}],
        })
    notebook['cells'][1:1] = preview_cells
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    print(f'Updated notebook previews: {notebook_path}')
