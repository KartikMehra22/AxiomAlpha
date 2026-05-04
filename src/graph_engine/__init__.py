"""Pure graph-engine primitives extracted from NB04/NB05.

`networkx` wrappers for: correlation matrix → Mantegna distance → MST,
centrality measures, and rolling-window dynamic graph metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "correlation_matrix",
    "build_mst",
    "compute_centralities",
    "dynamic_graph_metrics",
]


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between every pair of return columns."""
    return returns.corr()


def _corr_to_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Mantegna distance: d_ij = sqrt(2 * (1 - rho_ij)). In [0, 2]."""
    return np.sqrt(2.0 * (1.0 - corr.clip(-1.0, 1.0)))


def build_mst(corr: pd.DataFrame):
    """Minimum-spanning tree over a correlation matrix using Mantegna distance.

    Returns a `networkx.Graph` with N-1 edges (where N is the number of
    tickers in `corr`). Edge weights are the Mantegna distances.
    """
    import networkx as nx

    dist = _corr_to_distance(corr)
    n = len(dist)
    g = nx.Graph()
    g.add_nodes_from(dist.columns)
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(
                dist.index[i], dist.columns[j],
                weight=float(dist.iat[i, j]),
            )
    return nx.minimum_spanning_tree(g, weight="weight")


def compute_centralities(graph) -> pd.DataFrame:
    """Per-node centrality metrics: degree, betweenness, eigenvector, pagerank.

    Returns a DataFrame indexed by node label with one column per measure.
    Eigenvector centrality may fail to converge on disconnected graphs;
    in that case the column is NaN-filled.
    """
    import networkx as nx

    deg = nx.degree_centrality(graph)
    bet = nx.betweenness_centrality(graph)
    pr  = nx.pagerank(graph) if graph.number_of_edges() else {n: 0.0 for n in graph.nodes}
    try:
        ev = nx.eigenvector_centrality_numpy(graph)
    except Exception:
        ev = {n: float("nan") for n in graph.nodes}
    return pd.DataFrame({
        "degree":      pd.Series(deg),
        "betweenness": pd.Series(bet),
        "eigenvector": pd.Series(ev),
        "pagerank":    pd.Series(pr),
    }).reindex(list(graph.nodes))


def dynamic_graph_metrics(
    returns: pd.DataFrame, window: int = 60,
) -> pd.DataFrame:
    """Rolling-window network metrics: density, avg_corr, clustering.

    For each window of `window` days, computes the correlation matrix and
    extracts:
      - density:    fraction of |corr|>0.5 pairs
      - avg_corr:   mean upper-triangle correlation
      - clustering: networkx.average_clustering on the |corr|>0.5 graph

    Returns a DataFrame with Date index.
    """
    import networkx as nx

    out_rows: list[dict] = []
    n = len(returns)
    if n < window:
        return pd.DataFrame(columns=["density", "avg_corr", "clustering"])
    tickers = list(returns.columns)
    for end in range(window, n + 1):
        sub = returns.iloc[end - window:end]
        if sub.isna().all().any():
            continue
        corr = sub.corr().values
        triu_idx = np.triu_indices_from(corr, k=1)
        triu_vals = corr[triu_idx]
        avg_corr = float(np.nanmean(triu_vals))
        strong_mask = np.abs(triu_vals) > 0.5
        density = float(strong_mask.mean()) if len(triu_vals) else 0.0
        g = nx.Graph()
        g.add_nodes_from(tickers)
        for k, (i, j) in enumerate(zip(*triu_idx)):
            if strong_mask[k]:
                g.add_edge(tickers[i], tickers[j])
        clustering = float(nx.average_clustering(g)) if g.number_of_edges() else 0.0
        out_rows.append({
            "Date":       returns.index[end - 1],
            "density":    density,
            "avg_corr":   avg_corr,
            "clustering": clustering,
        })
    df = pd.DataFrame(out_rows)
    if not df.empty:
        df = df.set_index("Date")
        df.index.name = "Date"
    return df
