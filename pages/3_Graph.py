"""Graph page — MST, communities, dynamic network metrics, per-day centrality."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.settings import DATA_PROC, SECTOR_MAP

LUCIDE = "https://cdn.jsdelivr.net/npm/lucide-static@latest/icons"
def icon(name, size=18):
    return (f'<img src="{LUCIDE}/{name}.svg" width="{size}" height="{size}" '
            f'align="absmiddle" style="vertical-align:middle;margin-right:6px;">')

st.set_page_config(page_title="Graph · AxiomAlpha", layout="wide")
st.markdown(f"<h1>{icon('share-2', 28)} Graph engine</h1>", unsafe_allow_html=True)
st.caption("Correlation networks, MST topology, community detection, and per-day centrality.")


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    p = DATA_PROC / name
    if not p.exists():
        return None
    return pd.read_csv(p)


mst_edges = load_csv("mst_edges.csv")
centrality = load_csv("centrality_scores.csv")
centrality_daily = load_csv("centrality_scores_daily.csv")
dynamic_metrics = load_csv("dynamic_graph_metrics.csv")
communities = load_csv("community_assignments.csv")
leader_follower = load_csv("leader_follower.csv")
systemic = load_csv("systemic_risk_scores.csv")

if centrality is None:
    st.error("`centrality_scores.csv` missing — run `python run_all.py --from 02` first.")
    st.stop()

# Normalize centrality index column if it's "Unnamed: 0"
if "Unnamed: 0" in centrality.columns:
    centrality = centrality.rename(columns={"Unnamed: 0": "ticker"})

# --- MST visualisation via plotly ---
st.markdown(f"<h3>{icon('git-fork', 22)} Minimum spanning tree</h3>", unsafe_allow_html=True)

if mst_edges is None or mst_edges.empty:
    st.warning("`mst_edges.csv` missing.")
else:
    try:
        import networkx as nx
        g = nx.Graph()
        edge_col_pair = ("source", "target") if {"source", "target"}.issubset(mst_edges.columns) else \
                        (mst_edges.columns[0], mst_edges.columns[1])
        weight_col = "weight" if "weight" in mst_edges.columns else None
        for _, row in mst_edges.iterrows():
            u, v = row[edge_col_pair[0]], row[edge_col_pair[1]]
            w = float(row[weight_col]) if weight_col else 1.0
            g.add_edge(u, v, weight=w)
        pos = nx.spring_layout(g, seed=42, k=1.5, iterations=200, weight=None)

        edge_x, edge_y = [], []
        for u, v in g.edges:
            edge_x.extend([pos[u][0], pos[v][0], None])
            edge_y.extend([pos[u][1], pos[v][1], None])

        # Node colour by sector, size by degree.
        deg = dict(g.degree())
        sectors = sorted(set(SECTOR_MAP.values()))
        palette = px.colors.qualitative.Set2
        sec_color = {s: palette[i % len(palette)] for i, s in enumerate(sectors)}

        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        for n in g.nodes:
            node_x.append(pos[n][0])
            node_y.append(pos[n][1])
            sec = SECTOR_MAP.get(n, "Other")
            node_color.append(sec_color.get(sec, "#888888"))
            node_size.append(15 + 5 * deg.get(n, 0))
            node_text.append(f"{n}<br>{sec}<br>degree={deg.get(n, 0)}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                 line=dict(color="#444", width=1), hoverinfo="none"))
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                 marker=dict(color=node_color, size=node_size,
                                             line=dict(color="white", width=0.5)),
                                 text=list(g.nodes), textposition="top center",
                                 hoverinfo="text", hovertext=node_text,
                                 textfont=dict(size=10, color="white")))
        fig.update_layout(template="plotly_dark", showlegend=False,
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                          height=600, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"MST render failed: {e}")

# --- Dynamic graph metrics ---
if dynamic_metrics is not None and not dynamic_metrics.empty:
    st.markdown(f"<h3>{icon('activity', 22)} Dynamic network metrics</h3>", unsafe_allow_html=True)
    dm = dynamic_metrics.copy()
    if "Date" in dm.columns:
        dm["Date"] = pd.to_datetime(dm["Date"])
    metric_cols = [c for c in dm.columns if c != "Date"]
    chosen = st.multiselect("Metrics", metric_cols, default=metric_cols)
    if chosen:
        dm_long = dm.melt(id_vars="Date", value_vars=chosen, var_name="Metric", value_name="Value")
        fig = px.line(dm_long, x="Date", y="Value", color="Metric",
                      template="plotly_dark", height=380)
        fig.update_layout(xaxis_title=None, legend=dict(orientation="h", yanchor="bottom",
                                                       y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

# --- Per-day centrality heatmap ---
if centrality_daily is not None and not centrality_daily.empty:
    st.markdown(f"<h3>{icon('grid-3x3', 22)} Per-day centrality</h3>", unsafe_allow_html=True)
    metric = st.selectbox("Centrality measure",
                          ["graph_pagerank", "graph_eigenvector", "graph_betweenness", "graph_degree"])
    cd = centrality_daily.copy()
    cd["Date"] = pd.to_datetime(cd["Date"])
    pivot = cd.pivot_table(index="ticker", columns="Date", values=metric, aggfunc="mean")
    # Downsample columns to make the heatmap readable.
    if pivot.shape[1] > 400:
        cols = np.linspace(0, pivot.shape[1] - 1, 400).astype(int)
        pivot = pivot.iloc[:, cols]
    fig = px.imshow(pivot, aspect="auto", template="plotly_dark",
                    color_continuous_scale="Plasma", height=520,
                    labels=dict(color=metric))
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Cell colour = centrality on that date for that ticker, from rolling 60-day MST.")

# --- Centrality / systemic table ---
st.markdown(f"<h3>{icon('list-ordered', 22)} Full-sample centrality ranking</h3>", unsafe_allow_html=True)
cent_table = centrality.copy()
if "ticker" not in cent_table.columns and cent_table.index.name == "ticker":
    cent_table = cent_table.reset_index()
elif "ticker" not in cent_table.columns:
    cent_table.insert(0, "ticker", cent_table.index)
cols_show = [c for c in ["ticker", "degree", "betweenness", "eigenvector", "pagerank", "systemic_risk_score"]
             if c in cent_table.columns]
st.dataframe(
    cent_table[cols_show].sort_values("pagerank", ascending=False).reset_index(drop=True)
        .style.background_gradient(subset=[c for c in cols_show if c != "ticker"],
                                   cmap="viridis").format("{:.4f}", subset=cols_show[1:]),
    use_container_width=True, hide_index=True, height=440,
)

# --- Community detection ---
if communities is not None and not communities.empty:
    st.markdown(f"<h3>{icon('users', 22)} Community detection vs GICS sectors</h3>", unsafe_allow_html=True)
    cdf = communities.copy()
    if "Unnamed: 0" in cdf.columns:
        cdf = cdf.rename(columns={"Unnamed: 0": "ticker"})
    elif cdf.columns[0].lower() in ("index", "ticker"):
        cdf = cdf.rename(columns={cdf.columns[0]: "ticker"})
    sector_col = "official_sector" if "official_sector" in cdf.columns else None
    if sector_col is None and "ticker" in cdf.columns:
        cdf["official_sector"] = cdf["ticker"].map(SECTOR_MAP)
        sector_col = "official_sector"
    methods = [c for c in ["louvain", "girvan_newman", "label_prop"] if c in cdf.columns]
    if methods and sector_col:
        method = st.selectbox("Algorithm", methods, index=0)
        pivot = pd.crosstab(cdf[method], cdf[sector_col])
        fig = px.imshow(pivot, text_auto=True, aspect="auto", template="plotly_dark",
                        color_continuous_scale="Blues", height=420)
        fig.update_layout(xaxis_title="GICS sector", yaxis_title=f"{method} community")
        st.plotly_chart(fig, use_container_width=True)

# --- Leader / follower ratio ---
if leader_follower is not None and not leader_follower.empty:
    st.markdown(f"<h3>{icon('arrow-right-left', 22)} Leader / follower</h3>", unsafe_allow_html=True)
    lf = leader_follower.copy()
    if "Unnamed: 0" in lf.columns:
        lf = lf.rename(columns={"Unnamed: 0": "ticker"})
    if "ratio" in lf.columns and "ticker" in lf.columns:
        lf = lf.sort_values("ratio", ascending=False)
        fig = px.bar(lf, x="ticker", y="ratio", template="plotly_dark", height=320,
                     color="ratio", color_continuous_scale="RdYlGn")
        fig.update_layout(xaxis_title=None, yaxis_title="Leadership ratio")
        st.plotly_chart(fig, use_container_width=True)

    elif "ticker" not in lf.columns:
        st.warning("Leader/follower artifact lacks ticker identifiers; regenerate notebook 05.")
