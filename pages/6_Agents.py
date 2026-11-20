"""Agents page — NB14 LangGraph state, per-agent summaries, final allocation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.settings import DATA_PROC, DATA_OUT, SECTOR_MAP

LUCIDE = "https://cdn.jsdelivr.net/npm/lucide-static@latest/icons"
def icon(name, size=18):
    return (f'<img src="{LUCIDE}/{name}.svg" width="{size}" height="{size}" '
            f'align="absmiddle" style="vertical-align:middle;margin-right:6px;">')

st.set_page_config(page_title="Agents · AxiomAlpha", layout="wide")
st.markdown(f"<h1>{icon('network', 28)} Agent orchestration</h1>", unsafe_allow_html=True)
st.caption("Seven LangGraph nodes — Data, Quant, Graph, ML, NLP, Risk, Strategy — and their final state.")


@st.cache_data
def load_pipeline_output() -> dict:
    p = DATA_PROC / "pipeline_output.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


pipe = load_pipeline_output()
if not pipe:
    st.error("`pipeline_output.json` missing. Run `python run_all.py --from 14`.")
    st.stop()

# --- Top status row ---
qs = pipe.get("quant_summary", {})
sd = pipe.get("strategy_decision", {})
risk_audit = pipe.get("risk_audit", {})
ml_summary = pipe.get("ml_summary", {})
nlp_summary = pipe.get("nlp_summary", {})
graph_summary = pipe.get("graph_summary", {})
data_report = pipe.get("data_report", {})

status_color = {
    "Bull": "green", "Bear": "red", "Volatile": "orange", "Unknown": "gray"
}.get(qs.get("current_regime_name", {0: "Bear", 1: "Volatile", 2: "Bull"}.get(qs.get("current_regime_encoded"), "Unknown")), "gray")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Regime", qs.get("current_regime_name", {0: "Bear", 1: "Volatile", 2: "Bull"}.get(qs.get("current_regime_encoded"), "Unknown")))
c2.metric("Cash %", f"{float(sd.get('cash_pct', 0)):.1%}")
c3.metric("Holdings", len(sd.get("top_holdings", [])))
c4.metric("Status", pipe.get("status", "unknown"))

if pipe.get("risk_alert"):
    st.error(f"Risk guardrail breached: {risk_audit.get('alert', 'see RiskAgent')}")
else:
    st.success("Risk guardrails clean.")

st.divider()

# The graph is sequential; invalid data and risk alerts stop the workflow.
st.caption("Data → Quant → Graph → ML → NLP → Risk → Strategy; data or risk failures stop the decision.")

# --- Per-agent expanders ---
st.markdown(f"<h3>{icon('boxes', 22)} Per-agent summaries</h3>", unsafe_allow_html=True)

def render_kv(d: dict) -> None:
    """Render a dict as a flat key/value table where possible."""
    if not d:
        st.caption("(empty)")
        return
    rows = []
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            rows.append((k, "—" if v is None else str(v)))
        elif isinstance(v, list):
            rows.append((k, f"<{len(v)} items>"))
        elif isinstance(v, dict):
            rows.append((k, f"<{len(v)} keys>"))
        else:
            rows.append((k, str(type(v).__name__)))
    st.dataframe(pd.DataFrame(rows, columns=["field", "value"]),
                 hide_index=True, use_container_width=True)


with st.expander(f"DataAgent — {data_report.get('log_returns_rows', '?')} days × "
                 f"{data_report.get('log_returns_tickers', '?')} tickers", expanded=False):
    render_kv(data_report)

with st.expander(f"QuantAgent — regime = {qs.get('current_regime_name', '?')}", expanded=False):
    render_kv(qs)

with st.expander(f"GraphAgent — {len(graph_summary.get('top_systemic_tickers', []))} systemic anchors",
                 expanded=False):
    top_sys = graph_summary.get("top_systemic_tickers", [])
    if top_sys:
        df_sys = pd.DataFrame(top_sys)
        fig = px.bar(df_sys, x="ticker", y="score", template="plotly_dark", height=260,
                     color="score", color_continuous_scale="Reds")
        fig.update_layout(yaxis_title="Systemic-risk score", xaxis_title=None,
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    top_pr = graph_summary.get("top_pagerank_tickers", [])
    if top_pr:
        st.write("Top pagerank:")
        st.dataframe(pd.DataFrame(top_pr).style.format({"pagerank": "{:.4f}"}),
                     hide_index=True, use_container_width=True)

with st.expander(f"MLAgent — {len(ml_summary.get('top_long_signals', []))} long picks",
                 expanded=False):
    top_long = ml_summary.get("top_long_signals", [])
    if top_long:
        df_ml = pd.DataFrame(top_long)
        fig = px.bar(df_ml, x="ticker", y="p_up", template="plotly_dark", height=260,
                     color="p_up", color_continuous_scale="Greens",
                     range_y=[0.4, max(0.65, df_ml["p_up"].max() + 0.02)])
        fig.add_hline(y=0.5, line_dash="dash", line_color="white", opacity=0.5)
        fig.update_layout(yaxis_title="P(up tomorrow)", xaxis_title=None,
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Signal date: {ml_summary.get('as_of', 'n/a')} · Model trained through: {ml_summary.get('training_end', 'n/a')}. Models refit at fold boundaries, not every month.")

with st.expander(f"NLPAgent — sentiment across "
                 f"{len(nlp_summary.get('avg_sentiment_30d', {}))} tickers", expanded=False):
    sent = nlp_summary.get("avg_sentiment_30d", {})
    if sent:
        df_s = pd.Series(sent).sort_values().reset_index()
        df_s.columns = ["ticker", "sentiment_30d"]
        df_s["sector"] = df_s["ticker"].map(SECTOR_MAP)
        fig = px.bar(df_s, x="ticker", y="sentiment_30d", color="sector",
                     template="plotly_dark", height=260)
        fig.add_hline(y=0, line_color="white", opacity=0.5)
        fig.update_layout(xaxis_title=None, yaxis_title="Mean sentiment (30d)")
        st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Most positive: {nlp_summary.get('most_positive', '-')}  ·  "
        f"Most negative: {nlp_summary.get('most_negative', '-')}"
    )

with st.expander(f"RiskAgent — alert = {pipe.get('risk_alert', False)}", expanded=False):
    if risk_audit.get("perf_dict"):
        pd_perf = pd.DataFrame(risk_audit["perf_dict"]).T
        st.dataframe(pd_perf, use_container_width=True)
    if risk_audit.get("alert"):
        st.error(risk_audit["alert"])

with st.expander(f"StrategyAgent — {len(sd.get('top_holdings', []))} holdings",
                 expanded=True):
    holdings = sd.get("top_holdings", [])
    if holdings:
        df_h = pd.DataFrame(holdings).sort_values("weight", ascending=True)
        df_h["sector"] = df_h["ticker"].map(SECTOR_MAP)
        fig = px.bar(df_h, x="weight", y="ticker", orientation="h", color="sector",
                     template="plotly_dark", height=420)
        fig.update_layout(yaxis_title=None, xaxis_title="Weight",
                          xaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)
    if sd.get("rationale"):
        st.caption(sd["rationale"])

# --- Sector pie ---
if sd.get("top_holdings"):
    st.markdown(f"<h3>{icon('pie-chart', 22)} Final sector allocation</h3>", unsafe_allow_html=True)
    df_h = pd.DataFrame(sd["top_holdings"])
    df_h["sector"] = df_h["ticker"].map(SECTOR_MAP)
    sec_alloc = df_h.groupby("sector")["weight"].sum().reset_index()
    cash_pct = float(sd.get("cash_pct", 0))
    if cash_pct > 0:
        sec_alloc = pd.concat([
            sec_alloc,
            pd.DataFrame([{"sector": "Cash", "weight": cash_pct}])
        ], ignore_index=True)
    fig = px.pie(sec_alloc, values="weight", names="sector",
                 template="plotly_dark", hole=0.4, height=420)
    fig.update_traces(textinfo="label+percent")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "Explanations are templated (NB13 agent classes), not LLM-generated. "
    "`langchain` and `anthropic` are intentionally not in requirements.txt."
)
