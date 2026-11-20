"""Backtest page — equity curves, drawdowns, regime breakdown, costs."""
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

from configs.settings import (
    DATA_PROC, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR,
    TRANSACTION_COST_BPS, SLIPPAGE_BPS,
)
from src.quant_core import portfolio_metrics

LUCIDE = "https://cdn.jsdelivr.net/npm/lucide-static@latest/icons"
def icon(name, size=18):
    return (f'<img src="{LUCIDE}/{name}.svg" width="{size}" height="{size}" '
            f'align="absmiddle" style="vertical-align:middle;margin-right:6px;">')

st.set_page_config(page_title="Backtest · AxiomAlpha", layout="wide")
st.markdown(f"<h1>{icon('chart-no-axes-combined', 28)} Backtest</h1>", unsafe_allow_html=True)
st.caption("Walk-forward monthly rebalancing; the test window begins after out-of-sample signals are available.")


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    p = DATA_PROC / name
    if not p.exists():
        return None
    return pd.read_csv(p)


equity = load_csv("equity_curves.csv")
backtest = load_csv("backtest_results.csv")
perf = load_csv("performance_metrics.csv")
regime_perf = load_csv("regime_performance.csv")
attribution = load_csv("attribution_results.csv")

if equity is None or backtest is None or perf is None:
    st.error(
        "Backtest artifacts missing. Run `python run_all.py` to regenerate "
        "`equity_curves.csv`, `backtest_results.csv`, `performance_metrics.csv`."
    )
    st.stop()

equity["Date"] = pd.to_datetime(equity["Date"])
backtest["Date"] = pd.to_datetime(backtest["Date"])

# --- Headline scorecard ---
st.markdown(f"<h3>{icon('gauge', 22)} Performance scorecard</h3>", unsafe_allow_html=True)
perf_display = perf.set_index("Strategy")
fmt = {
    "ann_return": "{:.2%}", "ann_vol": "{:.2%}", "sharpe": "{:.3f}",
    "sortino": "{:.3f}", "calmar": "{:.3f}", "max_dd": "{:.2%}",
    "var_95": "{:.2%}", "cvar_95": "{:.2%}", "n_days": "{:,.0f}",
}
st.dataframe(perf_display.style.format(fmt), use_container_width=True)

# --- Equity curve ---
st.markdown(f"<h3>{icon('line-chart', 22)} Equity curves</h3>", unsafe_allow_html=True)

equity_long = equity.melt(id_vars="Date", var_name="Strategy", value_name="Value")
fig = px.line(
    equity_long, x="Date", y="Value", color="Strategy",
    color_discrete_map={
        "Equal_Weight": "#51CF66", "Markowitz": "#FFD43B", "AxiomAlpha": "#00D4FF",
    },
    template="plotly_dark",
    height=420,
    title="$100 invested across the test period",
)
# Shade COVID + 2022 rate-hike windows.
fig.add_vrect(x0="2020-02-20", x1="2020-03-23", fillcolor="red", opacity=0.15, line_width=0,
              annotation_text="COVID", annotation_position="top left")
fig.add_vrect(x0="2022-01-01", x1="2022-12-31", fillcolor="orange", opacity=0.1, line_width=0,
              annotation_text="Rate-hike bear", annotation_position="top left")
fig.update_layout(yaxis_title="Portfolio value ($)", xaxis_title=None,
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True)

# --- Drawdown ---
st.markdown(f"<h3>{icon('arrow-down-from-line', 22)} Drawdowns</h3>", unsafe_allow_html=True)

def drawdown(returns: pd.Series) -> pd.Series:
    cum = (1 + returns).cumprod()
    peak = cum.cummax().clip(lower=1.0)
    return (cum - peak) / peak

dd = pd.DataFrame({
    "Date": backtest["Date"],
    "Equal_Weight": drawdown(backtest["Equal_Weight"]).values,
    "Markowitz":    drawdown(backtest["Markowitz"]).values,
    "AxiomAlpha":   drawdown(backtest["AxiomAlpha"]).values,
})
dd_long = dd.melt(id_vars="Date", var_name="Strategy", value_name="Drawdown")
fig2 = px.area(
    dd_long, x="Date", y="Drawdown", color="Strategy",
    color_discrete_map={
        "Equal_Weight": "#51CF66", "Markowitz": "#FFD43B", "AxiomAlpha": "#00D4FF",
    },
    template="plotly_dark",
    height=350,
)
fig2.update_layout(yaxis_tickformat=".0%", yaxis_title="Drawdown", xaxis_title=None,
                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig2, use_container_width=True)

# --- Regime-conditional ---
if regime_perf is not None:
    st.markdown(f"<h3>{icon('layers', 22)} Regime-conditional performance</h3>", unsafe_allow_html=True)
    metric_choice = st.selectbox("Metric", ["sharpe", "ann_return"], index=0)
    fig3 = px.bar(
        regime_perf, x="regime", y=metric_choice, color="strategy",
        barmode="group",
        category_orders={"regime": ["Bull", "Volatile", "Bear"]},
        color_discrete_map={
            "Equal Weight": "#51CF66", "Markowitz": "#FFD43B", "AxiomAlpha": "#00D4FF",
        },
        template="plotly_dark",
        height=350,
    )
    yformat = ".2%" if metric_choice in ("ann_return", "max_dd") else ".2f"
    fig3.update_layout(yaxis_tickformat=yformat, xaxis_title=None)
    st.plotly_chart(fig3, use_container_width=True)

# Cost sensitivity requires actual trade and drift records; the previous
# return-volatility proxy could not represent turnover and double-counted costs.
st.info("Cost sensitivity is unavailable until trade-level turnover is saved by the backtest.")

# --- Attribution ---
if attribution is not None and not attribution.empty:
    st.markdown(f"<h3>{icon('pie-chart', 22)} Daily return contributions (AxiomAlpha)</h3>",
                unsafe_allow_html=True)
    attribution["Date"] = pd.to_datetime(attribution.iloc[:, 0])
    attr_long = attribution.set_index("Date").reset_index().melt(
        id_vars="Date", var_name="Sector", value_name="Contribution"
    )
    fig4 = px.area(attr_long, x="Date", y="Contribution", color="Sector",
                   template="plotly_dark", height=350)
    fig4.update_layout(yaxis_tickformat=".2%", xaxis_title=None, yaxis_title="Daily additive contribution")
    st.plotly_chart(fig4, use_container_width=True)
