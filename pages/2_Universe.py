"""Universe page — per-asset stats, sector composition, individual walks."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.settings import (
    STOCK_UNIVERSE, SECTOR_MAP, TICKERS, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR, DATA_PROC,
)
from src.io.loaders import load_log_returns, load_prices, PipelineDataError
from src.quant_core import annualized_sharpe, annualized_sortino, max_drawdown

LUCIDE = "https://cdn.jsdelivr.net/npm/lucide-static@latest/icons"
def icon(name, size=18):
    return (f'<img src="{LUCIDE}/{name}.svg" width="{size}" height="{size}" '
            f'align="absmiddle" style="vertical-align:middle;margin-right:6px;">')

st.set_page_config(page_title="Universe · AxiomAlpha", layout="wide")
st.markdown(f"<h1>{icon('layout-grid', 28)} Universe</h1>", unsafe_allow_html=True)
st.caption(f"{len(TICKERS)} tickers across {len(STOCK_UNIVERSE)} sectors.")


@st.cache_data
def _load_log_returns() -> pd.DataFrame | None:
    try:
        return load_log_returns()
    except PipelineDataError:
        return None


@st.cache_data
def _load_prices() -> pd.DataFrame | None:
    try:
        return load_prices()
    except PipelineDataError:
        return None


@st.cache_data
def _load_risk_metrics() -> pd.DataFrame | None:
    p = DATA_PROC / "risk_metrics.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


log_returns = _load_log_returns()
prices = _load_prices()
risk = _load_risk_metrics()

if log_returns is None or prices is None:
    st.error(
        "`log_returns.csv` or `prices.csv` not found. Run `python run_all.py` "
        "(or `python run_all.py --from 02` if data/raw/ is already populated)."
    )
    st.stop()

# --- Sector composition ---
st.markdown(f"<h3>{icon('pie-chart', 22)} Sector composition</h3>", unsafe_allow_html=True)
sector_df = pd.DataFrame(
    [(sector, ticker) for sector, tickers in STOCK_UNIVERSE.items() for ticker in tickers],
    columns=["Sector", "Ticker"],
)
sec_count = sector_df.groupby("Sector").size().reset_index(name="Count")
col1, col2 = st.columns([1, 2])
with col1:
    fig = px.pie(sec_count, values="Count", names="Sector",
                 template="plotly_dark", hole=0.4, height=320)
    fig.update_traces(textinfo="label+percent")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.dataframe(
        sector_df.groupby("Sector")["Ticker"].apply(lambda x: " · ".join(x)).reset_index().rename(
            columns={"Ticker": "Members"}),
        use_container_width=True, hide_index=True,
    )

# --- Per-asset risk-adjusted summary ---
st.markdown(f"<h3>{icon('table', 22)} Per-asset summary</h3>", unsafe_allow_html=True)
summary = pd.DataFrame(index=log_returns.columns)
summary["Sector"]    = summary.index.map(SECTOR_MAP)
summary["AnnReturn"] = log_returns.mean() * TRADING_DAYS_PER_YEAR
summary["AnnVol"]    = log_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
summary["Sharpe"]    = log_returns.apply(lambda c: annualized_sharpe(c.dropna().values))
summary["Sortino"]   = log_returns.apply(lambda c: annualized_sortino(c.dropna().values))

cum = np.exp(log_returns.cumsum())
summary["MaxDD"] = cum.apply(lambda c: max_drawdown(c.dropna().values))
summary["VaR_99"] = log_returns.quantile(0.01)
summary = summary.reset_index().rename(columns={"index": "Ticker"})

# Optional sector filter
sector_filter = st.multiselect("Filter sectors", options=sorted(STOCK_UNIVERSE.keys()),
                               default=sorted(STOCK_UNIVERSE.keys()))
filtered = summary[summary["Sector"].isin(sector_filter)]

st.dataframe(
    filtered.style.format({
        "AnnReturn": "{:.2%}", "AnnVol": "{:.2%}", "Sharpe": "{:.3f}",
        "Sortino": "{:.3f}", "MaxDD": "{:.2%}", "VaR_99": "{:.2%}",
    }).background_gradient(subset=["Sharpe", "Sortino"], cmap="RdYlGn"),
    use_container_width=True, hide_index=True, height=440,
)

# --- Risk-Return scatter ---
st.markdown(f"<h3>{icon('scatter-chart', 22)} Risk-return scatter</h3>", unsafe_allow_html=True)
fig2 = px.scatter(filtered, x="AnnVol", y="AnnReturn", color="Sector", text="Ticker",
                  template="plotly_dark", hover_data=["Sharpe"], height=480)
fig2.update_traces(textposition="top center")
fig2.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                   xaxis_title="Annual volatility", yaxis_title="Annual return")
st.plotly_chart(fig2, use_container_width=True)

# --- Per-asset deep dive ---
st.markdown(f"<h3>{icon('search', 22)} Individual asset</h3>", unsafe_allow_html=True)
ticker = st.selectbox("Pick a ticker", options=list(log_returns.columns), index=0)

c1, c2 = st.columns(2)
with c1:
    fig_p = px.line(prices.reset_index(), x="Date", y=ticker,
                    template="plotly_dark", height=340,
                    title=f"{ticker} adjusted close")
    fig_p.update_layout(yaxis_title="Price ($)", xaxis_title=None)
    st.plotly_chart(fig_p, use_container_width=True)

with c2:
    rolling_vol = log_returns[ticker].rolling(20).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    fig_v = px.line(rolling_vol.dropna().reset_index().rename(columns={ticker: "Vol20"}),
                    x="Date", y="Vol20",
                    template="plotly_dark", height=340,
                    title=f"{ticker} 20-day annualized vol")
    fig_v.update_layout(yaxis_tickformat=".0%", yaxis_title="Vol", xaxis_title=None)
    st.plotly_chart(fig_v, use_container_width=True)

if risk is not None and "ticker" in risk.columns:
    r = risk[risk["ticker"] == ticker]
    if not r.empty:
        st.markdown(f"**Latest risk snapshot — {ticker}**")
        st.dataframe(r.set_index("Date" if "Date" in r.columns else r.columns[0]),
                     use_container_width=True)
