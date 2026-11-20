"""NLP page — sentiment IC, per-ticker scores, FinBERT examples."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.settings import DATA_PROC, DATA_RAW, SECTOR_MAP

LUCIDE = "https://cdn.jsdelivr.net/npm/lucide-static@latest/icons"
def icon(name, size=18):
    return (f'<img src="{LUCIDE}/{name}.svg" width="{size}" height="{size}" '
            f'align="absmiddle" style="vertical-align:middle;margin-right:6px;">')

st.set_page_config(page_title="NLP · AxiomAlpha", layout="wide")
st.markdown(f"<h1>{icon('message-circle', 28)} NLP & sentiment</h1>", unsafe_allow_html=True)

st.warning(
    "**Synthetic news.** Headlines are templated and partly determined by the same-day return. "
    "The sentiment analysis is a technical diagnostic and is not evidence of real-news alpha. "
    "Sentiment is not used by the portfolio strategy."
)


@st.cache_data
def load_json(name: str) -> dict:
    p = DATA_PROC / name
    if not p.exists():
        return {}
    return json.loads(p.read_text())


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


provenance = load_json("sentiment_provenance.json")
st.caption(f"Sentiment scorer: {provenance.get('scorer', 'not recorded')}")
nlp_metrics = load_json("nlp_metrics.json")
sentiment = load_csv(DATA_PROC / "sentiment_features.csv")
news = load_csv(DATA_RAW / "synthetic_news.csv")

# --- Headline IC ---
st.markdown(f"<h3>{icon('gauge', 22)} Sentiment IC</h3>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.metric("Overall IC (1d lag)", f"{nlp_metrics.get('ic_lagged', float('nan')):.4f}",
          help="Mean per-ticker Pearson correlation between available sentiment on the previous trading day and today’s return.")
c2.metric("Tickers covered", nlp_metrics.get("n_tickers", 0))
c3.metric("Lag (days)", nlp_metrics.get("sentiment_lag_days", 1))

# --- Per-ticker IC ---
if nlp_metrics.get("ic_per_ticker"):
    st.markdown(f"<h3>{icon('chart-bar', 22)} IC by ticker</h3>", unsafe_allow_html=True)
    ic = pd.Series(nlp_metrics["ic_per_ticker"]).sort_values()
    df = ic.reset_index()
    df.columns = ["ticker", "ic"]
    df["sector"] = df["ticker"].map(SECTOR_MAP)
    fig = px.bar(df, x="ticker", y="ic", color="sector",
                 template="plotly_dark", height=380,
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.add_hline(y=0, line_color="white", opacity=0.5)
    fig.update_layout(xaxis_title=None, yaxis_title="Pearson correlation")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Per-ticker correlations describe these synthetic headlines and should not be interpreted as real-news alpha."
    )

# --- Per-ticker recent sentiment ---
if sentiment is not None and not sentiment.empty:
    st.markdown(f"<h3>{icon('activity', 22)} Recent sentiment</h3>", unsafe_allow_html=True)
    s = sentiment.copy()
    if "Date" in s.columns:
        s["Date"] = pd.to_datetime(s["Date"])
        max_d = s["Date"].max()
        cutoff = max_d - pd.Timedelta(days=90)
        recent = s[s["Date"] >= cutoff]
    else:
        recent = s
    if "ticker" in recent.columns and "sentiment_score" in recent.columns:
        ts = (recent.groupby(["Date", "ticker"])["sentiment_score"].mean()
              .reset_index())
        chosen = st.multiselect(
            "Tickers to plot",
            sorted(ts["ticker"].unique()),
            default=sorted(ts["ticker"].unique())[:5],
        )
        if chosen:
            ts_filtered = ts[ts["ticker"].isin(chosen)]
            fig2 = px.line(ts_filtered, x="Date", y="sentiment_score", color="ticker",
                           template="plotly_dark", height=380)
            fig2.update_layout(xaxis_title=None, yaxis_title="Sentiment score",
                               legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                            xanchor="right", x=1))
            st.plotly_chart(fig2, use_container_width=True)

# --- Sample headlines + FinBERT scores ---
if news is not None and not news.empty:
    st.markdown(f"<h3>{icon('newspaper', 22)} Sample synthetic headlines</h3>",
                unsafe_allow_html=True)
    n = st.slider("Number of samples", 5, 30, 10)
    cols_to_show = [c for c in ["Date", "ticker", "headline", "sentiment_score", "sentiment_label"]
                    if c in news.columns]
    sample = news.sample(n=min(n, len(news)), random_state=42)[cols_to_show]
    if "sentiment_score" in sample.columns:
        st.dataframe(
            sample.style.format({"sentiment_score": "{:+.3f}"})
                  .background_gradient(subset=["sentiment_score"],
                                       cmap="RdYlGn", vmin=-1, vmax=1),
            hide_index=True, use_container_width=True,
        )
    else:
        st.dataframe(sample, hide_index=True, use_container_width=True)
elif (DATA_RAW / "synthetic_news.csv").exists():
    st.info("synthetic_news.csv exists but couldn't be parsed.")
else:
    st.caption("`data/raw/synthetic_news.csv` not present (gitignored). "
               "Run NB08 to generate it.")
