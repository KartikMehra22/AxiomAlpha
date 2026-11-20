"""ML signals page — model scorecard, per-ticker direction probs, calibration."""
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

from configs.settings import DATA_PROC, SECTOR_MAP

LUCIDE = "https://cdn.jsdelivr.net/npm/lucide-static@latest/icons"
def icon(name, size=18):
    return (f'<img src="{LUCIDE}/{name}.svg" width="{size}" height="{size}" '
            f'align="absmiddle" style="vertical-align:middle;margin-right:6px;">')

st.set_page_config(page_title="ML signals · AxiomAlpha", layout="wide")
st.markdown(f"<h1>{icon('brain', 28)} ML signals</h1>", unsafe_allow_html=True)
st.caption("Model diagnostics from notebook 06. The portfolio instead uses walk-forward random-forest signals from notebook 07.")


@st.cache_data
def load_json(name: str) -> dict:
    p = DATA_PROC / name
    if not p.exists():
        return {}
    return json.loads(p.read_text())


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    p = DATA_PROC / name
    if not p.exists():
        return None
    return pd.read_csv(p)


ml_eval = load_json("ml_eval.json")
ml_signals = load_csv("ml_signals.csv")
model_perf = load_csv("model_performance.csv")

if not ml_eval and ml_signals is None:
    st.error("ML artifacts missing. Run `python run_all.py --from 06`.")
    st.stop()

# --- Headline ML metrics ---
st.markdown(f"<h3>{icon('gauge', 22)} Model scorecard</h3>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Direction AUC", f"{ml_eval.get('avg_auc', float('nan')):.3f}",
          help="Mean ROC-AUC across validation folds for the selected model. Model selection used these folds; this is not an untouched holdout.")
c2.metric("Volatility RMSE", f"{ml_eval.get('avg_rmse', float('nan')):.4f}")
c3.metric("Volatility R²",  f"{ml_eval.get('avg_r2', float('nan')):.3f}")
c4.metric("Regime accuracy", f"{ml_eval.get('regime_accuracy', float('nan')):.1%}")

st.caption(
    "Best models — direction: **{}** · volatility: **{}** · regime: **{}**".format(
        ml_eval.get("best_model_direction", "-"),
        ml_eval.get("best_model_volatility", "-"),
        ml_eval.get("best_model_regime", "-"),
    )
)
if ml_eval.get("avg_auc", 0) < 0.55:
    st.warning(
        "Direction AUC is close to random. Treat the binary alpha as a noisy "
        "input rather than a strong signal — this is honest to the underlying "
        "data, not a bug."
    )

# --- Model performance table ---
if model_perf is not None and not model_perf.empty:
    st.markdown(f"<h3>{icon('table', 22)} Per-model performance</h3>", unsafe_allow_html=True)
    st.dataframe(model_perf, use_container_width=True, hide_index=True)

# --- ML signals snapshot ---
if ml_signals is not None and not ml_signals.empty:
    st.markdown(f"<h3>{icon('zap', 22)} Latest direction probabilities</h3>",
                unsafe_allow_html=True)
    sig = ml_signals.copy()
    if "Date" in sig.columns:
        sig["Date"] = pd.to_datetime(sig["Date"])
        latest = sig[sig["Date"] == sig["Date"].max()].copy()
    else:
        latest = sig
    if "ticker" in latest.columns and "direction_prob" in latest.columns:
        latest["sector"] = latest["ticker"].map(SECTOR_MAP)
        latest = latest.sort_values("direction_prob", ascending=False)
        fig = px.bar(latest, x="ticker", y="direction_prob", color="sector",
                     template="plotly_dark", height=380,
                     hover_data={"direction_prob": ":.3f"})
        fig.add_hline(y=0.5, line_dash="dash", line_color="white", opacity=0.5)
        fig.update_layout(xaxis_title=None, yaxis_title="P(up tomorrow)",
                          yaxis=dict(range=[
                              max(0.35, latest["direction_prob"].min() - 0.02),
                              min(0.65, latest["direction_prob"].max() + 0.02),
                          ]))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Top long / short candidates**")
        cl, cr = st.columns(2)
        with cl:
            st.write("Highest P(up):")
            st.dataframe(latest.nlargest(5, "direction_prob")[["ticker", "sector", "direction_prob"]]
                         .style.format({"direction_prob": "{:.3f}"}),
                         hide_index=True, use_container_width=True)
        with cr:
            st.write("Lowest P(up):")
            st.dataframe(latest.nsmallest(5, "direction_prob")[["ticker", "sector", "direction_prob"]]
                         .style.format({"direction_prob": "{:.3f}"}),
                         hide_index=True, use_container_width=True)
    else:
        st.dataframe(latest, use_container_width=True)

# --- Per-ticker IC if available ---
nlp_metrics = load_json("nlp_metrics.json")
if nlp_metrics.get("ic_per_ticker"):
    st.markdown(f"<h3>{icon('chart-bar', 22)} Per-ticker sentiment IC</h3>", unsafe_allow_html=True)
    ic_series = pd.Series(nlp_metrics["ic_per_ticker"]).sort_values()
    ic_df = ic_series.reset_index()
    ic_df.columns = ["ticker", "ic"]
    ic_df["sector"] = ic_df["ticker"].map(SECTOR_MAP)
    fig = px.bar(ic_df, x="ticker", y="ic", color="sector",
                 template="plotly_dark", height=320)
    fig.add_hline(y=0, line_color="white", opacity=0.5)
    fig.update_layout(xaxis_title=None, yaxis_title="Pearson correlation (previous trading day)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Synthetic news source — see the NLP page for the caveat. Overall IC = "
        f"{nlp_metrics.get('ic_lagged', float('nan')):.4f}."
    )
