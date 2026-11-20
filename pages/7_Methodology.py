"""Methodology page — limitations, integrity checks, test status."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.settings import (
    STOCK_UNIVERSE, TICKERS, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR,
    TRANSACTION_COST_BPS, SLIPPAGE_BPS,
    OPT_LONG_ONLY, OPT_MAX_WEIGHT, OPT_RISK_AVERSION, CVAR_ALPHA,
    GARCH_P, GARCH_Q, N_REGIMES, ROLLING_GRAPH_WINDOW,
    EMBEDDING_MODEL, FINBERT_MODEL, RANDOM_SEED, START_DATE, END_DATE,
)

LUCIDE = "https://cdn.jsdelivr.net/npm/lucide-static@latest/icons"
def icon(name, size=18):
    return (f'<img src="{LUCIDE}/{name}.svg" width="{size}" height="{size}" '
            f'align="absmiddle" style="vertical-align:middle;margin-right:6px;">')

st.set_page_config(page_title="Methodology · AxiomAlpha", layout="wide")
st.markdown(f"<h1>{icon('file-text', 28)} Methodology &amp; integrity</h1>",
            unsafe_allow_html=True)

# --- Limitations ---
st.markdown(f"<h3>{icon('flag-triangle-right', 22)} Limitations</h3>", unsafe_allow_html=True)
st.markdown("""
**Data and sentiment**

- The historical universe is a fixed selection of 30 US stocks and is subject to selection and survivorship bias.
- News is synthetic and partially conditioned on same-day returns. The scoring method is recorded in `sentiment_provenance.json`; sentiment is excluded from the portfolio.
- Market downloads and model availability can change between runs. Saved raw inputs have not been independently refreshed.

**Portfolio and evaluation**

- The optimizer and backtest share the same point-in-time strategy. Direction probabilities are out-of-sample but their return tilt is an uncalibrated assumption. Models refit at five fold boundaries, not every monthly rebalance; a fresh signal can come from an older model.
- Rolling volatility controls cash allocation; comparisons do not automatically match risk or market exposure.
- Estimated transaction costs use turnover and a simple impact term. Initial investment costs and a separate liquidity model are not included.
- There is no CVaR optimization constraint. The reported CVaR is an ex-post risk metric.

**Reproducibility**

- Results depend on the frozen inputs, software environment and model artifacts. Determinism should be verified, not assumed.
- The static notebook check is a limited pattern check. Tests and the determinism check must be run separately.
""")

st.divider()

# --- Active config ---
st.markdown(f"<h3>{icon('settings', 22)} Active configuration</h3>", unsafe_allow_html=True)
cfg = {
    "Universe": f"{len(TICKERS)} tickers across {len(STOCK_UNIVERSE)} sectors",
    "Date range": f"{START_DATE} → {END_DATE}",
    "Random seed": RANDOM_SEED,
    "Risk-free rate (annual)": f"{RISK_FREE_RATE:.1%}",
    "Trading days / year": TRADING_DAYS_PER_YEAR,
    "Transaction cost (bps, linear)": TRANSACTION_COST_BPS,
    "Slippage coef (bps, sqrt impact)": SLIPPAGE_BPS,
    "Long-only": OPT_LONG_ONLY,
    "Max weight per asset": f"{OPT_MAX_WEIGHT:.1%}",
    "GARCH (p, q)": f"({GARCH_P}, {GARCH_Q})",
    "Regimes": N_REGIMES,
    "Rolling graph window (days)": ROLLING_GRAPH_WINDOW,
    "Embedding model": EMBEDDING_MODEL,
    "FinBERT model": FINBERT_MODEL,
}
import pandas as pd
st.dataframe(
    pd.DataFrame([(k, str(v)) for k, v in cfg.items()], columns=["Setting", "Value"]),
    hide_index=True, use_container_width=True,
)

st.divider()

# --- Integrity checks ---
st.markdown(f"<h3>{icon('shield-check', 22)} Integrity checks</h3>", unsafe_allow_html=True)
st.caption(
    "Run-on-demand wrappers around the gates that protect the pipeline. "
    "Only the smoke check runs automatically with `python run_all.py` by default."
)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Run smoke check", type="primary", use_container_width=True):
        with st.spinner("python scripts/check_pipeline.py ..."):
            result = subprocess.run(
                [sys.executable, "scripts/check_pipeline.py"],
                cwd=str(ROOT), capture_output=True, text=True,
            )
        if result.returncode == 0:
            st.success("Smoke check passed.")
        else:
            st.error("Smoke check failed.")
        with st.expander("stdout"):
            st.code(result.stdout or "(empty)")
        if result.stderr:
            with st.expander("stderr"):
                st.code(result.stderr)

with col2:
    if st.button("Run pytest", use_container_width=True):
        with st.spinner("python -m pytest -q ..."):
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--no-header"],
                cwd=str(ROOT), capture_output=True, text=True,
            )
        if result.returncode == 0:
            st.success("All tests passed.")
        else:
            st.error("Some tests failed.")
        with st.expander("stdout"):
            st.code(result.stdout or "(empty)")

with col3:
    st.button("Run determinism check (slow)", use_container_width=True,
              help="Runs NB02→NB15 twice and compares byte hashes. ~20 min.",
              disabled=True)
    st.caption("Disabled in-app — run it from a shell: "
               "`python scripts/check_determinism.py --direct`")

st.divider()

# --- Files on disk ---
st.markdown(f"<h3>{icon('hard-drive', 22)} Pipeline artifacts on disk</h3>",
            unsafe_allow_html=True)
proc = ROOT / "data" / "processed"
if proc.exists():
    rows = []
    for f in sorted(proc.iterdir()):
        if f.is_file():
            rows.append({
                "File": f.name,
                "Size (KB)": round(f.stat().st_size / 1024, 1),
                "Modified": pd.Timestamp(f.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M"),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=400)
    else:
        st.warning("`data/processed/` exists but is empty. Run `python run_all.py`.")
else:
    st.warning("`data/processed/` not found.")

st.divider()
st.caption(
    "Reproducibility depends on data and environment; it has not been established by the smoke check. The README and "
    "`data/processed/RESULTS.md` are the canonical references — this app "
    "surfaces them but does not duplicate any numbers."
)
