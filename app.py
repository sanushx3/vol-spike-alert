"""Nifty 50 volatility spike warning (Streamlit app).

Run from the project root:
    streamlit run app.py

Needs these files from the notebooks:
    data/nifty50_features.csv
    models/lr_pipeline.joblib
    models/config.json
Optional (shown in the results tab):
    models/results_summary.csv
    images/threshold_tuning.png
"""
import base64
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:  # the app still runs with static charts
    HAS_PLOTLY = False

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "nifty50_features.csv"
MODEL_PATH = ROOT / "models" / "lr_pipeline.joblib"
CONFIG_PATH = ROOT / "models" / "config.json"
RESULTS_PATH = ROOT / "models" / "results_summary.csv"
FIGURE_PATH = ROOT / "images" / "threshold_tuning.png"
BACKGROUND_PATH = ROOT / "images" / "background.jpg"
PRICES_PATH = ROOT / "data" / "nifty50_prices.csv"   # optional clean prices
RAW_PATH = ROOT / "data" / "nifty50_raw.csv"         # fallback: the yfinance download
LOGO_PATH = ROOT / "images" / "logo.png"
ICON_PATH = ROOT / "images" / "icon.png"

# Same time split as the modeling notebook
TRAIN_END_YEAR = 2019
VAL_END_YEAR = 2021

# Last price in the raw download. The final day has no known "next day", so the
# last scored day is the one before it.
SNAPSHOT_END = "30 Oct 2026"

FEATURE_LABELS = {
    "feat_sma_return_5": "5-day average return",
    "feat_daily_range": "Today's high-low range",
    "feat_ema_ratio_5": "Close compared with 5-day trend",
    "feat_max_return_10": "Best day in the last 10",
    "feat_min_return_10": "Worst day in the last 10",
    "feat_vol_ratio": "Short-term vs 20-day volatility",
    "vol_20": "20-day volatility",
}

st.set_page_config(
    page_title="VolSpike | Nifty 50 spike warning",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else None,
    layout="wide",
)

# ---------------------------------------------------------------- look
# Minimal terminal-style palette: near-black page, thin lines, and green or red
# only where they carry meaning (green lowers the risk score, red raises it).
COLOR_UP = "#EF5350"      # raises the risk score, warning threshold
COLOR_DOWN = "#26A69A"    # lowers the risk score
COLOR_LINE = "#B2B5BE"    # risk score line
COLOR_SPIKE = "#F5C451"   # day before a real spike
PANEL = "#0B0E11"         # same as the page, so charts blend in
GRID = "#1E222D"
BORDER = "#2A2E39"
MUTED = "#787B86"
TEXT = "#D1D4DC"

# Bull and bear artwork (images/background.jpg) on the right of the page, with the
# content in a dark translucent column on the left. Set USE_ARTWORK = False for a
# plain dark page.
USE_ARTWORK = True
OVERLAY = 0.50      # 0 shows the artwork at full strength, 1 hides it
PANEL_LEFT = 0.92    # opacity of the content column at its left edge
PANEL_RIGHT = 0.62   # opacity at its right edge (lower shows more artwork behind it)

plt.rcParams.update(
    {
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "axes.edgecolor": BORDER,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.facecolor": PANEL,
        "legend.edgecolor": BORDER,
    }
)

if LOGO_PATH.exists():
    st.logo(
        str(LOGO_PATH),
        icon_image=str(ICON_PATH) if ICON_PATH.exists() else None,
        size="large",
    )

BASE_CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1200px; }
h1 { font-weight: 600; letter-spacing: -0.02em; font-size: 1.9rem; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; font-weight: 600; }
[data-testid="stMetricLabel"] p {
    text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.72rem; color: #787B86;
}
.ticker {
    display: flex; flex-wrap: wrap; align-items: center; gap: 1.6rem;
    padding: 0.55rem 0.9rem; margin: 0.4rem 0 1rem 0;
    border: 1px solid #2A2E39; border-radius: 6px;
    font-variant-numeric: tabular-nums;
}
.ticker .sym { font-weight: 700; letter-spacing: 0.06em; }
.ticker .lbl { color: #787B86; font-size: 0.72rem; letter-spacing: 0.08em; margin-right: 0.35rem; }
.ticker .chip {
    margin-left: auto; border: 1px solid; border-radius: 4px;
    padding: 0.1rem 0.55rem; font-size: 0.75rem; letter-spacing: 0.08em; font-weight: 600;
}
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)


@st.cache_data
def background_css(overlay: float, panel_left: float, panel_right: float) -> str:
    if not BACKGROUND_PATH.exists():
        return ""
    encoded = base64.b64encode(BACKGROUND_PATH.read_bytes()).decode()
    return f"""
    <style>
    .stApp {{
        background-color: #05070A;
        background-image: linear-gradient(rgba(5,7,10,{overlay}), rgba(5,7,10,{overlay})),
                          url("data:image/jpeg;base64,{encoded}");
        background-repeat: no-repeat;
        background-size: auto 100%;
        background-position: calc(100% + 25vw) center;
        background-attachment: fixed;
    }}
    [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: transparent; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stSidebar"] {{
        background-color: rgba(8,11,16,0.90);
        backdrop-filter: blur(4px);
    }}
    .block-container {{
        max-width: 980px;
        margin-left: 2rem;
        margin-right: auto;
        margin-top: 1rem;
        padding: 2rem 2.2rem 3rem 2.2rem;
        background: linear-gradient(90deg, rgba(8,11,16,{panel_left}), rgba(8,11,16,{panel_right}));
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        border-radius: 10px;
    }}
    </style>
    """


if USE_ARTWORK:
    artwork_css = background_css(OVERLAY, PANEL_LEFT, PANEL_RIGHT)
    if artwork_css:
        st.markdown(artwork_css, unsafe_allow_html=True)

missing = [p.name for p in (DATA_PATH, MODEL_PATH, CONFIG_PATH) if not p.exists()]
if missing:
    st.error(
        "Missing files: " + ", ".join(missing)
        + ". Run the notebooks and save their outputs first."
    )
    st.stop()


@st.cache_resource
def load_model():
    pipe = joblib.load(MODEL_PATH)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    return pipe, cfg


@st.cache_data
def load_scored_data(_pipe, feature_cols):
    df = pd.read_csv(DATA_PATH, parse_dates=["Dates"])
    df = df.sort_values("Dates").reset_index(drop=True)
    df["risk_score"] = _pipe.predict_proba(df[feature_cols])[:, 1]
    df["period"] = np.select(
        [df["Years"] <= TRAIN_END_YEAR, df["Years"] <= VAL_END_YEAR],
        ["Training period", "Validation period"],
        default="Test period",
    )
    return df


pipe, cfg = load_model()
feature_cols = cfg["feature_cols"]
threshold = float(cfg["best_threshold"])
df = load_scored_data(pipe, feature_cols)


@st.cache_data
def load_prices():
    """Daily Nifty 50 prices. Uses data/nifty50_prices.csv if present, otherwise
    the raw yfinance file (which has two extra header rows)."""
    try:
        if PRICES_PATH.exists():
            px = pd.read_csv(PRICES_PATH, parse_dates=["Dates"])
        elif RAW_PATH.exists():
            px = pd.read_csv(RAW_PATH, skiprows=[1, 2]).rename(columns={"Price": "Dates"})
            px["Dates"] = pd.to_datetime(px["Dates"])
            for col in ["Open", "High", "Low", "Close"]:
                px[col] = pd.to_numeric(px[col])
        else:
            return None
        return px[["Dates", "Close"]].sort_values("Dates").reset_index(drop=True)
    except Exception:
        return None


def prepare_spike_table(scored, prices, thr):
    """One row per spike. A row of `scored` with Target == 1 means a spike came on the
    NEXT trading day, so the spike date is the following date."""
    close = prices.set_index("Dates")["Close"]
    move = close.pct_change() * 100
    nxt = scored["Dates"].shift(-1)
    spk = scored[scored["Target"] == 1].copy()
    spk["spike_date"] = nxt.loc[spk.index].values
    spk = spk.dropna(subset=["spike_date"])
    spk["spike_close"] = close.reindex(spk["spike_date"]).values
    spk["spike_move"] = move.reindex(spk["spike_date"]).values
    spk = spk.dropna(subset=["spike_close", "spike_move"])
    spk["warned"] = np.where(spk["risk_score"] > thr, "Yes", "No")
    spk["spike_label"] = spk["spike_date"].dt.strftime("%d %b %Y")
    return spk


def build_market_figure(scored, prices, thr, picked_date, one_year, show_warnings):
    close = prices.set_index("Dates")["Close"]
    spk = prepare_spike_table(scored, prices, thr)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.0, row_heights=[0.64, 0.36]
    )

    # top panel: Nifty 50 close, spike days, optional warning days
    fig.add_trace(
        go.Scatter(
            x=prices["Dates"], y=prices["Close"], mode="lines",
            line=dict(color=COLOR_LINE, width=1.3), name="Nifty 50 close",
            hovertemplate="%{x|%d %b %Y}<br>Close %{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=spk["spike_date"], y=spk["spike_close"], mode="markers", name="Spike day",
            marker=dict(color=COLOR_UP, size=9, symbol="diamond", line=dict(color=PANEL, width=1)),
            customdata=list(zip(spk["spike_move"], spk["warned"], spk["risk_score"])),
            hovertemplate=(
                "<b>Spike on %{x|%d %b %Y}</b><br>Move %{customdata[0]:+.2f}%<br>"
                "Model warned the day before: %{customdata[1]}<br>"
                "Risk score the day before: %{customdata[2]:.2f}<extra></extra>"
            ),
        ),
        row=1, col=1,
    )
    if show_warnings:
        warn = scored[scored["risk_score"] > thr]
        outcome = np.where(warn["Target"] == 1, "Spike", "No spike")
        fig.add_trace(
            go.Scatter(
                x=warn["Dates"], y=close.reindex(warn["Dates"]).values, mode="markers",
                name="Warning issued",
                marker=dict(color=COLOR_SPIKE, size=4, opacity=0.85),
                customdata=list(zip(warn["risk_score"], outcome)),
                hovertemplate=(
                    "<b>Warning issued %{x|%d %b %Y}</b><br>Risk score %{customdata[0]:.2f}"
                    "<br>Next day: %{customdata[1]}<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

    # bottom panel: risk score, threshold, and the days before real spikes
    fig.add_trace(
        go.Scatter(
            x=scored["Dates"], y=scored["risk_score"], mode="lines",
            line=dict(color=COLOR_LINE, width=1), name="Risk score",
            hovertemplate="%{x|%d %b %Y}<br>Risk score %{y:.2f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=spk["Dates"], y=spk["risk_score"], mode="markers", name="Day before a spike",
            marker=dict(color=COLOR_UP, size=6, symbol="diamond"),
            customdata=list(zip(spk["spike_label"])),
            hovertemplate=(
                "<b>%{x|%d %b %Y}</b><br>Risk score %{y:.2f}"
                "<br>Spike followed on %{customdata[0]}<extra></extra>"
            ),
        ),
        row=2, col=1,
    )
    fig.add_hline(y=thr, line_dash="dash", line_color=COLOR_UP, line_width=1, row=2, col=1)

    # the day picked in the sidebar
    d = picked_date.strftime("%Y-%m-%d")
    fig.add_shape(
        type="line", x0=d, x1=d, y0=0, y1=1, xref="x", yref="paper",
        line=dict(color=TEXT, width=1, dash="dot"),
    )

    fig.update_layout(
        height=560, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=12), hovermode="closest", dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=BORDER)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=BORDER)
    fig.update_yaxes(title_text="Close", row=1, col=1)
    fig.update_yaxes(title_text="Risk score", range=[0, 1], row=2, col=1)
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor=PANEL, activecolor=BORDER, font=dict(color=TEXT),
            x=1, xanchor="right", y=1.04, yanchor="bottom",
        ),
        row=1, col=1,
    )
    if one_year:
        start = (picked_date - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        end = (picked_date + pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        fig.update_xaxes(range=[start, end])
    return fig

st.title("Nifty 50 volatility spike warning")
st.caption(
    "An educational model that scores how likely a large move is on the next "
    "trading day. It is not financial advice."
)
st.caption(
    f"Historical explorer, not a live forecast. The data is a fixed snapshot "
    f"that ends on {SNAPSHOT_END}, kept unchanged so the results can be "
    f"reproduced."
)

# ---------------------------------------------------------------- sidebar
min_date = df["Dates"].min().date()
max_date = df["Dates"].max().date()

st.sidebar.header("Pick a day")
selected = st.sidebar.date_input(
    "Trading day", value=max_date, min_value=min_date, max_value=max_date
)
chart_range = st.sidebar.radio(
    "Chart range", ["One year around the day", "Full history"]
)
st.sidebar.caption(f"Warning threshold: {threshold:.2f}")
st.sidebar.caption(f"Data: {min_date:%d %b %Y} to {max_date:%d %b %Y}")

sub = df[df["Dates"] <= pd.Timestamp(selected)]
if sub.empty:
    st.warning("There is no data on or before that date. Pick a later day.")
    st.stop()
row = sub.iloc[-1]
shown_date = row["Dates"]
if shown_date.date() != selected:
    st.sidebar.caption(
        f"No trading data on that date, so the latest earlier day is shown "
        f"({shown_date:%d %b %Y})."
    )

strip_score = float(row["risk_score"])
strip_warn = strip_score > threshold
strip_color = COLOR_UP if strip_warn else COLOR_DOWN
st.markdown(
    f"""
<div class="ticker">
  <span class="sym">NIFTY 50</span>
  <span><span class="lbl">DATE</span>{shown_date:%d %b %Y}</span>
  <span><span class="lbl">RISK</span>{strip_score:.2f}</span>
  <span><span class="lbl">THRESHOLD</span>{threshold:.2f}</span>
  <span class="chip" style="color:{strip_color}; border-color:{strip_color}">{"WARNING ON" if strip_warn else "WARNING OFF"}</span>
</div>
""",
    unsafe_allow_html=True,
)

tab_explorer, tab_try, tab_results, tab_about = st.tabs(
    ["Explore a day", "Try your own numbers", "How well it works", "About the model"]
)

# ---------------------------------------------------------------- explorer
with tab_explorer:
    score = float(row["risk_score"])
    warn = score > threshold
    actual = int(row["Target"])

    st.subheader(f"{shown_date:%A, %d %B %Y}")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Risk score",
        f"{score:.2f}",
        help="A ranking from 0 to 1. Higher means higher risk. It is not a true "
        "probability, because the model was trained with extra weight on spike days.",
    )
    c2.metric("Warning", "On" if warn else "Off", help=f"On when the score is above {threshold:.2f}.")
    c3.metric(
        "What happened next",
        "Spike" if actual == 1 else "No spike",
        help="Looks back at the real next trading day. A spike is a move above 2%, "
        "or above 2.5 times the recent 20-day volatility.",
    )

    if warn:
        st.warning(
            "The model sees a higher than usual chance of a large move on the "
            "next trading day."
        )
    else:
        st.success("No warning: the model sees no unusual risk for the next trading day.")

    period = row["period"]
    if period == "Training period":
        st.info(
            "This day is in the training period (up to 2019), so the model has "
            "already seen it. For a fair check, pick a day from 2022 onward."
        )
    elif period == "Validation period":
        st.caption(
            "Validation period (2020-2021): the model was not trained on these "
            "days, but the warning threshold was chosen using them."
        )
    else:
        st.caption(
            "Test period (2022 onward): not used for training or tuning, so "
            "this is a fair check."
        )

    left, right = st.columns(2)

    with left:
        st.markdown("**What the model saw**")
        values = row[feature_cols].astype(float)
        table = pd.DataFrame(
            {
                "Signal": [FEATURE_LABELS.get(c, c) for c in feature_cols],
                "Value": values.round(4).values,
            }
        )
        st.dataframe(table, hide_index=True)

    with right:
        st.markdown("**What pushed the score up or down**")
        scaler, clf = pipe[0], pipe[-1]
        x_row = row[feature_cols].to_frame().T.astype(float)
        z = scaler.transform(x_row)[0]
        contrib = pd.Series(z * clf.coef_[0], index=feature_cols).sort_values()
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        colors = [COLOR_UP if v > 0 else COLOR_DOWN for v in contrib.values]
        ax.barh([FEATURE_LABELS.get(c, c) for c in contrib.index], contrib.values, color=colors)
        ax.axvline(0, color=TEXT, lw=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Effect compared with an average day")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Red bars raise the risk score, blue bars lower it.")

    st.markdown("**Nifty 50 and risk score**")
    show_warnings = st.checkbox("Also mark the days the model warned", value=False)
    prices = load_prices()
    if HAS_PLOTLY and prices is not None:
        market_fig = build_market_figure(
            df, prices, threshold, shown_date,
            chart_range == "One year around the day", show_warnings,
        )
        st.plotly_chart(market_fig, config={"displaylogo": False, "scrollZoom": True})
        st.caption(
            "Hover or tap a red diamond to see a spike's date and size and whether the "
            "model warned the day before. Drag to move, scroll or pinch to zoom, and use "
            "the buttons for 1 month to all history."
        )
    else:
        if not HAS_PLOTLY:
            st.info("For the interactive chart, run: pip install plotly")
        else:
            st.info("Price data not found. Add data/nifty50_raw.csv or data/nifty50_prices.csv.")
        if chart_range == "One year around the day":
            start = shown_date - pd.Timedelta(days=365)
            end = shown_date + pd.Timedelta(days=365)
            win = df[(df["Dates"] >= start) & (df["Dates"] <= end)]
        else:
            win = df

        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.plot(win["Dates"], win["risk_score"], lw=1, color=COLOR_LINE, label="Risk score")
        ax.axhline(threshold, color=COLOR_UP, ls="--", lw=1, label=f"Warning threshold ({threshold:.2f})")
        spikes = win[win["Target"] == 1]
        ax.scatter(
            spikes["Dates"], spikes["risk_score"], s=14, color=COLOR_SPIKE,
            zorder=3, label="Day before a real spike",
        )
        ax.axvline(shown_date, color=TEXT, lw=1, alpha=0.6)
        ax.set_ylabel("Risk score")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

# ---------------------------------------------------------------- try it
with tab_try:
    st.subheader("Try your own numbers")
    st.write(
        "Change the seven signals and watch the risk score react. The boxes start "
        "with the values from the day picked in the sidebar. Pick a stressful day "
        "such as March 2020 to start from a high-risk setting."
    )
    st.caption(
        "Each box accepts values inside the range seen in the data, because the "
        "model cannot be trusted outside it. This shows how the model responds. "
        "It is not a validated forecast."
    )

    start_values = row[feature_cols].astype(float)
    typed = {}
    left_col, right_col = st.columns(2)
    for i, name in enumerate(feature_cols):
        lo = float(df[name].min())
        hi = float(df[name].max())
        with (left_col if i % 2 == 0 else right_col):
            typed[name] = st.number_input(
                FEATURE_LABELS.get(name, name),
                min_value=lo,
                max_value=hi,
                value=float(start_values[name]),
                step=(hi - lo) / 200,
                format="%.6f",
                key=f"try_{name}_{shown_date:%Y%m%d}",
            )

    x_try = pd.Series(typed)[feature_cols].to_frame().T.astype(float)
    score_try = float(pipe.predict_proba(x_try)[:, 1][0])
    base_score = float(row["risk_score"])
    warn_try = score_try > threshold

    m1, m2 = st.columns(2)
    m1.metric(
        "Risk score",
        f"{score_try:.2f}",
        delta=f"{score_try - base_score:+.2f} compared with the picked day",
        delta_color="inverse",
    )
    m2.metric("Warning", "On" if warn_try else "Off", help=f"On when the score is above {threshold:.2f}.")
    if warn_try:
        st.warning("With these numbers the model sees a higher than usual chance of a large move on the next trading day.")
    else:
        st.success("With these numbers the model sees no unusual risk for the next trading day.")

    st.markdown("**What pushed the score up or down**")
    z_try = pipe[0].transform(x_try)[0]
    contrib_try = pd.Series(z_try * pipe[-1].coef_[0], index=feature_cols).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bar_colors = [COLOR_UP if v > 0 else COLOR_DOWN for v in contrib_try.values]
    ax.barh([FEATURE_LABELS.get(c, c) for c in contrib_try.index], contrib_try.values, color=bar_colors)
    ax.axvline(0, color=TEXT, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Effect compared with an average day")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Red bars raise the risk score, blue bars lower it.")

    with st.expander("Where do these numbers come from?"):
        st.markdown(
            """
Every signal is worked out from daily Nifty 50 prices. A **daily return** is
today's close divided by yesterday's close, minus 1.

| Signal | How it is calculated |
|---|---|
| 20-day volatility | Standard deviation of the last 20 daily returns |
| 5-day average return | Average of the last 5 daily returns |
| Today's high-low range | (High - Low) divided by the close |
| Close compared with 5-day trend | Today's close divided by its 5-day exponential average |
| Best day in the last 10 | Highest of the last 10 daily returns |
| Worst day in the last 10 | Lowest of the last 10 daily returns |
| Short-term vs 20-day volatility | Standard deviation of the last 5 returns divided by that of the last 20 |

**Where to get prices.** Daily open, high, low and close for the Nifty 50 are
published on the NSE India website (historical index data) and on Yahoo Finance
under the symbol ^NSEI (Historical Data). Download the table, keep the last 21
trading days and work out the signals as above.

This app does not turn prices into signals for you. The boxes take the finished
numbers, which keeps the demo fast and shows exactly what the model reacts to.
"""
        )

# ---------------------------------------------------------------- results
with tab_results:
    st.subheader("Results on data the model did not train on")
    if RESULTS_PATH.exists():
        results = pd.read_csv(RESULTS_PATH)
        st.dataframe(results.round(3), hide_index=True)
    else:
        st.caption("Add models/results_summary.csv to show the full results table here.")

    # Numbers below come from the modeling notebook. Update them if you rerun it.
    st.markdown(
        """
- In the test period (2022 onward), about 4.9% of days were followed by a spike.
- With the warning threshold at 0.45, the model caught 18 of 58 test spikes and
  raised 108 false alarms.
- Its PR-AUC on the test period is 0.159, about three times what random warnings
  would score (0.049). That is better than the baselines, but still modest.
- A simple volatility rule worked on 2020-2021 but never fired in the calmer test
  period, because volatility there never reached the level learned in training.
"""
    )
    if FIGURE_PATH.exists():
        st.image(str(FIGURE_PATH), caption="Precision and recall at different thresholds (validation set)")

# ---------------------------------------------------------------- about
with tab_about:
    st.subheader("What this model does")
    st.markdown(
        """
It looks at the seven signals shown on the explorer tab, all calculated from
prices up to the end of the chosen day. It gives a risk score for the **next**
trading day.

**What counts as a spike.** The next day's move (up or down) is bigger than 2%,
or bigger than 2.5 times the recent 20-day volatility. About 9% of days in the
data are followed by a spike.

**How it was built.** Data is split by time: training up to 2019, validation in
2020-2021 and test from 2022. Scaling was fitted on training data only. The
warning threshold was picked on the validation set and then left unchanged for
the test set. The final model is logistic regression, chosen because it matched
a random forest on validation results and shows why a warning is raised.

**Limits to keep in mind**
- Only 58 spikes fall in the test period, so results are noisy.
- Volatility regimes change. The model was tuned partly on the 2020 crash and
  gives many false alarms in calmer markets.
- It uses only Nifty 50 prices. News, events and other markets are not included.
- Most warnings turn out to be false alarms. Treat the score as a prompt to look
  closer, not as a trade signal.

This is an educational project and not financial advice.
"""
    )
