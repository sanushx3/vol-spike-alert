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
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "nifty50_features.csv"
MODEL_PATH = ROOT / "models" / "lr_pipeline.joblib"
CONFIG_PATH = ROOT / "models" / "config.json"
RESULTS_PATH = ROOT / "models" / "results_summary.csv"
FIGURE_PATH = ROOT / "images" / "threshold_tuning.png"

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

st.set_page_config(page_title="Nifty 50 spike warning", layout="wide")

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
        colors = ["#c0392b" if v > 0 else "#2c6fbb" for v in contrib.values]
        ax.barh([FEATURE_LABELS.get(c, c) for c in contrib.index], contrib.values, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Effect compared with an average day")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Red bars raise the risk score, blue bars lower it.")

    st.markdown("**Risk score over time**")
    if chart_range == "One year around the day":
        start = shown_date - pd.Timedelta(days=365)
        end = shown_date + pd.Timedelta(days=365)
        win = df[(df["Dates"] >= start) & (df["Dates"] <= end)]
    else:
        win = df

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(win["Dates"], win["risk_score"], lw=1, color="#2c6fbb", label="Risk score")
    ax.axhline(threshold, color="#c0392b", ls="--", lw=1, label=f"Warning threshold ({threshold:.2f})")
    spikes = win[win["Target"] == 1]
    ax.scatter(
        spikes["Dates"], spikes["risk_score"], s=14, color="#e69f00",
        zorder=3, label="Day before a real spike",
    )
    ax.axvline(shown_date, color="black", lw=1, alpha=0.6)
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
    bar_colors = ["#c0392b" if v > 0 else "#2c6fbb" for v in contrib_try.values]
    ax.barh([FEATURE_LABELS.get(c, c) for c in contrib_try.index], contrib_try.values, color=bar_colors)
    ax.axvline(0, color="black", lw=0.8)
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
