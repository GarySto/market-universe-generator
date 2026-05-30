import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Momentum Scanner", layout="wide")

# Load data
df = pd.read_csv("output/universe.csv")

st.title("📈 Momentum Scanner Dashboard")
st.caption("Automatically generated daily from your GitHub Actions pipeline")

# --- Top 10 Momentum Tickers ---
st.subheader("🏆 Top 10 Momentum Tickers")

top10 = df.head(10)[[
    "ticker", "score", "gap_pct", "rvol", "premarket_rvol",
    "trend_5d", "breakout_score", "volatility_score"
]]

st.dataframe(top10, use_container_width=True)

# --- Gap % vs RVOL Scatter ---
st.subheader("🔥 Gap % vs Relative Volume (RVOL)")

scatter_source = df.head(50)  # limit to top 50 for clarity

scatter = (
    alt.Chart(scatter_source)
    .mark_circle(size=60)
    .encode(
        x=alt.X("gap_pct", title="Gap %"),
        y=alt.Y("rvol", title="Relative Volume (RVOL)"),
        color=alt.Color("score", scale=alt.Scale(scheme="redyellowgreen")),
        tooltip=["ticker", "score", "gap_pct", "rvol", "premarket_rvol", "trend_5d", "breakout_score"],
    )
    .interactive()
)

st.altair_chart(scatter, use_container_width=True)

# --- Trend Strength ---
st.subheader("📊 Trend Strength (Last 5 Days)")

trend_chart = (
    alt.Chart(df.head(20))
    .mark_bar()
    .encode(
        x=alt.X("ticker:N", sort="-y"),
        y=alt.Y("trend_5d:Q"),
        color=alt.Color("trend_5d", scale=alt.Scale(scheme="blues")),
        tooltip=["ticker", "trend_5d"],
    )
)

st.altair_chart(trend_chart, use_container_width=True)

# --- Breakout Score ---
st.subheader("🚀 Breakout Score (0 = bottom, 1 = top of range)")

breakout_chart = (
    alt.Chart(df.head(20))
    .mark_bar()
    .encode(
        x=alt.X("ticker:N", sort="-y"),
        y=alt.Y("breakout_score:Q"),
        color=alt.Color("breakout_score", scale=alt.Scale(scheme="greens")),
        tooltip=["ticker", "breakout_score"],
    )
)

st.altair_chart(breakout_chart, use_container_width=True)

# --- Full Table ---
st.subheader("📋 Full Universe (Sortable)")
st.dataframe(df, use_container_width=True)

st.success("Dashboard loaded successfully.")

import pandas as pd
import altair as alt

tab1, tab2 = st.tabs(["Scanner", "Backtest"])

with tab2:
    st.header("📈 Backtest Results (Last 30 Days)")

    try:
        bt = pd.read_csv("output/backtest_results.csv")

        st.subheader("Summary")
        st.write(f"Total trades: {len(bt)}")
        st.write(f"Win rate: {100 * bt['hit_target'].mean():.2f}%")
        st.write(f"Average return: {100 * bt['return'].mean():.2f}%")

        st.subheader("Equity Curve")
        bt["equity"] = (1 + bt["return"]).cumprod()
        chart = alt.Chart(bt).mark_line().encode(
            x="date:T",
            y="equity:Q"
        )
        st.altair_chart(chart, use_container_width=True)

        st.subheader("Distribution of Returns")
        hist = alt.Chart(bt).mark_bar().encode(
            x=alt.X("return:Q", bin=True),
            y="count()"
        )
        st.altair_chart(hist, use_container_width=True)

    except Exception as e:
        st.warning("Backtest results not found. Run backtest.py first.")
        st.error(str(e))
