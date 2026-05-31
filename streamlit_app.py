import streamlit as st
import pandas as pd
import altair as alt
import yfinance as yf
from datetime import datetime, timezone

st.set_page_config(page_title="Momentum Scanner", layout="wide")

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(ttl=300)
def load_universe():
    df = pd.read_csv("output/universe.csv")
    # Drop any header row that slipped through, deduplicate
    df = df[df["ticker"] != "Ticker"]
    df = df.drop_duplicates(subset="ticker", keep="first")
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df

@st.cache_data(ttl=300)
def fetch_live_scores(tickers):
    """
    Re-fetches premarket data for a short list of tickers right now.
    Returns a small dataframe with ticker, live_price, live_gap_pct,
    live_premarket_vol, and a simple live_score.
    """
    rows = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.fast_info
            hist = t.history(period="5d")
            if hist.empty:
                continue

            yesterday_close = float(hist["Close"].iloc[-1])
            avg_vol = float(hist["Volume"].mean())

            pre_price = getattr(info, "pre_market_price", None) or yesterday_close
            pre_vol   = getattr(info, "pre_market_volume", None) or 0

            gap_pct       = (pre_price - yesterday_close) / yesterday_close if yesterday_close else 0
            premarket_rvol = pre_vol / avg_vol if avg_vol else 0

            rows.append({
                "ticker":          ticker,
                "live_price":      round(pre_price, 2),
                "live_gap_pct":    round(gap_pct, 4),
                "live_pre_rvol":   round(premarket_rvol, 3),
                "yesterday_close": round(yesterday_close, 2),
            })
        except Exception:
            continue

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def traffic_light(score_morning, score_now):
    """
    Returns an emoji + label based on how much the score has changed.
    Green  = score held or improved.
    Amber  = dropped but still above 7.
    Red    = dropped significantly or below 7.
    """
    delta = score_now - score_morning
    if score_now >= score_morning * 0.95:
        return "🟢", "Still valid"
    elif score_now >= 7:
        return "🟡", "Fading"
    else:
        return "🔴", "Gone"


# ============================================================
# LOAD DATA
# ============================================================

df = load_universe()

st.title("📈 Momentum Scanner Dashboard")
st.caption("Automatically generated daily from your GitHub Actions pipeline")

tab1, tab2 = st.tabs(["Scanner", "Trade Today"])


# ============================================================
# TAB 1 — SCANNER
# ============================================================

with tab1:

    st.subheader("🏆 Top 10 Momentum Tickers")
    top10 = df.head(10)[[
        "ticker", "score", "gap_pct", "rvol", "premarket_rvol",
        "trend_5d", "breakout_score", "volatility_score"
    ]]
    st.dataframe(top10, use_container_width=True)

    st.subheader("🔥 Gap % vs Relative Volume (RVOL)")
    scatter = (
        alt.Chart(df.head(50))
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

    st.subheader("📋 Full Universe (Sortable)")
    st.dataframe(df, use_container_width=True)


# ============================================================
# TAB 2 — TRADE TODAY
# ============================================================

with tab2:

    st.header("🚀 Trade Today")

    # Morning candidates (score > 9, top 5)
    today_top = df[df["score"] > 9].head(5).copy()

    if today_top.empty:
        st.warning("No tickers above score 9 today. Check back once premarket is active (from around 13:00 BST).")
        st.stop()

    # ---- Morning snapshot table ----
    st.subheader("Morning candidates (13:00 scan)")
    display_cols = ["ticker", "score", "gap_pct", "premarket_rvol",
                    "rvol", "trend_5d", "breakout_score", "volatility_score"]
    st.dataframe(today_top[display_cols], use_container_width=True)

    st.info("Suggested entry time: **14:00–14:15 BST** — after the pre-trade confirmation check below.")

    # ---- Score comparison bar chart ----
    st.subheader("📊 Morning score comparison")
    score_chart = (
        alt.Chart(today_top)
        .mark_bar()
        .encode(
            x=alt.X("ticker:N", sort="-y"),
            y=alt.Y("score:Q"),
            color=alt.Color("score", scale=alt.Scale(scheme="redyellowgreen")),
            tooltip=["ticker", "score", "rvol", "trend_5d", "breakout_score"]
        )
    )
    st.altair_chart(score_chart, use_container_width=True)

    # ---- Pre-trade confirmation check ----
    st.divider()
    st.subheader("🔄 Pre-trade confirmation — is momentum still in play?")
    st.caption(
        "This fetches live data right now for your top candidates. "
        "Run this around 14:00–14:15 BST, about 15–30 minutes before the market opens."
    )

    if st.button("Refresh live data now"):
        tickers_to_check = today_top["ticker"].tolist()

        with st.spinner("Fetching live premarket data..."):
            live_df = fetch_live_scores(tickers_to_check)

        if live_df.empty:
            st.error("Could not fetch live data. yfinance may be rate-limited — try again in a minute.")
        else:
            # Merge morning data with live data
            merged = today_top[["ticker", "score", "gap_pct", "premarket_rvol", "breakout_score"]].merge(
                live_df, on="ticker", how="left"
            )

            # Estimate a live score (simplified — gap and premarket rvol are the live signals)
            merged["live_score"] = (
                3 * merged["live_gap_pct"].fillna(0)
                + 2 * merged["live_pre_rvol"].fillna(0)
                + 2 * merged["breakout_score"].fillna(0)
            )

            merged["score_delta"] = merged["live_score"] - merged["score"]

            # ---- Traffic light table ----
            st.subheader("Traffic light status")

            for _, row in merged.iterrows():
                light, label = traffic_light(row["score"], row["live_score"])
                delta_str = f"+{row['score_delta']:.2f}" if row["score_delta"] >= 0 else f"{row['score_delta']:.2f}"
                col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 3])
                col1.markdown(f"## {light}")
                col2.metric("Ticker", row["ticker"])
                col3.metric("Morning score", f"{row['score']:.2f}")
                col4.metric("Live score", f"{row['live_score']:.2f}", delta=delta_str)
                col5.markdown(f"**{label}**  \nGap: {row['live_gap_pct']*100:.2f}%  |  Pre-mkt RVOL: {row['live_pre_rvol']:.2f}x")

            st.divider()

            # ---- Before vs now comparison chart ----
            st.subheader("📊 Score: morning vs now")

            chart_data = pd.concat([
                merged[["ticker", "score"]].rename(columns={"score": "value"}).assign(when="Morning (13:00)"),
                merged[["ticker", "live_score"]].rename(columns={"live_score": "value"}).assign(when="Now")
            ])

            comparison_chart = (
                alt.Chart(chart_data)
                .mark_bar()
                .encode(
                    x=alt.X("ticker:N", title="Ticker"),
                    y=alt.Y("value:Q", title="Score"),
                    color=alt.Color("when:N",
                        scale=alt.Scale(domain=["Morning (13:00)", "Now"], range=["#4a9eff", "#ff7043"]),
                        legend=alt.Legend(title="When")
                    ),
                    xOffset="when:N",
                    tooltip=["ticker", "when", alt.Tooltip("value:Q", format=".2f")]
                )
            )
            st.altair_chart(comparison_chart, use_container_width=True)

            # ---- Gap % comparison ----
            st.subheader("📊 Gap %: morning vs now")

            gap_data = pd.concat([
                merged[["ticker", "gap_pct"]].rename(columns={"gap_pct": "value"}).assign(when="Morning (13:00)"),
                merged[["ticker", "live_gap_pct"]].rename(columns={"live_gap_pct": "value"}).assign(when="Now")
            ])

            gap_chart = (
                alt.Chart(gap_data)
                .mark_bar()
                .encode(
                    x=alt.X("ticker:N", title="Ticker"),
                    y=alt.Y("value:Q", title="Gap %", axis=alt.Axis(format=".1%")),
                    color=alt.Color("when:N",
                        scale=alt.Scale(domain=["Morning (13:00)", "Now"], range=["#4a9eff", "#ff7043"]),
                        legend=alt.Legend(title="When")
                    ),
                    xOffset="when:N",
                    tooltip=["ticker", "when", alt.Tooltip("value:Q", format=".2%")]
                )
            )
            st.altair_chart(gap_chart, use_container_width=True)

            # ---- Pre-market RVOL comparison ----
            st.subheader("📊 Premarket RVOL: morning vs now")

            rvol_data = pd.concat([
                merged[["ticker", "premarket_rvol"]].rename(columns={"premarket_rvol": "value"}).assign(when="Morning (13:00)"),
                merged[["ticker", "live_pre_rvol"]].rename(columns={"live_pre_rvol": "value"}).assign(when="Now")
            ])

            rvol_chart = (
                alt.Chart(rvol_data)
                .mark_bar()
                .encode(
                    x=alt.X("ticker:N", title="Ticker"),
                    y=alt.Y("value:Q", title="Premarket RVOL"),
                    color=alt.Color("when:N",
                        scale=alt.Scale(domain=["Morning (13:00)", "Now"], range=["#4a9eff", "#ff7043"]),
                        legend=alt.Legend(title="When")
                    ),
                    xOffset="when:N",
                    tooltip=["ticker", "when", alt.Tooltip("value:Q", format=".2f")]
                )
            )
            st.altair_chart(rvol_chart, use_container_width=True)

            # ---- Plain English summary ----
            st.divider()
            st.subheader("Summary")
            green = [r["ticker"] for _, r in merged.iterrows() if traffic_light(r["score"], r["live_score"])[0] == "🟢"]
            amber = [r["ticker"] for _, r in merged.iterrows() if traffic_light(r["score"], r["live_score"])[0] == "🟡"]
            red   = [r["ticker"] for _, r in merged.iterrows() if traffic_light(r["score"], r["live_score"])[0] == "🔴"]

            if green:
                st.success(f"**Still valid — momentum holding:** {', '.join(green)}")
            if amber:
                st.warning(f"**Fading — proceed with caution:** {', '.join(amber)}")
            if red:
                st.error(f"**Gone — momentum lost, skip these today:** {', '.join(red)}")

    else:
        st.info("Press the button above around 14:00–14:15 BST to run the pre-trade confirmation check.")

    st.success("Dashboard loaded successfully.")
