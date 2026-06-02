import streamlit as st
import pandas as pd
import altair as alt
import yfinance as yf
import numpy as np
from datetime import datetime, date, timedelta, timezone

st.set_page_config(page_title="Momentum Scanner", layout="wide")

# ============================================================
# GLOSSARY (shared across tabs via expander)
# ============================================================

GLOSSARY = {
    "Score": "The original momentum score — a weighted combination of all the signals below.",
    "Score (long-only)": "A version of the score that only rewards positive gaps. If gap_pct is negative, this is set to 0 so sell-offs never appear as long candidates.",
    "Gap % (gap_pct)": "How much the premarket price has moved compared to yesterday's closing price, expressed as a fraction. A gap of 0.05 means the stock is up 5% before the market opens.",
    "RVOL (rvol)": "Relative Volume — yesterday's trading volume divided by the 10-day average volume.",
    "Premarket RVOL (premarket_rvol)": "Relative volume for premarket trading only. A fresher signal than regular RVOL.",
    "Premarket momentum score (premarket_momentum_score)": "A proxy for premarket run-up: positive gaps with premarket volume get a higher value.",
    "Trend 5d (trend_5d)": "How many of the last 5 trading days closed higher than they opened (0–5).",
    "Breakout Score (breakout_score)": "Where yesterday's close sits within the 10-day high/low range (0 = bottom, 1 = top).",
    "Volatility Score (volatility_score)": "ATR-based measure of how much the stock typically moves in a day.",
    "ATR (atr_10d)": "Average True Range over the last 10 days.",
    "BST": "British Summer Time — UTC+1, which is the timezone the dashboard uses.",
    "ET": "Eastern Time — US market timezone. 14:30 BST = 09:30 ET.",
    "Premarket": "The trading session before the official US market open at 14:30 BST.",
}


def show_glossary():
    with st.expander("📖 What do these terms mean? (click to expand)"):
        for term, definition in GLOSSARY.items():
            st.markdown(f"**{term}** — {definition}")


# ============================================================
# HELPERS & DATA LOADING
# ============================================================

@st.cache_data(ttl=300)
def load_universe():
    df = pd.read_csv("output/universe.csv")
    df = df[df["ticker"] != "Ticker"]
    df = df.drop_duplicates(subset="ticker", keep="first")

    # Backwards compatibility: older CSVs won't have these columns
    if "score_long" not in df.columns:
        df["score_long"] = df["score"]
    if "premarket_momentum_score" not in df.columns:
        df["premarket_momentum_score"] = 0.0
    if "direction" not in df.columns:
        df["direction"] = np.where(df["gap_pct"] > 0, "up", "down")

    # Sort by long-only score by default
    df = df.sort_values("score_long", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(ttl=300)
def fetch_live_scores(tickers):
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
            pre_vol = getattr(info, "pre_market_volume", None) or 0
            gap_pct = (pre_price - yesterday_close) / yesterday_close if yesterday_close else 0
            premarket_rvol = pre_vol / avg_vol if avg_vol else 0

            # Live premarket momentum proxy
            pre_momo = max(gap_pct, 0) * (1 + premarket_rvol)

            rows.append(
                {
                    "ticker": ticker,
                    "live_price": round(pre_price, 2),
                    "live_gap_pct": round(gap_pct, 4),
                    "live_pre_rvol": round(premarket_rvol, 3),
                    "live_premarket_momentum_score": round(pre_momo, 4),
                    "yesterday_close": round(yesterday_close, 2),
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def traffic_light(score_morning, score_now):
    if score_now >= score_morning * 0.95:
        return "🟢", "Still valid"
    elif score_now >= 7:
        return "🟡", "Fading"
    else:
        return "🔴", "Gone"


def get_trading_days(n=14):
    """Return the last n weekdays (Mon–Fri) as date objects, most recent first."""
    days = []
    d = date.today()
    while len(days) < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            days.append(d)
    return days


@st.cache_data(ttl=3600, show_spinner=False)
def score_one_day(target_date_str, tickers):
    """
    Score tickers as-of a given historical date using a real gap proxy.

    gap_pct is estimated as: (today's open - yesterday's close) / yesterday's close.
    """
    target_date = date.fromisoformat(target_date_str)
    end_dt = datetime.combine(target_date + timedelta(days=1), datetime.min.time())
    start_dt = datetime.combine(target_date, datetime.min.time()) - timedelta(days=20)

    records = []
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            hist = ticker.history(start=start_dt, end=end_dt)
            if hist.empty or len(hist) < 11:
                continue

            today_row = hist.iloc[-1]
            yesterday_row = hist.iloc[-2]

            yesterday_close = float(yesterday_row["Close"])
            today_open = float(today_row["Open"])
            today_volume = float(today_row["Volume"])

            gap_pct = (today_open - yesterday_close) / yesterday_close if yesterday_close else 0

            hist_before = hist.iloc[:-1]
            last_10 = hist_before.tail(10)
            avg_volume_10d = float(last_10["Volume"].mean())
            high_10d = float(last_10["High"].max())
            low_10d = float(last_10["Low"].min())
            atr_10d = float((last_10["High"] - last_10["Low"]).mean())
            trend_5d = int((hist_before["Close"].diff() > 0).tail(5).sum())

            rvol = today_volume / avg_volume_10d if avg_volume_10d else 0

            breakout_score = (
                (yesterday_close - low_10d) / (high_10d - low_10d)
                if high_10d != low_10d
                else 0
            )
            volatility_score = atr_10d / yesterday_close if yesterday_close else 0

            score = (
                3 * gap_pct
                + 2 * rvol
                + 0.5 * trend_5d
                + 2 * breakout_score
                + 1 * volatility_score
            )

            records.append(
                {
                    "ticker": t,
                    "score": round(score, 4),
                    "gap_pct": round(gap_pct, 4),
                    "rvol": round(rvol, 4),
                    "premarket_rvol": 0.0,
                    "trend_5d": trend_5d,
                    "breakout_score": round(breakout_score, 4),
                    "volatility_score": round(volatility_score, 4),
                    "yesterday_close": round(yesterday_close, 2),
                    "today_open": round(today_open, 2),
                }
            )
        except Exception:
            continue

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_intraday(ticker_str, trade_date_str):
    """
    Fetch 1-minute candle data, 12:00–15:00 BST.
    prepost=True captures premarket candles where available.
    """
    import pytz

    trade_date = date.fromisoformat(trade_date_str)
    bst = pytz.timezone("Europe/London")
    start = datetime.combine(trade_date, datetime.min.time())
    end = start + timedelta(days=1)
    try:
        ticker_obj = yf.Ticker(ticker_str)
        df = ticker_obj.history(
            start=start,
            end=end,
            interval="1m",
            prepost=True,
            auto_adjust=True,
        )
        if df.empty:
            df = yf.download(
                ticker_str,
                start=start,
                end=end,
                interval="1m",
                prepost=True,
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                return pd.DataFrame()
            df = df.reset_index()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        else:
            df = df.reset_index()

        df.columns = [str(c).strip() for c in df.columns]
        time_col = next(
            (
                c
                for c in df.columns
                if any(k in c.lower() for k in ["datetime", "date", "timestamp"])
            ),
            df.columns[0],
        )
        df = df.rename(
            columns={
                time_col: "datetime",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = 0

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

        window_start = bst.localize(
            datetime(trade_date.year, trade_date.month, trade_date.day, 12, 0)
        ).astimezone(pytz.utc)
        window_end = bst.localize(
            datetime(trade_date.year, trade_date.month, trade_date.day, 15, 0)
        ).astimezone(pytz.utc)
        df = df[(df["datetime"] >= window_start) & (df["datetime"] < window_end)].copy()
        if df.empty:
            return pd.DataFrame()

        df = df.reset_index(drop=True)
        df["bst_time"] = df["datetime"].apply(
            lambda x: x.astimezone(bst).strftime("%H:%M")
        )

        entry_ref = bst.localize(
            datetime(trade_date.year, trade_date.month, trade_date.day, 13, 30)
        ).astimezone(pytz.utc)
        market_open = bst.localize(
            datetime(trade_date.year, trade_date.month, trade_date.day, 14, 30)
        ).astimezone(pytz.utc)
        df["mins_from_entry"] = df["datetime"].apply(
            lambda x: int((x - entry_ref).total_seconds() / 60)
        )
        df["phase"] = df["datetime"].apply(
            lambda x: "post-open" if x >= market_open else "premarket"
        )
        return df
    except Exception:
        return pd.DataFrame()


def simulate_trade(candles, target_pct=0.10):
    """
    Entry: open of 13:30 BST candle.
    Exit: first candle where high >= +10%, or forced close at 14:45 BST.
    """
    entry_candidates = candles[candles["mins_from_entry"] >= 0]
    if entry_candidates.empty:
        return {}
    entry_row = entry_candidates.iloc[0]
    entry_price = float(entry_row["open"])
    entry_time = entry_row["bst_time"]
    target_price = entry_price * (1 + target_pct)

    result = {
        "entry_price": entry_price,
        "entry_time": entry_time,
        "target_price": round(target_price, 4),
        "exit_price": None,
        "exit_time": None,
        "exit_bst": None,
        "hit_target": False,
        "pct_return": None,
    }

    trade_window = candles[
        (candles["mins_from_entry"] >= 0) & (candles["mins_from_entry"] <= 75)
    ]
    for _, row in trade_window.iterrows():
        mins = int(row["mins_from_entry"])
        if float(row["high"]) >= target_price:
            result.update(
                {
                    "exit_price": round(target_price, 4),
                    "exit_time": row["bst_time"],
                    "exit_bst": row["bst_time"],
                    "hit_target": True,
                }
            )
            break
        if mins >= 75:
            result.update(
                {
                    "exit_price": round(float(row["close"]), 4),
                    "exit_time": row["bst_time"],
                    "exit_bst": row["bst_time"],
                    "hit_target": False,
                }
            )
            break

    if result["exit_price"] is not None:
        result["pct_return"] = round(
            (result["exit_price"] - entry_price) / entry_price * 100, 2
        )
    return result


# ============================================================
# LOAD DATA
# ============================================================

df = load_universe()

st.title("📈 Momentum Scanner Dashboard")
st.caption("Automatically generated daily from your GitHub Actions pipeline")

tab1, tab2, tab3, tab4 = st.tabs(["Scanner", "Trade Today", "Backtest", "My Trades"])

# ============================================================
# TAB 1 — SCANNER
# ============================================================

with tab1:
    st.markdown(
        "This tab shows today's full ranked universe of stocks — built each morning at 13:00 BST "
        "by the automated pipeline. The top of the list is where to look first. "
        "Anything scoring above 9 on the **long-only score** with a meaningful gap and high premarket RVOL "
        "is worth investigating further."
    )
    show_glossary()
    st.divider()

    st.subheader("🏆 Top 10 Momentum Tickers (Long-only)")
    st.caption(
        "Filtered to tickers with a **positive premarket gap** and sorted by the long-only score. "
        "This removes NAMM-style sell-offs where RVOL is high but price is moving down."
    )

    long_only = df[df["gap_pct"] > 0].copy()
    top10 = (
        long_only.sort_values("score_long", ascending=False)
        .head(10)[
            [
                "ticker",
                "score_long",
                "score",
                "gap_pct",
                "rvol",
                "premarket_rvol",
                "premarket_momentum_score",
                "trend_5d",
                "breakout_score",
                "volatility_score",
            ]
        ]
    )
    st.dataframe(top10, use_container_width=True)

    gap_values = df["gap_pct"].dropna()
    has_gap_data = bool((gap_values.abs() > 0.001).any())

    if has_gap_data:
        st.subheader("🔥 Gap % vs Relative Volume (RVOL)")
        st.caption(
            "Each dot is one ticker. Top-right corner = gapping up AND high volume — "
            "the strongest combination for this strategy. Colour = long-only score."
        )
        scatter = (
            alt.Chart(df[df["gap_pct"].abs() > 0.001].head(50))
            .mark_circle(size=80)
            .encode(
                x=alt.X(
                    "gap_pct:Q",
                    title="Gap % (premarket vs yesterday's close)",
                    axis=alt.Axis(format=".1%"),
                ),
                y=alt.Y("rvol:Q", title="RVOL (volume vs 10-day average)"),
                color=alt.Color(
                    "score_long:Q",
                    scale=alt.Scale(scheme="redyellowgreen"),
                    legend=alt.Legend(title="Score (long-only)"),
                ),
                tooltip=[
                    "ticker",
                    "score_long",
                    "score",
                    "gap_pct",
                    "rvol",
                    "premarket_rvol",
                    "premarket_momentum_score",
                    "trend_5d",
                    "breakout_score",
                ],
            )
            .interactive()
        )
        st.altair_chart(scatter, use_container_width=True)
    else:
        st.subheader("📊 RVOL vs Breakout Score")
        st.info(
            "**Gap % data isn't available yet** — it populates when the 13:00 BST scan runs during premarket hours. "
            "Until then this chart shows RVOL vs Breakout Score."
        )
        scatter = (
            alt.Chart(df.head(50))
            .mark_circle(size=80)
            .encode(
                x=alt.X(
                    "breakout_score:Q",
                    title="Breakout score (0 = bottom of recent range, 1 = top)",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                y=alt.Y("rvol:Q", title="RVOL (volume vs 10-day average)"),
                color=alt.Color(
                    "score_long:Q",
                    scale=alt.Scale(scheme="redyellowgreen"),
                    legend=alt.Legend(title="Score (long-only)"),
                ),
                tooltip=[
                    "ticker",
                    "score_long",
                    "score",
                    "breakout_score",
                    "rvol",
                    "trend_5d",
                    "volatility_score",
                ],
            )
            .interactive()
        )
        st.altair_chart(scatter, use_container_width=True)

    st.subheader("📊 Trend Strength (Last 5 Days)")
    trend_chart = (
        alt.Chart(df.head(20))
        .mark_bar()
        .encode(
            x=alt.X("ticker:N", sort="-y", title="Ticker"),
            y=alt.Y("trend_5d:Q", title="Days up (out of 5)"),
            color=alt.Color(
                "trend_5d",
                scale=alt.Scale(scheme="blues"),
                legend=alt.Legend(title="Days up"),
            ),
            tooltip=["ticker", "trend_5d"],
        )
    )
    st.altair_chart(trend_chart, use_container_width=True)

    st.subheader("🚀 Breakout Score (0 = bottom of range, 1 = top of range)")
    breakout_chart = (
        alt.Chart(df.head(20))
        .mark_bar()
        .encode(
            x=alt.X("ticker:N", sort="-y", title="Ticker"),
            y=alt.Y("breakout_score:Q", title="Breakout score (0–1)"),
            color=alt.Color(
                "breakout_score",
                scale=alt.Scale(scheme="greens"),
                legend=alt.Legend(title="Score"),
            ),
            tooltip=["ticker", "breakout_score"],
        )
    )
    st.altair_chart(breakout_chart, use_container_width=True)

    st.subheader("📋 Full Universe (Sortable)")
    st.caption(
        "Includes both up-gap and down-gap names. Use the 'direction' column to see sell-offs vs movers."
    )
    st.dataframe(df, use_container_width=True)

# ============================================================
# TAB 2 — TRADE TODAY
# ============================================================

with tab2:
    st.header("🚀 Trade Today")
    st.markdown(
        "This tab filters the morning scan down to your top **long-only** candidates — "
        "stocks with a **positive premarket gap**, **premarket volume**, and a strong long-only score. "
        "The strategy is to buy in premarket between 13:30 and 14:15 BST, target a **+10% return**, "
        "and exit no later than **14:45 BST**."
    )
    show_glossary()
    st.divider()

    has_gap_data = bool((df["gap_pct"].abs() > 0.001).any())

    if has_gap_data:
        # “True premarket run-up” filter — tweak thresholds as you learn
        today_top = df[
            (df["gap_pct"] > 0.03)  # 3%+ gap
            & (df["premarket_rvol"] > 1.5)  # at least 1.5x premarket volume
            & (df["score_long"] > 10)
        ].copy()
        today_top = today_top.sort_values("score_long", ascending=False).head(5)
    else:
        today_top = pd.DataFrame()

    if today_top.empty:
        if not has_gap_data:
            st.warning(
                "No premarket gap data available yet — all gap_pct values are showing as 0. "
                "This happens when the scan runs before meaningful premarket activity has built up. "
                "Check back after 13:00 BST when the next scan runs."
            )
        else:
            st.warning(
                "No tickers meet the **long-only premarket momentum criteria** today "
                "(gap > 3%, premarket RVOL > 1.5x, score_long > 10). "
                "That means the market isn't showing the pattern this strategy hunts for. "
                "The correct move is to sit out and wait for tomorrow."
            )
    else:
        st.subheader("Morning candidates (13:00 scan — long-only)")
        st.caption(
            "These are the stocks with a positive premarket gap, elevated premarket volume, "
            "and a strong long-only score. NAMM-style down-moves with high RVOL are excluded by design."
        )
        display_cols = [
            "ticker",
            "score_long",
            "score",
            "gap_pct",
            "premarket_rvol",
            "premarket_momentum_score",
            "rvol",
            "trend_5d",
            "breakout_score",
            "volatility_score",
        ]
        st.dataframe(today_top[display_cols], use_container_width=True)
        st.info(
            "**Suggested entry window: 13:30–14:15 BST** — during the premarket session. "
            "Exit target: +10% from your entry price, or close the position by 14:45 BST."
        )

        st.subheader("📊 Morning score comparison (long-only)")
        score_chart = (
            alt.Chart(today_top)
            .mark_bar()
            .encode(
                x=alt.X("ticker:N", sort="-y", title="Ticker"),
                y=alt.Y("score_long:Q", title="Momentum score (long-only)"),
                color=alt.Color(
                    "score_long",
                    scale=alt.Scale(scheme="redyellowgreen"),
                    legend=alt.Legend(title="Score (long-only)"),
                ),
                tooltip=[
                    "ticker",
                    "score_long",
                    "score",
                    "gap_pct",
                    "premarket_rvol",
                    "premarket_momentum_score",
                    "rvol",
                    "trend_5d",
                    "breakout_score",
                ],
            )
        )
        st.altair_chart(score_chart, use_container_width=True)

        st.divider()
        st.subheader("🔄 Pre-trade confirmation — is momentum still in play?")
        st.markdown(
            "Run this around **14:00–14:15 BST**. It fetches the latest premarket data and recomputes a "
            "**live long-only score**. If the gap has flipped negative, the live score is forced to 0."
        )

        if st.button("Refresh live data now"):
            tickers_to_check = today_top["ticker"].tolist()
            with st.spinner("Fetching live premarket data..."):
                live_df = fetch_live_scores(tickers_to_check)

            if live_df.empty:
                st.error(
                    "Could not fetch live data. yfinance may be rate-limited — try again in a minute."
                )
            else:
                merged = today_top[
                    [
                        "ticker",
                        "score_long",
                        "score",
                        "gap_pct",
                        "premarket_rvol",
                        "breakout_score",
                        "premarket_momentum_score",
                    ]
                ].merge(live_df, on="ticker", how="left")

                # Live long-only score: if live_gap <= 0, score_long_live = 0
                live_gap = merged["live_gap_pct"].fillna(0)
                live_pre_rvol = merged["live_pre_rvol"].fillna(0)
                live_pre_momo = merged["live_premarket_momentum_score"].fillna(0)
                breakout = merged["breakout_score"].fillna(0)

                base_live = (
                    5 * live_gap
                    + 1.5 * merged["rvol"].fillna(0)
                    + 2 * breakout
                    + 2 * live_pre_momo
                )
                merged["live_score_long"] = np.where(live_gap > 0, base_live, 0.0)
                merged["score_delta"] = merged["live_score_long"] - merged["score_long"]

                st.subheader("Traffic light status")
                for _, row in merged.iterrows():
                    light, label = traffic_light(
                        row["score_long"], row["live_score_long"]
                    )
                    delta_val = row["score_delta"]
                    delta_str = (
                        f"+{delta_val:.2f}" if delta_val >= 0 else f"{delta_val:.2f}"
                    )
                    col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 3])
                    col1.markdown(f"## {light}")
                    col2.metric("Ticker", row["ticker"])
                    col3.metric("Morning long-only", f"{row['score_long']:.2f}")
                    col4.metric("Live long-only", f"{row['live_score_long']:.2f}", delta=delta_str)
                    col5.markdown(
                        f"**{label}**  \nGap: {row['live_gap_pct']*100:.2f}%  |  Pre-mkt RVOL: {row['live_pre_rvol']:.2f}x"
                    )

                st.divider()

                for metric, morning_col, live_col, fmt, title, caption in [
                    (
                        "Score (long-only)",
                        "score_long",
                        "live_score_long",
                        ".2f",
                        "📊 Long-only score: morning vs now",
                        "Blue = the 13:00 scan score. Orange = the live score right now.",
                    ),
                    (
                        "Gap %",
                        "gap_pct",
                        "live_gap_pct",
                        ".2%",
                        "📊 Gap %: morning vs now",
                        "If the gap has shrunk or flipped negative, enthusiasm is waning.",
                    ),
                    (
                        "Premarket RVOL",
                        "premarket_rvol",
                        "live_pre_rvol",
                        ".2f",
                        "📊 Premarket RVOL: morning vs now",
                        "Is trading activity increasing or decreasing since 13:00?",
                    ),
                ]:
                    st.subheader(title)
                    st.caption(caption)
                    chart_data = pd.concat(
                        [
                            merged[["ticker", morning_col]]
                            .rename(columns={morning_col: "value"})
                            .assign(when="Morning (13:00)"),
                            merged[["ticker", live_col]]
                            .rename(columns={live_col: "value"})
                            .assign(when="Now"),
                        ]
                    )
                    c = (
                        alt.Chart(chart_data)
                        .mark_bar()
                        .encode(
                            x=alt.X("ticker:N", title="Ticker"),
                            y=alt.Y(
                                "value:Q",
                                axis=alt.Axis(format=fmt if "%" in fmt else None),
                            ),
                            color=alt.Color(
                                "when:N",
                                scale=alt.Scale(
                                    domain=["Morning (13:00)", "Now"],
                                    range=["#4a9eff", "#ff7043"],
                                ),
                                legend=alt.Legend(title="When"),
                            ),
                            xOffset="when:N",
                            tooltip=["ticker", "when", alt.Tooltip("value:Q", format=fmt)],
                        )
                    )
                    st.altair_chart(c, use_container_width=True)

                st.divider()
                st.subheader("Summary")
                green = [
                    r["ticker"]
                    for _, r in merged.iterrows()
                    if traffic_light(r["score_long"], r["live_score_long"])[0] == "🟢"
                ]
                amber = [
                    r["ticker"]
                    for _, r in merged.iterrows()
                    if traffic_light(r["score_long"], r["live_score_long"])[0] == "🟡"
                ]
                red = [
                    r["ticker"]
                    for _, r in merged.iterrows()
                    if traffic_light(r["score_long"], r["live_score_long"])[0] == "🔴"
                ]
                if green:
                    st.success(
                        f"**Still valid — momentum holding:** {', '.join(green)}"
                    )
                if amber:
                    st.warning(
                        f"**Fading — proceed with caution:** {', '.join(amber)}"
                    )
                if red:
                    st.error(
                        f"**Gone — momentum lost, skip these today:** {', '.join(red)}"
                    )
        else:
            st.info(
                "Press the button above around 14:00–14:15 BST to run the pre-trade confirmation check."
            )

    st.success("Dashboard loaded successfully.")

# ============================================================
# TAB 3 — BACKTEST (simple version)
# ============================================================

with tab3:
    st.header("📅 14-Day Backtest")
    st.markdown(
        "Re-score each of the last 14 trading days using only data that would have been available at the time. "
        "Pick a day, then a ticker from that day's top candidates, and see whether the +10% target would have been hit."
    )
    show_glossary()
    st.divider()

    days = get_trading_days(14)
    day_labels = [d.isoformat() for d in days]
    selected_day = st.selectbox("Choose a backtest day", day_labels)

    if st.button("Score this day"):
        with st.spinner("Scoring day..."):
            bt_df = score_one_day(selected_day, df["ticker"].tolist())
        if bt_df.empty:
            st.error("No data available for that day.")
        else:
            st.subheader(f"Top candidates for {selected_day}")
            st.dataframe(bt_df.head(10), use_container_width=True)

            tickers_for_day = bt_df.head(10)["ticker"].tolist()
            chosen = st.selectbox("Pick a ticker to replay", tickers_for_day)
            if st.button("Load candle chart"):
                candles = fetch_intraday(chosen, selected_day)
                if candles.empty:
                    st.error("No 1-minute data available for that day/ticker.")
                else:
                    result = simulate_trade(candles)
                    st.write("Simulated trade result:", result)

# ============================================================
# TAB 4 — MY TRADES (from trades.csv)
# ============================================================

with tab4:
    st.header("📒 My Trades")
    st.markdown(
        "Simple log view of your trades from `trades.csv`. This is where the real learning happens."
    )
    try:
        trades = pd.read_csv("trades.csv")
        st.dataframe(trades, use_container_width=True)
        if not trades.empty:
            trades["pnl_pct"] = (
                (trades["exit_price"] - trades["entry_price"]) / trades["entry_price"]
                * 100
            )
            st.metric(
                "Number of trades", len(trades),
            )
            st.metric(
                "Average % return",
                f"{trades['pnl_pct'].mean():.2f}%",
            )
    except Exception:
        st.info("No trades.csv found yet — your first logged trade will appear here.")
