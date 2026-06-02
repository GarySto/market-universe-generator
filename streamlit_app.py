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
    "Score": "The overall momentum score — a weighted combination of all the signals below. Higher is better. The strategy targets stocks scoring above 9 on a live trading day.",
    "Gap % (gap_pct)": "How much the premarket price has moved compared to yesterday's closing price, expressed as a fraction. A gap of 0.05 means the stock is up 5% before the market opens. This is the strongest signal — stocks that gap up significantly with volume behind them tend to keep moving.",
    "RVOL (rvol)": "Relative Volume — yesterday's trading volume divided by the 10-day average volume. A reading of 2.0 means twice the normal volume traded. High RVOL suggests elevated interest in the stock.",
    "Premarket RVOL (premarket_rvol)": "The same RVOL calculation but applied to premarket volume specifically. A fresher signal than regular RVOL — if volume is building before the market opens, that's worth paying attention to.",
    "Trend 5d (trend_5d)": "How many of the last 5 trading days closed higher than they opened. Ranges from 0 (down every day) to 5 (up every day). A score of 4 or 5 means consistent recent momentum.",
    "Breakout Score (breakout_score)": "Where yesterday's closing price sits within the stock's 10-day high-to-low range. 0 means it closed at the very bottom of its recent range. 1 means it closed at the very top. Values above 0.7 suggest the stock is pushing toward a breakout.",
    "Volatility Score (volatility_score)": "A measure of how much the stock moves on a typical day, based on the average daily range (high minus low) over 10 days, expressed as a fraction of the price. Higher means more potential movement — which cuts both ways.",
    "ATR (atr_10d)": "Average True Range — the average daily price range (high minus low) over the last 10 days. Used internally to calculate the volatility score.",
    "BST": "British Summer Time — UTC+1, which is the timezone the dashboard uses for all times. The US market opens at 14:30 BST.",
    "ET": "Eastern Time — the timezone US markets operate in. 14:30 BST = 09:30 ET (market open).",
    "Premarket": "The trading session before the official US market open at 14:30 BST. Shares can be bought and sold but with lower volume and wider spreads than regular hours.",
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
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
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


def premarket_momentum_score(gap_pct, pre_rvol, breakout_score):
    """
    Same shape as in universe.py — used for live comparison.
    """
    return (
        5 * gap_pct +
        2 * pre_rvol +
        2 * breakout_score
    )


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
    This is the best available historical substitute for premarket gap — it captures
    stocks that actually opened significantly higher than where they closed the night before,
    which is what the live strategy is hunting for.
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
                if high_10d != low_10d else 0
            )
            volatility_score = atr_10d / yesterday_close if yesterday_close else 0

            score = (
                3 * gap_pct +
                2 * rvol +
                0.5 * trend_5d +
                2 * breakout_score +
                1 * volatility_score
            )

            records.append({
                "ticker":           t,
                "score":            round(score, 4),
                "gap_pct":          round(gap_pct, 4),
                "rvol":             round(rvol, 4),
                "premarket_rvol":   0.0,
                "trend_5d":         trend_5d,
                "breakout_score":   round(breakout_score, 4),
                "volatility_score": round(volatility_score, 4),
                "yesterday_close":  round(yesterday_close, 2),
                "today_open":       round(today_open, 2),
            })
        except Exception:
            continue

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_intraday(ticker_str, trade_date_str):
    """
    Fetch 1-minute candle data, 12:00-15:00 BST.
    prepost=True captures premarket candles where available.
    yfinance holds ~30 days of 1-min data; reliability drops beyond 14 days.
    """
    import pytz
    trade_date = date.fromisoformat(trade_date_str)
    bst = pytz.timezone("Europe/London")
    start = datetime.combine(trade_date, datetime.min.time())
    end = start + timedelta(days=1)
    try:
        ticker_obj = yf.Ticker(ticker_str)
        df = ticker_obj.history(
            start=start, end=end,
            interval="1m",
            prepost=True,
            auto_adjust=True,
        )
        if df.empty:
            df = yf.download(
                ticker_str, start=start, end=end,
                interval="1m", prepost=True,
                progress=False, auto_adjust=True,
            )
            if df.empty:
                return pd.DataFrame()
            df = df.reset_index()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        else:
            df = df.reset_index()

        df.columns = [str(c).strip() for c in df.columns]
        time_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ["datetime", "date", "timestamp"])),
            df.columns[0]
        )
        df = df.rename(columns={
            time_col: "datetime",
            "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        })
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = 0

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

        window_start = bst.localize(datetime(trade_date.year, trade_date.month, trade_date.day, 12, 0)).astimezone(pytz.utc)
        window_end = bst.localize(datetime(trade_date.year, trade_date.month, trade_date.day, 15, 0)).astimezone(pytz.utc)
        df = df[(df["datetime"] >= window_start) & (df["datetime"] < window_end)].copy()
        if df.empty:
            return pd.DataFrame()

        df = df.reset_index(drop=True)
        df["bst_time"] = df["datetime"].apply(lambda x: x.astimezone(bst).strftime("%H:%M"))

        entry_ref = bst.localize(datetime(trade_date.year, trade_date.month, trade_date.day, 13, 30)).astimezone(pytz.utc)
        market_open = bst.localize(datetime(trade_date.year, trade_date.month, trade_date.day, 14, 30)).astimezone(pytz.utc)
        df["mins_from_entry"] = df["datetime"].apply(lambda x: int((x - entry_ref).total_seconds() / 60))
        df["phase"] = df["datetime"].apply(lambda x: "post-open" if x >= market_open else "premarket")
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
        "entry_price":  entry_price,
        "entry_time":   entry_time,
        "target_price": round(target_price, 4),
        "exit_price":   None,
        "exit_time":    None,
        "exit_bst":     None,
        "hit_target":   False,
        "pct_return":   None,
    }

    trade_window = candles[(candles["mins_from_entry"] >= 0) & (candles["mins_from_entry"] <= 75)]
    for _, row in trade_window.iterrows():
        mins = int(row["mins_from_entry"])
        if float(row["high"]) >= target_price:
            result.update({
                "exit_price": round(target_price, 4),
                "exit_time": row["bst_time"],
                "exit_bst": row["bst_time"],
                "hit_target": True
            })
            break
        if mins >= 75:
            result.update({
                "exit_price": round(float(row["close"]), 4),
                "exit_time": row["bst_time"],
                "exit_bst": row["bst_time"],
                "hit_target": False
            })
            break

    if result["exit_price"] is not None:
        result["pct_return"] = round((result["exit_price"] - entry_price) / entry_price * 100, 2)
    return result


@st.cache_data(ttl=300)
def load_trades():
    try:
        df = pd.read_csv("trades.csv")
    except FileNotFoundError:
        return pd.DataFrame()
    if df.empty:
        return df
    # Add pct_return if not present
    if "entry_price" in df.columns and "exit_price" in df.columns and "pct_return" not in df.columns:
        df["pct_return"] = (df["exit_price"] - df["entry_price"]) / df["entry_price"] * 100
    return df


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
        "Anything scoring above 9 with a meaningful gap and high RVOL is worth investigating further."
    )
    show_glossary()
    st.divider()

    st.subheader("🏆 Top 10 Momentum Tickers")
    st.caption(
        "Sorted by score descending. These are the strongest momentum candidates from today's scan. "
        "Note: gap_pct and premarket_rvol will show as 0 outside of premarket hours — "
        "the scan runs at 13:00 BST specifically to capture live premarket data."
    )
    top10 = df.head(10)[[
        "ticker", "score", "gap_pct", "rvol", "premarket_rvol",
        "trend_5d", "breakout_score", "volatility_score"
    ]]
    st.dataframe(top10, use_container_width=True)

    gap_values = df["gap_pct"].dropna()
    has_gap_data = bool((gap_values.abs() > 0.001).any())

    if has_gap_data:
        st.subheader("🔥 Gap % vs Relative Volume (RVOL)")
        st.caption(
            "Each dot is one ticker. Top-right corner = gapping up AND high volume — "
            "the strongest combination for this strategy. Colour = score (green higher, red lower)."
        )
        scatter = (
            alt.Chart(df[df["gap_pct"].abs() > 0.001].head(50))
            .mark_circle(size=80)
            .encode(
                x=alt.X("gap_pct:Q", title="Gap % (premarket vs yesterday's close)",
                        axis=alt.Axis(format=".1%")),
                y=alt.Y("rvol:Q", title="RVOL (volume vs 10-day average)"),
                color=alt.Color("score:Q", scale=alt.Scale(scheme="redyellowgreen"),
                                legend=alt.Legend(title="Score")),
                tooltip=["ticker", "score", "gap_pct", "rvol", "premarket_rvol", "trend_5d", "breakout_score"],
            )
            .interactive()
        )
        st.altair_chart(scatter, use_container_width=True)
    else:
        st.subheader("📊 RVOL vs Breakout Score")
        st.info(
            "**Gap % data isn't available yet** — it populates when the 13:00 BST scan runs during premarket hours. "
            "Until then this chart shows RVOL vs Breakout Score, which are always populated and "
            "still useful for spotting relative strength. Top-right = high volume AND near the top of recent price range."
        )
        scatter = (
            alt.Chart(df.head(50))
            .mark_circle(size=80)
            .encode(
                x=alt.X("breakout_score:Q",
                        title="Breakout score (0 = bottom of recent range, 1 = top)",
                        scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("rvol:Q", title="RVOL (volume vs 10-day average)"),
                color=alt.Color("score:Q", scale=alt.Scale(scheme="redyellowgreen"),
                                legend=alt.Legend(title="Score")),
                tooltip=["ticker", "score", "breakout_score", "rvol", "trend_5d", "volatility_score"],
            )
            .interactive()
        )
        st.altair_chart(scatter, use_container_width=True)

    st.subheader("📊 Trend Strength (Last 5 Days)")
    st.caption(
        "How many of the last 5 trading days closed higher than they opened. "
        "A full bar (5) means the stock has gone up every day this week. "
        "This is a confirmation signal — a stock that's been trending is more likely to continue."
    )
    trend_chart = (
        alt.Chart(df.head(20))
        .mark_bar()
        .encode(
            x=alt.X("ticker:N", sort="-y", title="Ticker"),
            y=alt.Y("trend_5d:Q", title="Days up (out of 5)"),
            color=alt.Color("trend_5d", scale=alt.Scale(scheme="blues"),
                            legend=alt.Legend(title="Days up")),
            tooltip=["ticker", "trend_5d"],
        )
    )
    st.altair_chart(trend_chart, use_container_width=True)

    st.subheader("🚀 Breakout Score (0 = bottom of range, 1 = top of range)")
    st.caption(
        "Where yesterday's close sits within the stock's 10-day high-to-low range. "
        "Stocks near 1.0 are trading at the top of their recent range — potential breakout candidates. "
        "Stocks near 0 have been beaten down recently. For momentum trading, you generally want this above 0.7."
    )
    breakout_chart = (
        alt.Chart(df.head(20))
        .mark_bar()
        .encode(
            x=alt.X("ticker:N", sort="-y", title="Ticker"),
            y=alt.Y("breakout_score:Q", title="Breakout score (0–1)"),
            color=alt.Color("breakout_score", scale=alt.Scale(scheme="greens"),
                            legend=alt.Legend(title="Score")),
            tooltip=["ticker", "breakout_score"],
        )
    )
    st.altair_chart(breakout_chart, use_container_width=True)

    st.subheader("📋 Full Universe (Sortable)")
    st.caption("Click any column header to sort. All tickers from today's scan.")
    st.dataframe(df, use_container_width=True)


# ============================================================
# TAB 2 — TRADE TODAY
# ============================================================

with tab2:
    st.header("🚀 Trade Today")
    st.markdown(
        "This tab filters the morning scan down to your top candidates — stocks scoring above 9 "
        "with strong momentum signals. The strategy is to buy in premarket between 13:00 and 14:15 BST, "
        "target a **+10% return**, and exit no later than **14:45 BST** (15 minutes after market open) "
        "regardless of outcome. The pre-trade confirmation check below lets you verify momentum is still "
        "in play before committing."
    )
    show_glossary()
    st.divider()

    has_gap_data = bool((df["gap_pct"].abs() > 0.001).any())

    if has_gap_data:
        today_top = df[(df["score"] > 10) & (df["gap_pct"] > 0)].head(5).copy()
    else:
        today_top = pd.DataFrame()

    if today_top.empty:
        if not has_gap_data:
            st.warning(
                "No premarket gap data available yet — all gap_pct values are showing as 0. "
                "This happens when the scan runs before meaningful premarket activity has built up (before about 13:00 BST / 08:00 ET). "
                "The 13:00 BST scan should populate this correctly. "
                "Candidates shown without a real gap are driven by RVOL and trend alone, "
                "which is not a strong enough signal for this strategy on its own. "
                "Check back after 13:00 BST when the next scan runs."
            )
        else:
            st.warning(
                "No tickers above score 9 with a real premarket gap today. "
                "This is a valid outcome — it means the market isn't showing the kind of activity this strategy looks for. "
                "The correct move is to sit out and wait for tomorrow."
            )
    else:
        st.subheader("Morning candidates (13:00 scan)")
        st.caption(
            "These are the stocks scoring above 10 with a real premarket gap from this morning's automated scan. "
            "They represent the strongest momentum signals available right now. "
            "Before trading any of them, run the pre-trade confirmation check below to make sure "
            "the momentum is still in play — a lot can change between 13:00 and 14:30."
        )
        display_cols = ["ticker", "score", "gap_pct", "premarket_rvol",
                        "rvol", "trend_5d", "breakout_score", "volatility_score"]
        st.dataframe(today_top[display_cols], use_container_width=True)
        st.info(
            "**Suggested entry window: 13:30–14:15 BST** — during the premarket session. "
            "The market opens at 14:30 BST. "
            "Exit target: +10% from your entry price, or close the position by 14:45 BST."
        )

        st.subheader("📊 Morning score comparison")
        st.caption("Colour goes from red (lower score) to green (higher score). Only stocks above 9 shown.")
        score_chart = (
            alt.Chart(today_top)
            .mark_bar()
            .encode(
                x=alt.X("ticker:N", sort="-y", title="Ticker"),
                y=alt.Y("score:Q", title="Momentum score"),
                color=alt.Color("score", scale=alt.Scale(scheme="redyellowgreen"),
                                legend=alt.Legend(title="Score")),
                tooltip=["ticker", "score", "rvol", "trend_5d", "breakout_score"]
            )
        )
        st.altair_chart(score_chart, use_container_width=True)

        st.divider()
        st.subheader("🔄 Pre-trade confirmation — is momentum still in play?")
        st.markdown(
            "Run this around **14:00–14:15 BST**, about 15–30 minutes before the market opens. "
            "It fetches the latest available premarket data for your top candidates and compares it "
            "to the 13:00 scan. The traffic lights tell you whether each stock's momentum has held, "
            "faded, or gone entirely. "
            "\n\n"
            "🟢 **Still valid** — score has held or improved. Momentum is intact.  \n"
            "🟡 **Fading** — score has dropped but is still above 7. Proceed with caution.  \n"
            "🔴 **Gone** — score has dropped significantly. Skip this one today."
        )

        if st.button("Refresh live data now"):
            tickers_to_check = today_top["ticker"].tolist()
            with st.spinner("Fetching live premarket data..."):
                live_df = fetch_live_scores(tickers_to_check)

            if live_df.empty:
                st.error(
                    "Could not fetch live data. yfinance may be rate-limited — try again in a minute. "
                    "This is not unusual; yfinance is a free data source with no rate limit guarantees."
                )
            else:
                merged = today_top[["ticker", "score", "gap_pct", "premarket_rvol", "breakout_score"]].merge(
                    live_df, on="ticker", how="left"
                )
                merged["live_score"] = premarket_momentum_score(
                    merged["live_gap_pct"].fillna(0),
                    merged["live_pre_rvol"].fillna(0),
                    merged["breakout_score"].fillna(0),
                )
                merged["score_delta"] = merged["live_score"] - merged["score"]

                st.subheader("Traffic light status")
                for _, row in merged.iterrows():
                    light, label = traffic_light(row["score"], row["live_score"])
                    delta_str = f"+{row['score_delta']:.2f}" if row["score_delta"] >= 0 else f"{row['score_delta']:.2f}"
                    col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 3])
                    col1.markdown(f"## {light}")
                    col2.metric("Ticker", row["ticker"])
                    col3.metric("Morning score", f"{row['score']:.2f}")
                    col4.metric("Live score", f"{row['live_score']:.2f}", delta=delta_str)
                    col5.markdown(
                        f"**{label}**  \n"
                        f"Gap: {row['live_gap_pct']*100:.2f}%  |  Pre-mkt RVOL: {row['live_pre_rvol']:.2f}x"
                    )

                st.divider()

                for metric, morning_col, live_col, fmt, title, caption in [
                    ("Score", "score", "live_score", ".2f",
                     "📊 Score: morning vs now",
                     "Blue = the 13:00 scan score. Orange = the live score right now. A falling bar means momentum has faded."),
                    ("Gap %", "gap_pct", "live_gap_pct", ".2%",
                     "📊 Gap %: morning vs now",
                     "The gap is the premarket price move vs yesterday's close. If the gap has shrunk significantly since 13:00, enthusiasm is waning."),
                    ("Premarket RVOL", "premarket_rvol", "live_pre_rvol", ".2f",
                     "📊 Premarket RVOL: morning vs now",
                     "Premarket relative volume — is trading activity increasing or decreasing since 13:00? Rising RVOL into the open is a good sign."),
                ]:
                    st.subheader(title)
                    st.caption(caption)
                    chart_data = pd.concat([
                        merged[["ticker", morning_col]].rename(columns={morning_col: "value"}).assign(when="Morning (13:00)"),
                        merged[["ticker", live_col]].rename(columns={live_col: "value"}).assign(when="Now"),
                    ])
                    c = (
                        alt.Chart(chart_data)
                        .mark_bar()
                        .encode(
                            x=alt.X("ticker:N", title="Ticker"),
                            y=alt.Y("value:Q", axis=alt.Axis(format=fmt if "%" in fmt else None)),
                            color=alt.Color(
                                "when:N",
                                scale=alt.Scale(domain=["Morning (13:00)", "Now"], range=["#4a9eff", "#ff7043"]),
                                legend=alt.Legend(title="When")
                            ),
                            xOffset="when:N",
                            tooltip=["ticker", "when", alt.Tooltip("value:Q", format=fmt)]
                        )
                    )
                    st.altair_chart(c, use_container_width=True)

                st.divider()
                st.subheader("Summary")
                green = [r["ticker"] for _, r in merged.iterrows() if traffic_light(r["score"], r["live_score"])[0] == "🟢"]
                amber = [r["ticker"] for _, r in merged.iterrows() if traffic_light(r["score"], r["live_score"])[0] == "🟡"]
                red = [r["ticker"] for _, r in merged.iterrows() if traffic_light(r["score"], r["live_score"])[0] == "🔴"]
                if green:
                    st.success(f"**Still valid — momentum holding:** {', '.join(green)}")
                if amber:
                    st.warning(f"**Fading — proceed with caution:** {', '.join(amber)}")
                if red:
                    st.error(f"**Gone — momentum lost, skip these today:** {', '.join(red)}")
        else:
            st.info("Press the button above around 14:00–14:15 BST to run the pre-trade confirmation check.")

    st.success("Dashboard loaded successfully.")


# ============================================================
# TAB 3 — BACKTEST
# ============================================================

with tab3:
    st.header("📅 14-Day Backtest")
    st.markdown(
        "This re-scores each trading day in the last 14 days using only the data that would have "
        "been available at the time — so there's no cheating with hindsight. Pick a day, pick a ticker "
        "from that day's top candidates, and see the 1-minute candle chart for that morning. "
        "The chart shows whether the +10% target would have been hit before the 14:45 BST exit."
    )

    show_glossary()
    st.divider()

    with st.expander("⚠️ Important: what the backtest can and can't tell you (read this first)", expanded=True):
        st.markdown("""
**What the scores mean here vs in live trading**

The backtest uses the same scoring formula as the live scanner, but with one critical difference:
historical premarket prices aren't available via the data source (yfinance). Instead, the gap proxy is:

> (today's open - yesterday's close) / yesterday's close

That means:
- It only captures stocks that **actually opened** significantly higher than they closed.
- It won't see overnight premarket moves that faded before the open.

So the backtest is conservative: if something looks good here, it genuinely opened strong on that day.
        """)

    trading_days = get_trading_days(14)
    if not trading_days:
        st.warning("No recent trading days available.")
    else:
        day_strs = [d.isoformat() for d in trading_days]
        selected_day_str = st.selectbox(
            "Choose a backtest day (most recent first):",
            options=day_strs,
            format_func=lambda s: date.fromisoformat(s).strftime("%A %d %B %Y"),
        )

        if selected_day_str:
            with st.spinner("Scoring universe for that day..."):
                bt_df = score_one_day(selected_day_str, df["ticker"].tolist())

            if bt_df.empty:
                st.error("No data available for that day — yfinance may not have enough history for some tickers.")
            else:
                st.subheader("Top candidates for that day")
                st.caption("These are the highest scoring tickers **as of that morning**, using only data available then.")
                st.dataframe(bt_df.head(15), use_container_width=True)

                tickers_for_day = bt_df.head(15)["ticker"].tolist()
                chosen_ticker = st.selectbox("Pick a ticker to simulate the trade:", options=tickers_for_day)

                if chosen_ticker:
                    with st.spinner("Fetching 1-minute candles for that day..."):
                        candles = fetch_intraday(chosen_ticker, selected_day_str)

                    if candles.empty:
                        st.error("No intraday data available for that ticker/day combination.")
                    else:
                        sim = simulate_trade(candles, target_pct=0.10)

                        st.subheader(f"1-minute chart for {chosen_ticker} on {date.fromisoformat(selected_day_str).strftime('%d %B %Y')}")
                        st.caption("Window: 12:00–15:00 BST. Entry at 13:30 BST, target +10%, hard exit 14:45 BST.")

                        base = alt.Chart(candles).encode(
                            x=alt.X("bst_time:N", title="Time (BST)")
                        )

                        rule = base.mark_rule().encode(
                            y="low:Q",
                            y2="high:Q",
                            color=alt.value("#888888"),
                            tooltip=["bst_time", "open", "high", "low", "close", "volume"]
                        )

                        bar = base.mark_bar().encode(
                            y="open:Q",
                            y2="close:Q",
                            color=alt.condition("datum.close >= datum.open",
                                                alt.value("#2ecc71"),
                                                alt.value("#e74c3c"))
                        )

                        price_chart = (rule + bar).properties(height=400)

                        if sim and sim.get("entry_price") is not None:
                            target_line = alt.Chart(pd.DataFrame({
                                "y": [sim["target_price"]],
                                "label": ["Target +10%"]
                            })).mark_rule(color="#f39c12", strokeDash=[4, 4]).encode(
                                y="y:Q"
                            )
                            price_chart = price_chart + target_line

                        st.altair_chart(price_chart, use_container_width=True)

                        if sim:
                            st.subheader("Simulated trade outcome")
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Entry price", f"${sim['entry_price']:.2f}", sim["entry_time"])
                            if sim["exit_price"] is not None:
                                col2.metric("Exit price", f"${sim['exit_price']:.2f}", sim["exit_time"])
                                col3.metric("Return", f"{sim['pct_return']:.2f}%")
                                col4.metric("Hit target?", "✅ Yes" if sim["hit_target"] else "❌ No")
                            else:
                                col2.metric("Exit price", "—")
                                col3.metric("Return", "—")
                                col4.metric("Hit target?", "—")


# ============================================================
# TAB 4 — MY TRADES
# ============================================================

with tab4:
    st.header("📒 My Trades")
    st.markdown(
        "This tab tracks your actual trades — not the backtest. "
        "Each row in `trades.csv` is one real trade with its own notes and context."
    )

    show_glossary()
    st.divider()

    trades = load_trades()
    if trades.empty:
        st.info("No trades logged yet. Add rows to `trades.csv` to see them here.")
    else:
        st.subheader("Trade log")
        st.dataframe(trades, use_container_width=True)

        st.divider()

        st.subheader("Summary stats")
        num_trades = len(trades)
        avg_return = trades["pct_return"].mean() if "pct_return" in trades.columns else None
        wins = trades[trades["pct_return"] > 0].shape[0] if "pct_return" in trades.columns else 0
        win_rate = (wins / num_trades * 100) if num_trades > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Number of trades", num_trades)
        if avg_return is not None:
            c2.metric("Average % return", f"{avg_return:.2f}%")
        else:
            c2.metric("Average % return", "—")
        c3.metric("Win rate", f"{win_rate:.1f}%")

        st.divider()

        if "running_bank" in trades.columns:
            st.subheader("Equity curve (running bank)")
            eq_chart = (
                alt.Chart(trades.reset_index())
                .mark_line(point=True)
                .encode(
                    x=alt.X("index:Q", title="Trade number"),
                    y=alt.Y("running_bank:Q", title="Running bank (£)"),
                    tooltip=["date", "ticker", "running_bank"]
                )
            )
            st.altair_chart(eq_chart, use_container_width=True)

        if "pct_return" in trades.columns:
            st.subheader("Distribution of trade returns")
            hist = (
                alt.Chart(trades)
                .mark_bar()
                .encode(
                    x=alt.X("pct_return:Q", bin=alt.Bin(maxbins=20), title="% return"),
                    y=alt.Y("count():Q", title="Number of trades"),
                    tooltip=["count()"]
                )
            )
            st.altair_chart(hist, use_container_width=True)

            st.subheader("Average return by ticker")
            by_ticker = trades.groupby("ticker", as_index=False)["pct_return"].mean()
            bar = (
                alt.Chart(by_ticker)
                .mark_bar()
                .encode(
                    x=alt.X("ticker:N", sort="-y", title="Ticker"),
                    y=alt.Y("pct_return:Q", title="Average % return"),
                    tooltip=["ticker", "pct_return"]
                )
            )
            st.altair_chart(bar, use_container_width=True)
