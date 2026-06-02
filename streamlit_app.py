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
            pre_vol   = getattr(info, "pre_market_volume", None) or 0
            gap_pct        = (pre_price - yesterday_close) / yesterday_close if yesterday_close else 0
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

    This means the backtest now surfaces stocks that genuinely gapped on that day,
    not just stocks with high RVOL which could be any large-cap on any ordinary day.
    """
    target_date = date.fromisoformat(target_date_str)
    # We need today's data too (for the open price = gap proxy)
    end_dt   = datetime.combine(target_date + timedelta(days=1), datetime.min.time())
    start_dt = datetime.combine(target_date, datetime.min.time()) - timedelta(days=20)

    records = []
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            hist = ticker.history(start=start_dt, end=end_dt)
            if hist.empty or len(hist) < 11:
                continue

            # The target day's row is the last one (we fetched up to day+1)
            today_row       = hist.iloc[-1]
            yesterday_row   = hist.iloc[-2]

            yesterday_close  = float(yesterday_row["Close"])
            today_open       = float(today_row["Open"])
            today_volume     = float(today_row["Volume"])

            # Historical gap proxy: how much did it open vs prior close?
            gap_pct = (today_open - yesterday_close) / yesterday_close if yesterday_close else 0

            # Use last 10 days before target date for baseline metrics
            hist_before = hist.iloc[:-1]  # everything except target day
            last_10 = hist_before.tail(10)
            avg_volume_10d  = float(last_10["Volume"].mean())
            high_10d        = float(last_10["High"].max())
            low_10d         = float(last_10["Low"].min())
            atr_10d         = float((last_10["High"] - last_10["Low"]).mean())
            trend_5d        = int((hist_before["Close"].diff() > 0).tail(5).sum())

            rvol = today_volume / avg_volume_10d if avg_volume_10d else 0

            breakout_score = (
                (yesterday_close - low_10d) / (high_10d - low_10d)
                if high_10d != low_10d else 0
            )
            volatility_score = atr_10d / yesterday_close if yesterday_close else 0

            score = (
                3 * gap_pct
                + 2 * rvol
                + 0.5 * trend_5d
                + 2 * breakout_score
                + 1 * volatility_score
            )

            records.append({
                "ticker":          t,
                "score":           round(score, 4),
                "gap_pct":         round(gap_pct, 4),
                "rvol":            round(rvol, 4),
                "premarket_rvol":  0.0,
                "trend_5d":        trend_5d,
                "breakout_score":  round(breakout_score, 4),
                "volatility_score":round(volatility_score, 4),
                "yesterday_close": round(yesterday_close, 2),
                "today_open":      round(today_open, 2),
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
    end   = start + timedelta(days=1)
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
        time_col = next((c for c in df.columns
                         if any(k in c.lower() for k in ["datetime","date","timestamp"])),
                        df.columns[0])
        df = df.rename(columns={time_col: "datetime",
                                 "Open":"open","High":"high",
                                 "Low":"low","Close":"close","Volume":"volume"})
        for col in ["open","high","low","close","volume"]:
            if col not in df.columns:
                df[col] = 0

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

        window_start = bst.localize(datetime(trade_date.year, trade_date.month, trade_date.day, 12, 0)).astimezone(pytz.utc)
        window_end   = bst.localize(datetime(trade_date.year, trade_date.month, trade_date.day, 15, 0)).astimezone(pytz.utc)
        df = df[(df["datetime"] >= window_start) & (df["datetime"] < window_end)].copy()
        if df.empty:
            return pd.DataFrame()

        df = df.reset_index(drop=True)
        df["bst_time"] = df["datetime"].apply(lambda x: x.astimezone(bst).strftime("%H:%M"))

        entry_ref   = bst.localize(datetime(trade_date.year, trade_date.month, trade_date.day, 13, 30)).astimezone(pytz.utc)
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
    entry_row    = entry_candidates.iloc[0]
    entry_price  = float(entry_row["open"])
    entry_time   = entry_row["bst_time"]
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
            result.update({"exit_price": round(target_price, 4),
                           "exit_time": row["bst_time"], "exit_bst": row["bst_time"],
                           "hit_target": True})
            break
        if mins >= 75:
            result.update({"exit_price": round(float(row["close"]), 4),
                           "exit_time": row["bst_time"], "exit_bst": row["bst_time"],
                           "hit_target": False})
            break

    if result["exit_price"] is not None:
        result["pct_return"] = round((result["exit_price"] - entry_price) / entry_price * 100, 2)
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

    # Show gap chart if premarket data is populated; otherwise fall back to RVOL vs breakout
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

    # Require a real premarket gap — without gap_pct the score is driven
    # purely by RVOL and breakout which can surface large caps on ordinary days.
    # If no tickers have a real gap yet (early scan at 12:00 BST) show a warning
    # rather than candidates that don't reflect genuine premarket momentum.
    has_gap_data = bool((df["gap_pct"].abs() > 0.001).any())

    if has_gap_data:
        today_top = df[(df["score"] > 9) & (df["gap_pct"] > 0)].head(5).copy()
    else:
        today_top = pd.DataFrame()  # force the "no candidates" message

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
            "These are the stocks scoring above 9 from this morning's automated scan. "
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
                merged["live_score"] = (
                    3 * merged["live_gap_pct"].fillna(0)
                    + 2 * merged["live_pre_rvol"].fillna(0)
                    + 2 * merged["breakout_score"].fillna(0)
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
                    col5.markdown(f"**{label}**  \nGap: {row['live_gap_pct']*100:.2f}%  |  Pre-mkt RVOL: {row['live_pre_rvol']:.2f}x")

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
                            color=alt.Color("when:N",
                                scale=alt.Scale(domain=["Morning (13:00)", "Now"], range=["#4a9eff", "#ff7043"]),
                                legend=alt.Legend(title="When")),
                            xOffset="when:N",
                            tooltip=["ticker", "when", alt.Tooltip("value:Q", format=fmt)]
                        )
                    )
                    st.altair_chart(c, use_container_width=True)

                st.divider()
                st.subheader("Summary")
                green = [r["ticker"] for _, r in merged.iterrows() if traffic_light(r["score"], r["live_score"])[0] == "🟢"]
                amber = [r["ticker"] for _, r in merged.iterrows() if traffic_light(r["score"], r["live_score"])[0] == "🟡"]
                red   = [r["ticker"] for _, r in merged.iterrows() if traffic_light(r["score"], r["live_score"])[0] == "🔴"]
                if green: st.success(f"**Still valid — momentum holding:** {', '.join(green)}")
                if amber:  st.warning(f"**Fading — proceed with caution:** {', '.join(amber)}")
                if red:    st.error(f"**Gone — momentum lost, skip these today:** {', '.join(red)}")
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

    # ---- Important caveats about the backtest ----
    with st.expander("⚠️ Important: what the backtest can and can't tell you (read this first)", expanded=True):
        st.markdown("""
**What the scores mean here vs in live trading**

The backtest uses the same scoring formula as the live scanner, but with one critical difference:
historical premarket prices aren't available via the data source (yfinance). This means **gap_pct
is always 0 in backtest scores**, even if the stock did gap significantly on that day.

In live trading, gap_pct carries the highest weight (3×) in the scoring formula — it's the
strongest signal. In the backtest, scores are driven entirely by RVOL, trend, and breakout score.
This means:

- **Large-cap stocks (DELL, HPQ, MSFT etc.) can score higher than they deserve** in the backtest,
  because they consistently have high RVOL due to their trading volume — even on ordinary days when
  there's no real momentum story. A stock like DELL scoring 8 in the backtest might only score 4
  in live trading if there's no premarket gap. Conversely, a small-cap that gaps 15% in premarket
  could score 3 in the backtest but 18 in live trading.

- **The backtest is therefore best used to validate the candle behaviour and timing of your strategy**
  — does +10% get hit within 75 minutes for stocks that genuinely had momentum? — rather than to
  validate the scoring model itself.

- **Losses in the backtest don't necessarily mean the strategy is broken.** If a stock scored well
  only because of high RVOL (and not because of a real gap), it wouldn't have appeared in your
  live top 5 in the first place.

**Premarket candle data availability**

The 1-minute candle chart tries to show premarket activity from 12:00 BST. However, premarket
candle data availability varies significantly by ticker. Large liquid names (AAPL, NVDA, MSFT)
almost always have it. Smaller or less-traded names often don't — yfinance simply doesn't hold
that data. If you see a chart with no candles before 14:30, it doesn't mean nothing happened in
premarket; it means the data isn't available for that ticker.

**Data age limit**

yfinance holds approximately 30 days of 1-minute candle data. This backtest is capped at 14 days
because data quality and completeness drops significantly beyond that. Stick to the last 7–10 days
for the most reliable candle charts.
        """)

    st.divider()

    trading_days = get_trading_days(14)
    day_options  = [d.isoformat() for d in trading_days]

    col_left, col_right = st.columns([1, 2])
    with col_left:
        selected_date_str = st.selectbox(
            "Select a trading day (last 14 trading days)",
            options=day_options,
            format_func=lambda d: datetime.strptime(d, "%Y-%m-%d").strftime("%A %-d %B %Y"),
        )
    with col_right:
        st.caption(
            "The most recent days will give the most complete candle data. "
            "Older dates may have gaps in the 1-minute candle chart."
        )

    try:
        with open("tickers.txt") as f:
            bt_tickers = [t.strip() for t in f if t.strip() and t.strip() != "Ticker"]
    except Exception:
        bt_tickers = df["ticker"].tolist()

    if st.button("Score this day"):
        with st.spinner(f"Scoring {selected_date_str} — this takes 30–60 seconds for ~250 tickers..."):
            day_df = score_one_day(selected_date_str, bt_tickers)

        if day_df.empty:
            st.error(
                "No data returned for this date. It may be a US market holiday, "
                "or the data source is temporarily rate-limited. Try again or pick a different date."
            )
        else:
            st.session_state["bt_day_df"]   = day_df
            st.session_state["bt_date_str"] = selected_date_str

    if "bt_day_df" in st.session_state and st.session_state.get("bt_date_str") == selected_date_str:
        day_df = st.session_state["bt_day_df"]
        formatted_date = datetime.strptime(selected_date_str, "%Y-%m-%d").strftime("%A %-d %B %Y")

        st.divider()
        st.subheader(f"Top candidates on {formatted_date}")
        st.caption(
            "These are the stocks that scored highest on this day using historical data. "
            "Remember: gap_pct is 0 here because historical premarket prices aren't available. "
            "Scores are driven by RVOL, trend, and breakout — which means large-cap, high-volume "
            "stocks can appear higher than they would in live trading. "
            "Use this to understand what the candle looked like, not to validate the score directly."
        )

        top5 = day_df.head(5)
        st.dataframe(
            top5[["ticker", "score", "gap_pct", "rvol", "trend_5d", "breakout_score", "volatility_score", "yesterday_close"]],
            use_container_width=True
        )
        st.caption(
            "**gap_pct here** is (today's open − yesterday's close) / yesterday's close — "
            "the best available historical proxy for a premarket gap. "
            "A value of 0.08 means the stock opened 8% higher than the previous close, "
            "which is the kind of signal the live scanner looks for."
        )

        score_bar = (
            alt.Chart(top5)
            .mark_bar()
            .encode(
                x=alt.X("ticker:N", sort="-y", title="Ticker"),
                y=alt.Y("score:Q", title="Backtest score"),
                color=alt.Color("score:Q", scale=alt.Scale(scheme="redyellowgreen"),
                                legend=alt.Legend(title="Score")),
                tooltip=["ticker", "score", "gap_pct", "rvol", "trend_5d", "breakout_score"]
            )
        )
        st.altair_chart(score_bar, use_container_width=True)

        above_9 = day_df[day_df["score"] > 9]
        if above_9.empty:
            # Show ones that at least had a meaningful opening gap
            gappers = day_df[day_df["gap_pct"] > 0.03]
            if not gappers.empty:
                st.warning(
                    f"No tickers scored above 9, but {len(gappers)} had a meaningful opening gap (>3%): "
                    f"{', '.join(gappers.head(5)['ticker'].tolist())}. "
                    "These would have been worth investigating in premarket even if the overall score was below threshold."
                )
            else:
                st.warning(
                    "No tickers scored above 9 on this day, and no meaningful opening gaps were detected. "
                    "The strategy would have sat this day out — which is the correct call."
                )
        else:
            st.success(
                f"{len(above_9)} ticker(s) scored above 9: {', '.join(above_9['ticker'].tolist())}. "
                "Note these scores exclude gap_pct — live scores would be higher for any stock "
                "that was actually gapping on this day."
            )

        st.divider()

        st.subheader("1-minute candle replay")
        st.markdown(
            "Pick a ticker to see the 1-minute candle chart for that morning — from 12:00 BST through "
            "to 15:00 BST. This covers the premarket session and the first 30 minutes after the market "
            "opens at 14:30 BST. "
            "\n\n"
            "The strategy simulates buying at the **open of the 13:30 BST candle** "
            "(the start of your planned entry window), targeting **+10%**, "
            "with a hard exit at **14:45 BST** if the target isn't reached. "
            "\n\n"
            "If you don't see premarket candles before 14:30, the data simply isn't available "
            "for that ticker — this is a data limitation, not a chart error."
        )

        ticker_choice = st.selectbox(
            "Which ticker do you want to look at?",
            options=top5["ticker"].tolist(),
            key="bt_ticker_select"
        )

        if st.button("Load candle chart", key="bt_load_candles"):
            with st.spinner(f"Fetching 1-minute data for {ticker_choice} on {selected_date_str}..."):
                candles = fetch_intraday(ticker_choice, selected_date_str)
                st.session_state["bt_candles"]       = candles
                st.session_state["bt_candle_ticker"] = ticker_choice

        if "bt_candles" in st.session_state and st.session_state.get("bt_candle_ticker") == ticker_choice:
            candles = st.session_state["bt_candles"]

            if candles.empty:
                st.error(
                    "No 1-minute data returned for this ticker on this date. "
                    "yfinance holds roughly 30 days of 1-minute data and coverage can be patchy "
                    "for smaller stocks. Try a more recent date, or a more liquid ticker."
                )
            else:
                result = simulate_trade(candles)

                if not result:
                    st.warning(
                        "No candle found at 13:30 BST for this ticker. "
                        "The data may start after 13:30, or there may be a gap in coverage around that time."
                    )
                else:
                    entry_price  = result["entry_price"]
                    target_price = result["target_price"]

                    # Outcome banner
                    if result["hit_target"]:
                        st.success(
                            f"✅ **WIN** — +10% target hit at {result['exit_time']} BST.  "
                            f"Entry: **${entry_price:.2f}** → Exit: **${target_price:.2f}** "
                            f"(+{result['pct_return']:.1f}%)"
                        )
                    elif result["exit_price"] is not None:
                        sign = "+" if result.get("pct_return", 0) > 0 else ""
                        colour_fn = st.success if result.get("pct_return", 0) > 0 else st.error
                        colour_fn(
                            f"⏱️ **Time exit at 14:45 BST** — Target not reached.  "
                            f"Entry: **${entry_price:.2f}** → Closed: **${result['exit_price']:.2f}** "
                            f"({sign}{result['pct_return']:.1f}%)"
                        )
                    else:
                        st.warning("Not enough candle data to simulate the trade fully.")

                    # Premarket candle count + momentum assessment
                    premarket_candles = candles[candles["phase"] == "premarket"]
                    premarket_count = len(premarket_candles)

                    if premarket_count == 0:
                        st.warning(
                            "⚠️ **No premarket candle data available for this ticker on this date.** "
                            "The chart shows regular trading hours only (from 14:30 BST). "
                            "This is a data availability issue — yfinance premarket coverage is inconsistent. "
                            "Large-cap stocks like AAPL, NVDA and MSFT tend to have it; "
                            "smaller names often don't. The win/loss result above is still valid "
                            "based on what happened during regular hours. "
                            "**Without premarket data you can't assess the run-up, which means you "
                            "wouldn't have had the information needed to decide whether to buy this "
                            "stock in the first place. The strategy depends on seeing the momentum build.**"
                        )
                    else:
                        st.caption(
                            f"Showing {premarket_count} premarket candles (12:00–14:29 BST) "
                            f"plus regular hours to 15:00 BST."
                        )

                        # Assess whether there was a meaningful premarket run-up
                        # Compare price at start of premarket window vs price at entry (13:30)
                        pm_open  = float(premarket_candles["open"].iloc[0])
                        pm_entry = entry_price  # open at 13:30

                        # Also look at the trend within premarket — were closes generally rising?
                        pm_closes = premarket_candles["close"].values
                        rising_candles = sum(1 for i in range(1, len(pm_closes)) if pm_closes[i] > pm_closes[i-1])
                        pm_run_pct = (pm_entry - pm_open) / pm_open if pm_open > 0 else 0
                        pm_trend_pct = rising_candles / max(len(pm_closes) - 1, 1)

                        # Flag: Hub Cyber pattern = clear upward run into entry
                        if pm_run_pct >= 0.05 and pm_trend_pct >= 0.55:
                            st.success(
                                f"📈 **Strong premarket run detected** — price moved "
                                f"+{pm_run_pct*100:.1f}% from 12:00 to 13:30 BST, "
                                f"with {rising_candles} of {len(pm_closes)-1} premarket candles closing up. "
                                f"This is the kind of pattern the strategy is looking for — "
                                f"visible momentum building before the market opens."
                            )
                        elif pm_run_pct >= 0.02 and pm_trend_pct >= 0.5:
                            st.warning(
                                f"📊 **Moderate premarket movement** — price moved "
                                f"+{pm_run_pct*100:.1f}% from 12:00 to 13:30 BST. "
                                f"Some momentum present but not a strong conviction run. "
                                f"In live trading you'd want to see a clearer directional move before buying."
                            )
                        elif pm_run_pct <= -0.02:
                            st.error(
                                f"📉 **Premarket was falling** — price dropped "
                                f"{pm_run_pct*100:.1f}% from 12:00 to 13:30 BST. "
                                f"This is the opposite of what the strategy looks for. "
                                f"In live trading, you would not have bought this stock."
                            )
                        else:
                            st.info(
                                f"➡️ **Flat premarket** — price moved only "
                                f"{pm_run_pct*100:.1f}% from 12:00 to 13:30 BST. "
                                f"No clear directional momentum. In live trading, a flat or "
                                f"drifting premarket with no gap would typically mean this stock "
                                f"wouldn't appear in your top candidates at all."
                            )

                    # Build chart
                    candles["colour"] = candles.apply(
                        lambda r: "up" if r["close"] >= r["open"] else "down", axis=1
                    )
                    price_scale = alt.Scale(zero=False)

                    x_enc = alt.X("bst_time:O",
                                  title="Time (BST)",
                                  axis=alt.Axis(labelAngle=-45, tickMinStep=1,
                                                values=["12:00","12:30","13:00","13:30",
                                                        "14:00","14:30","14:45","15:00"]))

                    bodies = (
                        alt.Chart(candles)
                        .mark_bar(width=4)
                        .encode(
                            x=x_enc,
                            y=alt.Y("open:Q", title="Price ($)", scale=price_scale),
                            y2="close:Q",
                            color=alt.Color("colour:N",
                                scale=alt.Scale(domain=["up","down"], range=["#26a69a","#ef5350"]),
                                legend=None),
                            tooltip=[
                                alt.Tooltip("bst_time:N",  title="Time (BST)"),
                                alt.Tooltip("phase:N",     title="Session"),
                                alt.Tooltip("open:Q",      title="Open",  format=".2f"),
                                alt.Tooltip("high:Q",      title="High",  format=".2f"),
                                alt.Tooltip("low:Q",       title="Low",   format=".2f"),
                                alt.Tooltip("close:Q",     title="Close", format=".2f"),
                            ]
                        )
                    )

                    wicks = (
                        alt.Chart(candles)
                        .mark_rule(strokeWidth=1)
                        .encode(
                            x=x_enc,
                            y=alt.Y("low:Q",  scale=price_scale),
                            y2="high:Q",
                            color=alt.Color("colour:N",
                                scale=alt.Scale(domain=["up","down"], range=["#26a69a","#ef5350"]),
                                legend=None),
                        )
                    )

                    entry_line = (
                        alt.Chart(pd.DataFrame({"y": [entry_price]}))
                        .mark_rule(color="#cccccc", strokeWidth=1.5)
                        .encode(y=alt.Y("y:Q", scale=price_scale))
                    )

                    target_line = (
                        alt.Chart(pd.DataFrame({"y": [target_price]}))
                        .mark_rule(color="#ffb300", strokeDash=[6, 3], strokeWidth=2)
                        .encode(y=alt.Y("y:Q", scale=price_scale))
                    )

                    entry_vline = (
                        alt.Chart(pd.DataFrame({"bst_time": ["13:30"]}))
                        .mark_rule(color="#aaaaaa", strokeDash=[4,2], strokeWidth=1.5)
                        .encode(x=alt.X("bst_time:O"))
                    )
                    open_vline = (
                        alt.Chart(pd.DataFrame({"bst_time": ["14:30"]}))
                        .mark_rule(color="#4a9eff", strokeDash=[4,2], strokeWidth=1.5)
                        .encode(x=alt.X("bst_time:O"))
                    )
                    exit_vline = (
                        alt.Chart(pd.DataFrame({"bst_time": ["14:45"]}))
                        .mark_rule(color="#ff4444", strokeDash=[4,2], strokeWidth=1.5)
                        .encode(x=alt.X("bst_time:O"))
                    )

                    layers = [bodies, wicks, entry_line, target_line,
                              entry_vline, open_vline, exit_vline]

                    if result.get("exit_bst") and result.get("exit_price") is not None:
                        exit_colour = "#ffb300" if result["hit_target"] else "#ff4444"
                        exit_dot = (
                            alt.Chart(pd.DataFrame({
                                "bst_time": [result["exit_bst"]],
                                "y":        [result["exit_price"]],
                            }))
                            .mark_point(size=180, shape="diamond", filled=True, color=exit_colour)
                            .encode(
                                x=alt.X("bst_time:O"),
                                y=alt.Y("y:Q", scale=price_scale),
                                tooltip=[
                                    alt.Tooltip("bst_time:N", title="Exit time (BST)"),
                                    alt.Tooltip("y:Q",        title="Exit price", format=".2f"),
                                ]
                            )
                        )
                        layers.append(exit_dot)

                    chart = (
                        alt.layer(*layers)
                        .properties(
                            height=420,
                            title=(
                                f"{ticker_choice} — {formatted_date}  |  "
                                f"Entry 13:30 BST @ ${entry_price:.2f}  |  "
                                f"Target ${target_price:.2f} (+10%)  |  "
                                f"Hard exit 14:45 BST"
                            )
                        )
                    )
                    st.altair_chart(chart, use_container_width=True)

                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.markdown(f"**⬜ Entry price (13:30 BST)**  \n${entry_price:.2f} — grey horizontal line")
                    col_b.markdown(f"**🟡 +10% target**  \n${target_price:.2f} — amber dashed line")
                    col_c.markdown(f"**🔵 Market open (14:30 BST)**  \nBlue dashed vertical")
                    col_d.markdown(f"**🔴 Hard exit (14:45 BST)**  \nRed dashed vertical — close regardless")

                    st.caption(
                        "Green candles = price closed up that minute. Red = closed down. "
                        "The diamond marker shows the actual exit point. "
                        "Hover over any candle for the exact open/high/low/close."
                    )


# ============================================================
# TAB 4 — MY TRADES
# ============================================================

with tab4:

    st.header("📒 My Trades")
    st.markdown(
        "This tab tracks every trade made using the scanner. "
        "The data comes from `trades.csv` in the repository — updated manually after each trade. "
        "Win = +10% or above. Partial = positive but below 10%. Loss = negative return."
    )

    show_glossary()
    st.divider()

    # Try to load trades.csv
    try:
        trades = pd.read_csv("trades.csv")
        trades.columns = [c.strip() for c in trades.columns]
        trades = trades.dropna(how="all")

        # Only proceed if there are actual data rows with a ticker
        has_trades = (
            len(trades) > 0 and
            "ticker" in trades.columns and
            trades["ticker"].notna().any() and
            trades["ticker"].astype(str).str.strip().ne("").any()
        )

        if has_trades:
            for col in ["entry_price", "exit_price", "shares", "running_bank", "score", "gap_pct", "rvol"]:
                if col in trades.columns:
                    trades[col] = pd.to_numeric(trades[col], errors="coerce")

            trades["return_gbp"] = (
                (trades["exit_price"] - trades["entry_price"]) * trades["shares"]
            ).round(2)
            trades["return_pct"] = (
                (trades["exit_price"] - trades["entry_price"]) / trades["entry_price"] * 100
            ).round(2)
            trades["win_loss"] = trades["return_pct"].apply(
                lambda x: "Win" if pd.notna(x) and x >= 10
                else ("Partial" if pd.notna(x) and x > 0
                else ("Loss" if pd.notna(x) else ""))
            )

    except FileNotFoundError:
        has_trades = False
        trades = pd.DataFrame()
    except Exception:
        has_trades = False
        trades = pd.DataFrame()

    if not has_trades:
        st.info(
            "No trades logged yet. Once you make your first trade, add a row to `trades.csv` "
            "in your GitHub repo and it will appear here automatically. "
            "Use the Excel tracker to log trades, then export the data as CSV and commit it."
        )
        st.markdown("""
**trades.csv column format:**
```
date, ticker, entry_time, exit_time, entry_price, exit_price, shares, score, gap_pct, rvol, pretrade_check, why_picked, notes
```
        """)
    else:
        total_trades  = len(trades)
        wins          = len(trades[trades["return_pct"] >= 10])
        losses        = len(trades[trades["return_pct"] < 0])
        partials      = total_trades - wins - losses
        win_rate      = wins / total_trades if total_trades > 0 else 0
        total_pnl     = trades["return_gbp"].sum()
        avg_return    = trades["return_pct"].mean()
        current_bank  = 50 + total_pnl
        trades_to_go  = max(0, 98 - total_trades)
        best_trade    = trades["return_pct"].max()
        worst_trade   = trades["return_pct"].min()

        # ── KPI row ──
        st.subheader("Performance at a glance")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Current bank", f"£{current_bank:.2f}", f"{'+' if total_pnl >= 0 else ''}£{total_pnl:.2f}")
        k2.metric("Total trades", total_trades, f"{trades_to_go} to go")
        k3.metric("Win rate", f"{win_rate*100:.0f}%", f"{wins}W / {losses}L")
        k4.metric("Avg return", f"{avg_return:.1f}%")
        k5.metric("Best trade", f"+{best_trade:.1f}%")
        k6.metric("Worst trade", f"{worst_trade:.1f}%")

        # Progress to £1M
        st.divider()
        progress = min(current_bank / 1_000_000, 1.0)
        st.subheader("Progress to £1,000,000")
        st.progress(progress)
        st.caption(f"£{current_bank:.2f} of £1,000,000 — {progress*100:.4f}% there. {trades_to_go} trades remaining at +10% compounding.")

        st.divider()

        # ── Win/Loss breakdown ──
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Win / Loss breakdown")
            st.caption("Win = +10% or above. Partial = positive but below 10%. Loss = negative.")
            breakdown = pd.DataFrame({
                "Result": ["Win", "Partial", "Loss"],
                "Count":  [wins, partials, losses],
            })
            bar = (
                alt.Chart(breakdown)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Result:N", sort=["Win","Partial","Loss"]),
                    y=alt.Y("Count:Q", title="Number of trades"),
                    color=alt.Color("Result:N",
                        scale=alt.Scale(
                            domain=["Win","Partial","Loss"],
                            range=["#1d9e75","#f39c12","#c0392b"]),
                        legend=None),
                    tooltip=["Result","Count"]
                )
                .properties(height=220)
            )
            st.altair_chart(bar, use_container_width=True)

        with col_r:
            st.subheader("Return % per trade")
            st.caption("Each bar is one trade. The amber dashed line is your +10% target.")
            if "ticker" in trades.columns:
                trades["label"] = trades["ticker"] + " " + trades.get("date", pd.Series(range(len(trades)))).astype(str)
            else:
                trades["label"] = [f"Trade {i+1}" for i in range(len(trades))]

            trades["bar_color"] = trades["return_pct"].apply(
                lambda x: "Win" if x >= 10 else ("Partial" if x > 0 else "Loss")
            )
            returns_chart = (
                alt.Chart(trades)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("label:N", title="Trade", axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y("return_pct:Q", title="Return (%)"),
                    color=alt.Color("bar_color:N",
                        scale=alt.Scale(domain=["Win","Partial","Loss"],
                                        range=["#1d9e75","#f39c12","#c0392b"]),
                        legend=None),
                    tooltip=["label","return_pct","return_gbp"]
                )
                .properties(height=220)
            )
            target_line = (
                alt.Chart(pd.DataFrame({"y": [10]}))
                .mark_rule(color="#f39c12", strokeDash=[5,3], strokeWidth=1.5)
                .encode(y="y:Q")
            )
            st.altair_chart(returns_chart + target_line, use_container_width=True)

        # ── Running bank chart ──
        st.divider()
        st.subheader("📈 Running bank — the line that matters")
        st.caption("Starting from £50. Every trade adds or subtracts. This needs to keep going up and to the right.")

        trades["trade_num"] = range(1, len(trades)+1)
        bank_line = (
            alt.Chart(trades)
            .mark_line(point=True, color="#2E5FA3", strokeWidth=2)
            .encode(
                x=alt.X("trade_num:Q", title="Trade number"),
                y=alt.Y("running_bank:Q", title="Bank (£)", scale=alt.Scale(zero=False)),
                tooltip=["trade_num", "ticker", "return_pct", "running_bank"]
            )
            .properties(height=280)
        )
        start_line = (
            alt.Chart(pd.DataFrame({"y": [50]}))
            .mark_rule(color="#888888", strokeDash=[4,3], strokeWidth=1)
            .encode(y="y:Q")
        )
        st.altair_chart(bank_line + start_line, use_container_width=True)

        # ── Score analysis ──
        if "score" in trades.columns and trades["score"].notna().any():
            st.divider()
            st.subheader("Does a higher score mean a better trade?")
            st.caption(
                "This scatter shows whether the morning scanner score correlates with actual trade returns. "
                "Ideally you want to see higher scores clustering toward the top. "
                "If they don't, the scoring model needs reviewing."
            )
            score_scatter = (
                alt.Chart(trades)
                .mark_circle(size=80)
                .encode(
                    x=alt.X("score:Q", title="Morning score"),
                    y=alt.Y("return_pct:Q", title="Return (%)"),
                    color=alt.Color("bar_color:N",
                        scale=alt.Scale(domain=["Win","Partial","Loss"],
                                        range=["#1d9e75","#f39c12","#c0392b"]),
                        legend=None),
                    tooltip=["ticker","score","return_pct","gap_pct"]
                )
                .interactive()
                .properties(height=260)
            )
            st.altair_chart(score_scatter, use_container_width=True)

        # ── Full trade log ──
        st.divider()
        st.subheader("Full trade log")
        display_cols = [c for c in ["date","ticker","entry_time","exit_time",
                                     "entry_price","exit_price","return_pct",
                                     "win_loss","score","gap_pct","pretrade_check","why_picked","notes"]
                        if c in trades.columns or c in ["return_pct","win_loss"]]
        st.dataframe(trades[display_cols] if display_cols else trades, use_container_width=True)
