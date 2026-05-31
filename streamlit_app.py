import streamlit as st
import pandas as pd
import altair as alt
import yfinance as yf
import numpy as np
from datetime import datetime, date, timedelta, timezone

st.set_page_config(page_title="Momentum Scanner", layout="wide")

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


def get_trading_days(n=30):
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
    Score a list of tickers as-of a given date using historical data only.
    Returns a sorted dataframe of (ticker, score, gap_pct, rvol, ...).
    Uses the same logic as universe.py.
    """
    target_date = date.fromisoformat(target_date_str)
    end_dt   = datetime.combine(target_date, datetime.min.time())
    start_dt = end_dt - timedelta(days=20)

    records = []
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            hist = ticker.history(start=start_dt, end=end_dt)
            if hist.empty or len(hist) < 10:
                continue

            yesterday_close = float(hist["Close"].iloc[-2])
            last_10 = hist.tail(10)
            avg_volume_10d  = float(last_10["Volume"].mean())
            high_10d        = float(last_10["High"].max())
            low_10d         = float(last_10["Low"].min())
            atr_10d         = float((last_10["High"] - last_10["Low"]).mean())
            trend_5d        = int((hist["Close"].diff() > 0).tail(5).sum())

            # Historical backtest: no premarket available, use prev close as proxy
            yesterday_volume = float(hist["Volume"].iloc[-2])
            gap_pct          = 0.0   # unknown in backtest without premarket data
            rvol             = yesterday_volume / avg_volume_10d if avg_volume_10d else 0
            premarket_rvol   = 0.0

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
                "premarket_rvol":  round(premarket_rvol, 4),
                "trend_5d":        trend_5d,
                "breakout_score":  round(breakout_score, 4),
                "volatility_score":round(volatility_score, 4),
                "yesterday_close": round(yesterday_close, 2),
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
    Fetch 1-minute candle data for a ticker on a specific date.
    Trims to regular market hours only (09:30–10:00 ET = 13:30–14:00 UTC).
    Returns a dataframe indexed from minute 0 = market open.
    """
    trade_date = date.fromisoformat(trade_date_str)
    start = datetime.combine(trade_date, datetime.min.time())
    end   = start + timedelta(days=1)
    try:
        df = yf.download(
            ticker_str,
            start=start,
            end=end,
            interval="1m",
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.rename(columns={"Datetime": "datetime", "Open": "open",
                                 "High": "high", "Low": "low",
                                 "Close": "close", "Volume": "volume"})

        # Ensure UTC-aware datetimes
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

        # US market opens at 09:30 ET = 13:30 UTC
        # Find the first candle at or after 09:30 ET on the trade date
        import pytz
        et = pytz.timezone("America/New_York")
        market_open_et = et.localize(
            datetime(trade_date.year, trade_date.month, trade_date.day, 9, 30, 0)
        ).astimezone(pytz.utc).replace(tzinfo=None)
        market_open_utc = pd.Timestamp(market_open_et, tz="UTC")
        market_close_utc = market_open_utc + timedelta(minutes=30)

        df = df[(df["datetime"] >= market_open_utc) & (df["datetime"] < market_close_utc)]

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index(drop=True)
        # Minute 0 = market open (09:30 ET)
        df["minute"] = range(len(df))
        return df
    except Exception:
        return pd.DataFrame()


def simulate_trade(candles, entry_price, target_pct=0.10, hold_minutes=15):
    """
    Simulate a trade on 1-min candle data.
    Entry at open of first candle.
    Exit at first candle where high >= target, or after hold_minutes.
    Returns dict with outcome details.
    """
    target_price = entry_price * (1 + target_pct)
    result = {
        "entry_price":  entry_price,
        "target_price": round(target_price, 4),
        "exit_price":   None,
        "exit_minute":  None,
        "hit_target":   False,
        "pct_return":   None,
    }

    for i, row in candles.iterrows():
        minute = i  # 0-indexed from open
        if minute >= hold_minutes:
            # Time's up — exit at close of this candle
            result["exit_price"]  = round(float(row["close"]), 4)
            result["exit_minute"] = minute
            result["hit_target"]  = False
            break
        if float(row["high"]) >= target_price:
            result["exit_price"]  = round(target_price, 4)
            result["exit_minute"] = minute
            result["hit_target"]  = True
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

tab1, tab2, tab3 = st.tabs(["Scanner", "Trade Today", "Backtest"])


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

    today_top = df[df["score"] > 9].head(5).copy()

    if today_top.empty:
        st.warning("No tickers above score 9 today. Check back once premarket is active (from around 13:00 BST).")
    else:
        st.subheader("Morning candidates (13:00 scan)")
        display_cols = ["ticker", "score", "gap_pct", "premarket_rvol",
                        "rvol", "trend_5d", "breakout_score", "volatility_score"]
        st.dataframe(today_top[display_cols], use_container_width=True)
        st.info("Suggested entry time: **14:00–14:15 BST** — after the pre-trade confirmation check below.")

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

        st.divider()
        st.subheader("🔄 Pre-trade confirmation — is momentum still in play?")
        st.caption("Run this around 14:00–14:15 BST, about 15–30 minutes before the market opens.")

        if st.button("Refresh live data now"):
            tickers_to_check = today_top["ticker"].tolist()
            with st.spinner("Fetching live premarket data..."):
                live_df = fetch_live_scores(tickers_to_check)

            if live_df.empty:
                st.error("Could not fetch live data. yfinance may be rate-limited — try again in a minute.")
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

                # Side-by-side comparison charts
                for metric, morning_col, live_col, fmt, title in [
                    ("Score",          "score",         "live_score",    ".2f",  "📊 Score: morning vs now"),
                    ("Gap %",          "gap_pct",       "live_gap_pct",  ".2%",  "📊 Gap %: morning vs now"),
                    ("Premarket RVOL", "premarket_rvol","live_pre_rvol", ".2f",  "📊 Premarket RVOL: morning vs now"),
                ]:
                    st.subheader(title)
                    chart_data = pd.concat([
                        merged[["ticker", morning_col]].rename(columns={morning_col: "value"}).assign(when="Morning (13:00)"),
                        merged[["ticker", live_col]].rename(columns={live_col: "value"}).assign(when="Now"),
                    ])
                    c = (
                        alt.Chart(chart_data)
                        .mark_bar()
                        .encode(
                            x=alt.X("ticker:N"),
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

    st.header("📅 30-Day Backtest")
    st.markdown(
        "This re-scores each trading day in the last 30 days using only the data "
        "that would have been available at the time. Pick a day, pick a ticker, "
        "and see what the 1-minute candle chart looked like — and whether you'd have won."
    )

    # Note about gap_pct in backtest
    st.info(
        "**Note on gap %:** Historical premarket data isn't available via yfinance, so gap_pct "
        "shows as 0 in the backtest scores. The score here is driven by RVOL, trend, and breakout — "
        "it's a conservative estimate. In live trading, gap_pct is the strongest signal."
    )

    trading_days = get_trading_days(30)
    day_options  = [d.isoformat() for d in trading_days]

    col_left, col_right = st.columns([1, 2])

    with col_left:
        selected_date_str = st.selectbox(
            "Select a trading day",
            options=day_options,
            format_func=lambda d: datetime.strptime(d, "%Y-%m-%d").strftime("%A %-d %B %Y"),
        )

    with col_right:
        st.markdown("&nbsp;")  # spacer

    # Load tickers from file for the backtest
    try:
        with open("tickers.txt") as f:
            bt_tickers = [t.strip() for t in f if t.strip() and t.strip() != "Ticker"]
    except Exception:
        bt_tickers = df["ticker"].tolist()

    if st.button("Score this day"):
        with st.spinner(f"Scoring {selected_date_str} — this takes 30–60 seconds for ~250 tickers..."):
            day_df = score_one_day(selected_date_str, bt_tickers)

        if day_df.empty:
            st.error("No data returned for this date. It may be a US market holiday, or yfinance is rate-limited.")
        else:
            st.session_state["bt_day_df"]   = day_df
            st.session_state["bt_date_str"] = selected_date_str

    # Show results if we have them
    if "bt_day_df" in st.session_state and st.session_state.get("bt_date_str") == selected_date_str:
        day_df = st.session_state["bt_day_df"]
        formatted_date = datetime.strptime(selected_date_str, "%Y-%m-%d").strftime("%A %-d %B %Y")

        st.divider()
        st.subheader(f"Top candidates on {formatted_date}")

        top5 = day_df.head(5)
        st.dataframe(
            top5[["ticker", "score", "rvol", "trend_5d", "breakout_score", "volatility_score", "yesterday_close"]],
            use_container_width=True
        )

        # Score bar chart for this day
        score_bar = (
            alt.Chart(top5)
            .mark_bar()
            .encode(
                x=alt.X("ticker:N", sort="-y"),
                y=alt.Y("score:Q"),
                color=alt.Color("score", scale=alt.Scale(scheme="redyellowgreen")),
                tooltip=["ticker", "score", "rvol", "trend_5d", "breakout_score"]
            )
        )
        st.altair_chart(score_bar, use_container_width=True)

        # Score threshold filter
        above_9 = day_df[day_df["score"] > 9]
        if above_9.empty:
            st.warning("No tickers scored above 9 on this day — the strategy would have sat out.")
        else:
            st.success(f"{len(above_9)} ticker(s) scored above 9: {', '.join(above_9['ticker'].tolist())}")

        st.divider()

        # ---- Candle viewer ----
        st.subheader("1-minute candle replay")
        st.markdown(
            "Pick a ticker from the top 5 above to see what the first 30 minutes of trading looked like. "
            "Entry is at the open of the first candle. Target is +10%. "
            "Exit is at 15 minutes if the target isn't hit."
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
                    "No 1-minute data available for this ticker on this date. "
                    "yfinance only holds about 30 days of 1-minute data, so older dates won't work. "
                    "Try a more recent day."
                )
            else:
                entry_price = float(candles["open"].iloc[0])
                result = simulate_trade(candles, entry_price)

                # ---- Outcome banner ----
                if result["hit_target"]:
                    st.success(
                        f"✅ **WIN** — Target hit at minute {result['exit_minute']}. "
                        f"Entry: ${entry_price:.2f}  →  Exit: ${result['exit_price']:.2f}  "
                        f"(+{result['pct_return']:.1f}%)"
                    )
                elif result["exit_price"] is not None:
                    colour = "success" if result["pct_return"] and result["pct_return"] > 0 else "error"
                    fn = st.success if colour == "success" else st.error
                    fn(
                        f"⏱️ **Time exit** — Closed at minute {result['exit_minute']}. "
                        f"Entry: ${entry_price:.2f}  →  Exit: ${result['exit_price']:.2f}  "
                        f"({'+'if result['pct_return']>0 else ''}{result['pct_return']:.1f}%)"
                    )
                else:
                    st.warning("Not enough candle data to simulate the trade.")

                # ---- Candle chart ----
                candles["colour"] = candles.apply(
                    lambda r: "up" if r["close"] >= r["open"] else "down", axis=1
                )

                price_scale = alt.Scale(zero=False)

                # Candle bodies
                bodies = (
                    alt.Chart(candles)
                    .mark_bar(width=10)
                    .encode(
                        x=alt.X("minute:Q", title="Minutes after market open (09:30 ET / 14:30 BST)",
                                axis=alt.Axis(tickMinStep=1)),
                        y=alt.Y("open:Q", title="Price ($)", scale=price_scale),
                        y2="close:Q",
                        color=alt.Color("colour:N",
                            scale=alt.Scale(domain=["up", "down"], range=["#26a69a", "#ef5350"]),
                            legend=None),
                        tooltip=[
                            alt.Tooltip("minute:Q", title="Minute"),
                            alt.Tooltip("open:Q",   title="Open",   format=".2f"),
                            alt.Tooltip("high:Q",   title="High",   format=".2f"),
                            alt.Tooltip("low:Q",    title="Low",    format=".2f"),
                            alt.Tooltip("close:Q",  title="Close",  format=".2f"),
                            alt.Tooltip("volume:Q", title="Volume"),
                        ]
                    )
                )

                # Wicks
                wicks = (
                    alt.Chart(candles)
                    .mark_rule(strokeWidth=1)
                    .encode(
                        x="minute:Q",
                        y=alt.Y("low:Q",  scale=price_scale),
                        y2="high:Q",
                        color=alt.Color("colour:N",
                            scale=alt.Scale(domain=["up", "down"], range=["#26a69a", "#ef5350"]),
                            legend=None),
                    )
                )

                # Entry price line (solid white/grey — where you buy)
                entry_line = (
                    alt.Chart(pd.DataFrame({"y": [entry_price], "label": [f"Entry ${entry_price:.2f}"]}))
                    .mark_rule(color="#aaaaaa", strokeWidth=1.5)
                    .encode(y=alt.Y("y:Q", scale=price_scale))
                )

                # +10% target line (amber dashed)
                target_line = (
                    alt.Chart(pd.DataFrame({"y": [result["target_price"]], "label": [f"+10% target ${result['target_price']:.2f}"]}))
                    .mark_rule(color="#ffb300", strokeDash=[5, 3], strokeWidth=2)
                    .encode(y=alt.Y("y:Q", scale=price_scale))
                )

                # 15-minute exit vertical line (grey dashed)
                exit_rule = (
                    alt.Chart(pd.DataFrame({"x": [15]}))
                    .mark_rule(color="#666666", strokeDash=[4, 4], strokeWidth=1.5)
                    .encode(x="x:Q")
                )

                # Exit point marker — big dot at the actual exit candle
                if result["exit_minute"] is not None and result["exit_price"] is not None:
                    exit_dot_data = pd.DataFrame({
                        "minute": [result["exit_minute"]],
                        "price":  [result["exit_price"]],
                    })
                    exit_dot_colour = "#ffb300" if result["hit_target"] else "#ff4444"
                    exit_dot = (
                        alt.Chart(exit_dot_data)
                        .mark_point(size=150, shape="diamond", filled=True, color=exit_dot_colour)
                        .encode(
                            x="minute:Q",
                            y=alt.Y("price:Q", scale=price_scale),
                            tooltip=[
                                alt.Tooltip("minute:Q", title="Exit minute"),
                                alt.Tooltip("price:Q",  title="Exit price", format=".2f"),
                            ]
                        )
                    )
                    chart = (bodies + wicks + entry_line + target_line + exit_rule + exit_dot)
                else:
                    chart = (bodies + wicks + entry_line + target_line + exit_rule)

                chart = chart.properties(
                    height=380,
                    title=f"{ticker_choice} — {formatted_date} — 1-min candles (market open to 30 mins)"
                )
                st.altair_chart(chart, use_container_width=True)

                st.caption(
                    f"Grey solid line = entry price (${entry_price:.2f}, open of first candle at 09:30 ET).  "
                    f"Amber dashed line = +10% target (${result['target_price']:.2f}).  "
                    f"Grey dashed vertical = 15-minute forced exit.  "
                    f"Diamond marker = actual exit point.  "
                    f"Green candles = closed up. Red = closed down."
                )

                # Volume bar chart below
                vol_chart = (
                    alt.Chart(candles)
                    .mark_bar(width=10)
                    .encode(
                        x=alt.X("minute:Q", title="Minutes after open"),
                        y=alt.Y("volume:Q", title="Volume"),
                        color=alt.Color("colour:N",
                            scale=alt.Scale(domain=["up", "down"], range=["#26a69a", "#ef5350"]),
                            legend=None),
                    )
                    .properties(height=100, title="Volume by minute")
                )
                st.altair_chart(vol_chart, use_container_width=True)
