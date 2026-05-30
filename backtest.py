import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
import os

# CONFIG
LOOKBACK_DAYS = 30
SCORE_THRESHOLD = 9
TOP_N = 5
OUTPUT_FILE = "output/backtest_results.csv"
TICKERS_FILE = "tickers.txt"

# Load tickers
def load_tickers():
    with open(TICKERS_FILE, "r") as f:
        return [t.strip() for t in f.readlines() if t.strip()]

# Build universe for a given date (historical)
def build_universe_for_date(target_date):
    from universe import build_universe  # reuse your existing logic
    df = build_universe(target_date=target_date)
    df = df.sort_values("score", ascending=False)
    return df

# Get intraday 1-minute data for a ticker on a specific date
def get_intraday(ticker, date):
    data = yf.download(
        ticker,
        interval="1m",
        start=date,
        end=date + timedelta(days=1),
        auto_adjust=False,
        progress=False
    )
    if data.empty:
        return None
    data = data.tz_localize("UTC") if data.index.tz is None else data
    return data

# Simulate a single trade
def simulate_trade(intraday, entry_dt_utc, open_dt_utc, exit_dt_utc):
    # Find the first row at or after the entry timestamp
    entry_row = intraday[intraday.index >= entry_dt_utc].head(1)
    if entry_row.empty:
        return None

    # Safety check: if Open is NaN or malformed
    if entry_row["Open"].isna().all():
        return None

    # Extract a single float safely (fixes the Series→float error)
    entry_price = float(entry_row["Open"].astype(float).values[0])

    # Window from market open to 15 minutes after
    window = intraday[(intraday.index >= open_dt_utc) & (intraday.index <= exit_dt_utc)]
    if window.empty:
        return None

    max_high = float(window["High"].max())
    hit_target = max_high >= entry_price * 1.10

    if hit_target:
        exit_price = entry_price * 1.10
    else:
        exit_price = float(window["Close"].iloc[-1])

    ret = (exit_price - entry_price) / entry_price

    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "return": ret,
        "hit_target": hit_target
    }


# Main backtest loop
def run_backtest():
    tickers = load_tickers()
    results = []

    today = datetime.now(ZoneInfo("UTC")).date()

    for i in range(1, LOOKBACK_DAYS + 1):
        date = today - timedelta(days=i)

        print(f"Processing {date}...")

        # Build universe for that day
        df = build_universe_for_date(date)
        df = df[df["score"] > SCORE_THRESHOLD]

        if df.empty:
            continue

        top = df.head(TOP_N)

        # Define timestamps
        entry_dt = datetime.combine(date, time(13, 30), tzinfo=ZoneInfo("UTC"))
        open_dt = datetime.combine(date, time(14, 30), tzinfo=ZoneInfo("UTC"))
        exit_dt = datetime.combine(date, time(14, 45), tzinfo=ZoneInfo("UTC"))

        for _, row in top.iterrows():
            ticker = row["ticker"]
            intraday = get_intraday(ticker, date)

            if intraday is None:
                continue

            trade = simulate_trade(intraday, entry_dt, open_dt, exit_dt)
            if trade is None:
                continue

            results.append({
                "date": date,
                "ticker": ticker,
                "score": row["score"],
                **trade
            })

    # Save results
    os.makedirs("output", exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"Backtest complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_backtest()
