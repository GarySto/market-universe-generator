import yfinance as yf
import pandas as pd
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo  # Python 3.9+

TICKERS_FILE = "tickers.txt"

def load_tickers():
    with open(TICKERS_FILE, "r") as f:
        return [t.strip() for t in f.readlines() if t.strip()]

def get_intraday_data(ticker, date):
    # Pull 1d 1m data and filter to the target date
    data = yf.download(ticker, period="2d", interval="1m", auto_adjust=False)
    data = data.tz_localize("UTC") if data.index.tz is None else data
    data = data[data.index.date == date]
    return data

def simulate_trade(intraday, entry_dt_utc, open_dt_utc, exit_dt_utc, target_pct=0.10):
    # Entry: nearest price at or after entry_dt_utc
    entry_row = intraday[intraday.index >= entry_dt_utc].head(1)
    if entry_row.empty:
        return None  # no data

    entry_price = float(entry_row["Open"].iloc[0])

    # Window: from market open to 15 minutes after
    window = intraday[(intraday.index >= open_dt_utc) & (intraday.index <= exit_dt_utc)]
    if window.empty:
        return None

    max_high = float(window["High"].max())
    hit_target = max_high >= entry_price * (1 + target_pct)

    if hit_target:
        exit_price = entry_price * (1 + target_pct)
    else:
        # Exit at last close in window
        exit_price = float(window["Close"].iloc[-1])

    ret = (exit_price - entry_price) / entry_price
    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "return": ret,
        "hit_target": hit_target,
    }

def main():
    # Placeholder: later we’ll loop over historical dates and use your universe scores per day.
    # For now, this is where we’ll integrate with universe.csv snapshots.
    pass

if __name__ == "__main__":
    main()
