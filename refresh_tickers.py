"""
refresh_tickers.py — Run this occasionally to keep tickers.txt clean.

Reads your current tickers.txt, checks yesterday's close price via yfinance,
and outputs a cleaned tickers.txt that only contains stocks currently
within the $1.50–$75 price filter used by universe.py.

Also reports which tickers were removed and why, so you can replace them
with new candidates from your T212 instruments list.

Usage:
    python refresh_tickers.py

Output:
    tickers_cleaned.txt  — ready to replace tickers.txt in your repo
    tickers_removed.txt  — removed tickers with reason and last price
"""

import yfinance as yf
import pandas as pd
import time

PRICE_MIN = 1.50
PRICE_MAX = 75.0
VOL_MIN   = 200_000  # minimum 10-day avg volume

def load_tickers(path="tickers.txt"):
    return [t.strip() for t in open(path) if t.strip()]

def check_tickers(tickers, batch_size=50):
    in_range, removed = [], []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, period="15d", progress=False, auto_adjust=True)
            if data.empty:
                for t in batch:
                    removed.append((t, None, None, "no data"))
                continue

            closes  = data["Close"].iloc[-1]
            volumes = data["Volume"].tail(10).mean()

            for t in batch:
                price = closes.get(t) if hasattr(closes, 'get') else (closes[t] if t in closes.index else None)
                vol   = volumes.get(t) if hasattr(volumes, 'get') else (volumes[t] if t in volumes.index else None)

                if price is None or pd.isna(price):
                    removed.append((t, None, None, "no data / possibly delisted"))
                elif float(price) < PRICE_MIN:
                    removed.append((t, round(float(price), 2), None, f"below ${PRICE_MIN}"))
                elif float(price) > PRICE_MAX:
                    removed.append((t, round(float(price), 2), None, f"above ${PRICE_MAX}"))
                elif vol is not None and not pd.isna(vol) and float(vol) < VOL_MIN:
                    removed.append((t, round(float(price), 2), int(vol), f"avg volume below {VOL_MIN:,}"))
                else:
                    in_range.append(t)
        except Exception as e:
            for t in batch:
                removed.append((t, None, None, f"error: {e}"))
        time.sleep(1)

    return in_range, removed


if __name__ == "__main__":
    tickers = load_tickers()
    print(f"Checking {len(tickers)} tickers...")

    in_range, removed = check_tickers(tickers)

    # Write cleaned list
    with open("tickers_cleaned.txt", "w") as f:
        for t in sorted(in_range):
            f.write(t + "\n")

    # Write removed list with reasons
    with open("tickers_removed.txt", "w") as f:
        f.write("ticker,last_price,avg_volume,reason\n")
        for t, price, vol, reason in sorted(removed, key=lambda x: x[3]):
            f.write(f"{t},{price or ''},{vol or ''},{reason}\n")

    print(f"\n✅ In range: {len(in_range)}")
    print(f"❌ Removed:  {len(removed)}")
    print(f"\nFiles written: tickers_cleaned.txt, tickers_removed.txt")
    print(f"Review tickers_removed.txt, then replace tickers.txt with tickers_cleaned.txt")
    print(f"You have {470 - len(in_range)} slots to fill with new in-range candidates.")