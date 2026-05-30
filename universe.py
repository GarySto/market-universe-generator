import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# Load tickers
def load_tickers():
    with open("tickers.txt", "r") as f:
        return [t.strip() for t in f.readlines() if t.strip()]

# ---------------------------------------------------------
# ⭐ MAIN FUNCTION: Build universe for ANY date
# ---------------------------------------------------------
def build_universe(target_date=None):
    tickers = load_tickers()
    records = []

    # If no date provided, use today
    if target_date is None:
        target_date = datetime.utcnow().date()

    # yfinance needs datetime, not date
    end_dt = datetime.combine(target_date, datetime.min.time())
    start_dt = end_dt - timedelta(days=15)

    for t in tickers:
        try:
            ticker = yf.Ticker(t)

            # ---------- 1. Fetch historical data ----------
            hist = ticker.history(start=start_dt, end=end_dt)
            if hist.empty or len(hist) < 10:
                print(f"Skipping {t}: insufficient history")
                continue

            yesterday_close = hist["Close"].iloc[-2]
            last_10 = hist.tail(10)
            avg_volume_10d = last_10["Volume"].mean()
            high_10d = last_10["High"].max()
            low_10d = last_10["Low"].min()
            atr_10d = (last_10["High"] - last_10["Low"]).mean()

            # ---------- 2. Trend strength ----------
            trend_5d = (hist["Close"].diff() > 0).tail(5).sum()

            # ---------- 3. Premarket data ----------
            info = ticker.fast_info

            premarket_price = info.get("preMarketPrice", yesterday_close)
            if premarket_price is None:
                premarket_price = yesterday_close

            premarket_volume = info.get("preMarketVolume", 0) or 0

            # ---------- 4. Feature engineering ----------
            gap_pct = (premarket_price - yesterday_close) / yesterday_close
            rvol = hist["Volume"].iloc[-2] / avg_volume_10d
            premarket_rvol = premarket_volume / avg_volume_10d

            if high_10d != low_10d:
                breakout_score = (yesterday_close - low_10d) / (high_10d - low_10d)
            else:
                breakout_score = 0

            volatility_score = atr_10d / yesterday_close

            # ---------- 5. Final score ----------
            score = (
                3 * gap_pct +
                2 * rvol +
                0.5 * trend_5d +
                2 * breakout_score +
                1 * volatility_score
            )

            # ---------- 6. Save record ----------
            records.append({
                "ticker": t,
                "score": score,
                "gap_pct": gap_pct,
                "rvol": rvol,
                "premarket_rvol": premarket_rvol,
                "trend_5d": trend_5d,
                "breakout_score": breakout_score,
                "volatility_score": volatility_score,
            })

        except Exception as e:
            print(f"Skipping {t}: error {e}")
            continue

    df = pd.DataFrame(records)
    df = df.sort_values("score", ascending=False)
    return df

# ---------------------------------------------------------
# ⭐ When run normally, generate today's universe CSV
# ---------------------------------------------------------
if __name__ == "__main__":
    df = build_universe()
    df.to_csv("output/universe.csv", index=False)
    print("Universe written to output/universe.csv")
