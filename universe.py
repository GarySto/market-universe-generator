import yfinance as yf
import pandas as pd
import numpy as np
import os

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# Load tickers
with open("tickers.txt", "r") as f:
    tickers = [t.strip() for t in f.readlines() if t.strip()]

records = []

for t in tickers:
    try:
        ticker = yf.Ticker(t)

        # ---------- 1. Fetch historical data ----------
        hist = ticker.history(period="15d")
        if hist.empty or len(hist) < 10:
            print(f"Skipping {t}: insufficient history")
            continue

        # Yesterday close
        yesterday_close = hist["Close"].iloc[-2]

        # 10‑day metrics
        last_10 = hist.tail(10)
        avg_volume_10d = last_10["Volume"].mean()
        high_10d = last_10["High"].max()
        low_10d = last_10["Low"].min()

        # ATR (10‑day)
        atr_10d = np.mean(last_10["High"] - last_10["Low"])

        # Trend: number of green days in last 5
        trend_5d = sum(last_10["Close"].diff().tail(5) > 0)

        # ---------- 2. Fetch pre‑market data ----------
        info = ticker.fast_info  # yfinance fast premarket fields

        premarket_price = info.get("preMarketPrice", None)
        premarket_volume = info.get("preMarketVolume", None)

        if premarket_price is None or premarket_volume is None:
            print(f"Skipping {t}: no premarket data")
            continue

        # ---------- 3. Feature engineering ----------

        # Gap %
        gap_pct = (premarket_price - yesterday_close) / yesterday_close

        # Relative volume (session)
        rvol = hist["Volume"].iloc[-1] / avg_volume_10d

        # Premarket RVOL (vs 10‑day avg session volume)
        premarket_rvol = premarket_volume / avg_volume_10d

        # Breakout score (0 = bottom of range, 1 = top)
        if high_10d != low_10d:
            breakout_score = (yesterday_close - low_10d) / (high_10d - low_10d)
        else:
            breakout_score = 0

        # Volatility score (ATR relative to price)
        volatility_score = atr_10d / yesterday_close

        # ---------- 4. Final momentum score ----------
        score = (
            gap_pct * 3 +
            rvol * 2 +
            trend_5d * 0.5 +
            breakout_score * 2 +
            volatility_score * 1
        )

        records.append({
            "ticker": t,
            "premarket_price": premarket_price,
            "premarket_volume": premarket_volume,
            "gap_pct": gap_pct,
            "premarket_rvol": premarket_rvol,
            "avg_volume_10d": avg_volume_10d,
            "rvol": rvol,
            "trend_5d": trend_5d,
            "breakout_score": breakout_score,
            "atr_10d": atr_10d,
            "volatility_score": volatility_score,
            "score": score
        })

    except Exception as e:
        print(f"Error processing {t}: {e}")
        continue

# Convert to DataFrame
df = pd.DataFrame(records)

# Sort by score descending
df = df.sort_values("score", ascending=False)

# Save output
df.to_csv("output/universe.csv", index=False)

print("Advanced universe build complete.")
