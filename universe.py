import pandas as pd
import yfinance as yf
from datetime import datetime
import os
import numpy as np

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Data")
OUTPUT_PATH = os.path.join(DATA_PATH, "outputs")
LOG_PATH = os.path.join(BASE_DIR, "logs", "scan_log.txt")

# Load universe
universe = pd.read_csv(os.path.join(DATA_PATH, "universe.csv"))

results = []

for ticker in universe["Ticker"]:
    try:
        # Pull 3 months of daily data
        data = yf.download(ticker, period="3mo", interval="1d")

        if len(data) < 15:
            continue

        # Indicators
        data["10_High"] = data["High"].rolling(10).max()
        data["Avg_Vol"] = data["Volume"].rolling(10).mean()

        # Simple volatility (10‑day ATR approximation)
        data["Range"] = data["High"] - data["Low"]
        data["ATR10"] = data["Range"].rolling(10).mean()

        latest = data.iloc[-1]
        prev = data.iloc[-2]

        price = latest["Close"]
        volume = latest["Volume"]
        avg_vol = latest["Avg_Vol"]
        high_10 = latest["10_High"]
        atr10 = latest["ATR10"]

        # Gap %
        gap_pct = ((latest["Open"] - prev["Close"]) / prev["Close"]) * 100

        # Score system
        score = 0

        # Gap scoring
        if gap_pct >= 2:
            score += 3
        if gap_pct >= 5:
            score += 2

        # Relative volume
        if volume > 1.5 * avg_vol:
            score += 2
        if volume > 2 * avg_vol:
            score += 2

        # Breakout scoring
        if price >= high_10:
            score += 3
        elif price >= 0.99 * high_10:
            score += 1

        # Micro‑trend
        last3 = data["Close"].tail(3)
        if last3.is_monotonic_increasing:
            score += 1

        # Volatility filter
        if atr10 / price > 0.03:  # 3% daily range
            score += 1

        results.append({
            "Ticker": ticker,
            "Price": round(price, 2),
            "Gap %": round(gap_pct, 2),
            "Volume": int(volume),
            "Avg Volume": int(avg_vol),
            "ATR10": round(atr10, 2),
            "Score": score
        })

    except Exception as e:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a") as log:
            log.write(f"{datetime.now()} - Error with {ticker}: {e}\n")

df = pd.DataFrame(results)

# Safety check
if df.empty:
    print("No results found – try again or adjust your universe.")
    exit(0)

df = df.sort_values(by="Score", ascending=False)

os.makedirs(OUTPUT_PATH, exist_ok=True)
df.to_excel(os.path.join(OUTPUT_PATH, "latest_scan.xlsx"), index=False)

print("Scan complete. Top results:")
print(df.head(10))
