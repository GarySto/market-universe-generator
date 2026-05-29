import yfinance as yf
import pandas as pd
import os

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# Load tickers
with open("tickers.txt", "r") as f:
    tickers = [t.strip() for t in f.readlines() if t.strip()]

records = []

for t in tickers:
    try:
        data = yf.Ticker(t).history(period="1d")
        if data.empty:
            print(f"Skipping {t}: no data returned")
            continue

        last_price = data["Close"].iloc[-1]
        volume = data["Volume"].iloc[-1]

        records.append({
            "ticker": t,
            "price": float(last_price),
            "volume": int(volume)
        })

    except Exception as e:
        print(f"Error fetching {t}: {e}")
        continue

df = pd.DataFrame(records)

# Apply rules
df = df[df["price"] >= 0.50]
df = df[df["volume"] >= 500000]

# Sort by liquidity
df = df.sort_values("volume", ascending=False)

# Cap at 350
df = df.head(350)

# Save output
df.to_csv("output/universe.csv", index=False)

print("Universe build complete.")
