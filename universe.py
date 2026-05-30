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

        yesterday_close = hist["Close"].iloc[-2]

        last_10 = hist.tail(10)
        avg_volume_10d = last_10["Volume"].mean()
        high_10d = last_10["High"].max()
        low_10d = last_10["Low"].min()
        atr_10d = np.mean(last_10["High"] - last_10["Low"])
        trend_5d = sum(last_10["Close"].diff().tail(5) > 0)

        # ---------- 2. Fetch pre‑market data ----------
        info = ticker.fast_info

        premarket_price = info.get("preMarketPrice", None)
        premarket_volume = info.get("preMarketVolume", None)

        # ---------- 3. Fallback logic ----------
        if premarket_price is None:
            premarket_price = yesterday_close  # fallback
        if premarket_volume is None:
            premarket_volume = 0  # fallback

        # ---------- 4. Feature engineering ----------
        gap_pct = (premarket_price - yesterday_close) / yesterday_close

        rvol = hist["Volume"].iloc[-1] / avg_volume_10d
        premarket_rvol = premarket_volume / avg_volume_10d

        if high_10d != low_10d:
            breakout_score = (yesterday_close - low_10d) / (high_10d - low_10d)
        else:
            breakout_score = 0

        volatility_score = atr_10d / yesterday_close

        # ---------- 5. Final momentum score ----------
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

# ---------- 6. Handle empty DataFrame safely ----------
df = pd.DataFrame(records)

if df.empty:
    print("No valid tickers found. Saving empty file.")
    df.to_csv("output/universe.csv", index=False)
    exit()

# ---------- 7. Sort and save ----------
df = df.sort_values("score", ascending=False)
df.to_csv("output/universe.csv", index=False)

print("Advanced universe build complete.")

def build_universe(target_date=None):
    """
    Build the universe for a specific date.
    If target_date is None, use today's date.
    """

    import pandas as pd
    import yfinance as yf
    from datetime import datetime, timedelta

    tickers = load_tickers()  # your existing function

    rows = []

    # If no date provided, use today
    if target_date is None:
        target_date = datetime.utcnow().date()

    # yfinance needs a datetime, not a date
    end_dt = datetime.combine(target_date, datetime.min.time())
    start_dt = end_dt - timedelta(days=15)

    for ticker in tickers:
        try:
            data = yf.download(
                ticker,
                start=start_dt,
                end=end_dt,
                interval="1d",
                progress=False
            )

            if data.empty or len(data) < 12:
                continue

            # Your existing feature engineering logic goes here
            # (yesterday_close, avg_volume_10d, high_10d, low_10d, atr_10d, etc.)
            # Then compute score and append to rows

            # Example placeholder:
            rows.append({
                "ticker": ticker,
                "score": 0,  # replace with your real score
                # ... all other fields ...
            })

        except Exception:
            continue

    df = pd.DataFrame(rows)
    return df
