import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

os.makedirs("output", exist_ok=True)


def load_tickers():
    with open("tickers.txt", "r") as f:
        return [t.strip() for t in f.readlines() if t.strip() and t.strip() != "Ticker"]


def build_universe(target_date=None):
    tickers = load_tickers()
    records = []

    if target_date is None:
        target_date = datetime.utcnow().date()

    end_dt   = datetime.combine(target_date, datetime.min.time())
    start_dt = end_dt - timedelta(days=20)

    for t in tickers:
        try:
            ticker = yf.Ticker(t)

            # ── 1. Historical data ──────────────────────────────────────────
            hist = ticker.history(start=start_dt, end=end_dt)
            if hist.empty or len(hist) < 10:
                print(f"Skipping {t}: insufficient history")
                continue

            yesterday_close  = float(hist["Close"].iloc[-1])
            yesterday_volume = float(hist["Volume"].iloc[-1])

            last_10        = hist.tail(10)
            avg_volume_10d = float(last_10["Volume"].mean())
            high_10d       = float(last_10["High"].max())
            low_10d        = float(last_10["Low"].min())
            atr_10d        = float((last_10["High"] - last_10["Low"]).mean())
            trend_5d       = int((hist["Close"].diff() > 0).tail(5).sum())

            # ── 2. Minimum quality filters ──────────────────────────────────
            # Skip penny stocks and very illiquid names — fills are unreliable
            if yesterday_close < 2.0:
                print(f"Skipping {t}: price ${yesterday_close:.2f} below $2 minimum")
                continue
            if avg_volume_10d < 500_000:
                print(f"Skipping {t}: avg volume {avg_volume_10d:,.0f} below 500k minimum")
                continue

            # ── 3. Premarket data ───────────────────────────────────────────
            # Use getattr (not .get) — fast_info is an object, not a dict
            info = ticker.fast_info

            premarket_price = getattr(info, "pre_market_price", None)
            if not premarket_price or premarket_price <= 0:
                premarket_price = yesterday_close  # fallback: no gap

            premarket_volume = getattr(info, "pre_market_volume", None) or 0

            # ── 4. Feature engineering ──────────────────────────────────────
            gap_pct        = (premarket_price - yesterday_close) / yesterday_close
            rvol           = yesterday_volume / avg_volume_10d if avg_volume_10d else 0
            premarket_rvol = premarket_volume / avg_volume_10d if avg_volume_10d else 0

            breakout_score = (
                (yesterday_close - low_10d) / (high_10d - low_10d)
                if high_10d != low_10d else 0
            )
            volatility_score = atr_10d / yesterday_close if yesterday_close else 0

            # ── 5. Momentum score ───────────────────────────────────────────
            # Weights: gap (3x) > rvol/breakout (2x each) > volatility (1x) > trend (0.5x)
            score = (
                3   * gap_pct
                + 2 * rvol
                + 2 * breakout_score
                + 1 * volatility_score
                + 0.5 * trend_5d
            )

            records.append({
                "ticker":           t,
                "premarket_price":  round(premarket_price, 2),
                "premarket_volume": int(premarket_volume),
                "score":            round(score, 4),
                "gap_pct":          round(gap_pct, 4),
                "rvol":             round(rvol, 4),
                "premarket_rvol":   round(premarket_rvol, 4),
                "avg_volume_10d":   int(avg_volume_10d),
                "trend_5d":         trend_5d,
                "breakout_score":   round(breakout_score, 4),
                "atr_10d":          round(atr_10d, 4),
                "volatility_score": round(volatility_score, 4),
            })

        except Exception as e:
            print(f"Skipping {t}: {e}")
            continue

    if not records:
        print("Warning: no records generated")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = build_universe()
    if not df.empty:
        df.to_csv("output/universe.csv", index=False)
        print(f"Universe written — {len(df)} tickers scored")
        top5 = df[df["score"] > 9].head(5)
        if not top5.empty:
            print(f"\nTop candidates (score > 9):")
            print(top5[["ticker", "score", "gap_pct", "rvol", "breakout_score"]].to_string(index=False))
        else:
            print("\nNo tickers above score 9 today")
    else:
        print("No data written")
