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


def calc_rsi(closes, period=14):
    """
    Standard RSI(14) from a series of closes. Needs at least period+1
    values to produce anything meaningful — caller should fetch enough
    history (see start_dt below, widened from 15 to 60 days specifically
    for this).
    """
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))
    val = rsi_series.iloc[-1]
    return float(val) if pd.notna(val) else None


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
    # WIDENED from 15 -> 60 days: 15 days (~10 trading days) was enough for
    # the existing 10-day volume/breakout stats, but nowhere near enough
    # for a stable RSI(14), which needs at least 14 trading days of clean
    # deltas to settle, plus warmup. 60 calendar days gives a safe margin
    # without changing anything the 10-day stats below depend on.
    start_dt = end_dt - timedelta(days=60)

    for t in tickers:
        try:
            ticker = yf.Ticker(t)

            # ---------- 1. Fetch historical data ----------
            hist = ticker.history(start=start_dt, end=end_dt)
            if hist.empty or len(hist) < 20:
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

            # ---------- 3. RSI(14) — NEW ----------
            # Added after the win-rate sweep (10 Jul 2026) found RSI is by
            # far the strongest predictor of this model's actual outcomes —
            # RSI 90+ alone: 76.5% win rate vs ~50% baseline, a 26-point
            # swing. It was never in this formula before. See rsi_score
            # below for how heavily it's now weighted.
            rsi_val = calc_rsi(hist["Close"])

            # ---------- 4. Premarket data ----------
            # 31 Jul 2026 — KNOWN LIMITATION, deliberately left in place for now.
            #
            # `preMarketPrice` is not a fast_info field at all (it lives on
            # .info, where yfinance returns None for it silently anyway). So
            # this has ALWAYS fallen through to yesterday_close, which makes
            # gap_pct exactly 0.0 for every ticker, every day — which is
            # exactly what the dashboard shows. The Game is a premarket gap
            # strategy that has never once measured a gap.
            #
            # Fixing the SOURCE properly means a second yfinance call per
            # ticker with prepost=True, across ~1,100 tickers, which risks
            # blowing the Actions job time limit. That's a scoped piece of
            # work for its own session (two-pass: score without gap, then
            # fetch real premarket bars for the top ~100 only).
            #
            # What IS fixed today: premarket_price and prev_close are now
            # WRITTEN to universe.csv. place_demo_orders_game.py reads
            # premarket_price to size its trade and crashed with a KeyError
            # every time it found a candidate, because the column did not
            # exist. The Game can now actually place an order. The gap
            # signal being flat is a separate, tracked problem.
            info = ticker.fast_info
            try:
                premarket_price = info.get("preMarketPrice", None)
            except Exception:
                premarket_price = None

            if premarket_price is None or premarket_price != premarket_price:
                premarket_price = yesterday_close
                premarket_source = "prev_close_fallback"
            else:
                premarket_source = "live"

            try:
                premarket_volume = info.get("preMarketVolume", 0) or 0
            except Exception:
                premarket_volume = 0

            # ---------- 5. Feature engineering ----------
            gap_pct = (premarket_price - yesterday_close) / yesterday_close
            rvol = hist["Volume"].iloc[-2] / avg_volume_10d
            premarket_rvol = premarket_volume / avg_volume_10d

            if high_10d != low_10d:
                breakout_score = (yesterday_close - low_10d) / (high_10d - low_10d)
            else:
                breakout_score = 0

            volatility_score = atr_10d / yesterday_close

            # ---------- 6. Final score — REWEIGHTED (10 Jul 2026) ----------
            # OLD formula (score = 3*gap_pct + 2*rvol + 0.5*trend_5d +
            # 2*breakout_score + 1*volatility_score) had two real bugs the
            # sweep exposed:
            #   1. No RSI at all, despite it being the dominant predictor
            #   2. No normalisation — gap_pct is a small fraction (e.g.
            #      0.05 for 5%) while rvol is often 1-10+, so rvol silently
            #      dominated the "3x weighted" gap_pct term every time,
            #      despite gap_pct nominally having the higher weight.
            #
            # Each component below is now scaled to a bounded, comparable
            # range before weighting, so the weights actually mean what
            # they say:
            #   rsi_score   0-4   (primary driver — matches the sweep)
            #   gap_score   0-3   (capped — a 10%+ gap already maxes this out)
            #   rvol_score  0-2
            #   breakout    0-1   (already 0-1 by construction)
            # Max total ~10, keeping the existing "7.0+" Trade Today
            # threshold roughly meaningful without needing to retune it
            # blindly alongside this change.
            rsi_score = min((rsi_val or 0) / 100 * 4, 4)
            gap_score = min(abs(gap_pct) * 100 * 0.3, 3)
            rvol_score = min(rvol * 0.5, 2) if rvol == rvol else 0  # NaN-safe
            breakout_component = breakout_score if breakout_score == breakout_score else 0

            score = rsi_score + gap_score + rvol_score + breakout_component

            # ---------- 7. Save record ----------
            records.append({
                "ticker": t,
                "score": score,
                # premarket_price / prev_close — REQUIRED by
                # place_demo_orders_game.py to size a trade. Do not remove.
                "premarket_price": round(float(premarket_price), 4),
                "prev_close": round(float(yesterday_close), 4),
                "premarket_source": premarket_source,
                "rsi": round(rsi_val, 1) if rsi_val is not None else None,
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
