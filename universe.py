"""
GarAI Momentum Scanner — universe.py
======================================
Builds and scores the daily premarket universe.

Data sources:
  Primary:   T212 API  — live premarket prices (ISA-tradeable, no rate limits)
  Secondary: yfinance  — OHLCV history, RVOL, trend, breakout score
  Tertiary:  D drive   — RSI/MACD when running locally (home PC only)

Snapshot architecture:
  Every run saves output/premarket_HH.csv (e.g. premarket_08.csv)
  This lets the dashboard show score trends across the morning:
  08:00 → 10:00 → 12:00 → 13:00 UTC
  Rising scores = momentum building. Fading = skip.

Schedule (daily.yml): 08:00, 10:00, 12:00, 13:00 UTC Mon-Fri
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import time
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

os.makedirs("output", exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE    = 50       # tickers per yf.download() call
MIN_PRICE     = 2.00     # below = too volatile / wide spreads
MAX_PRICE     = 75.00    # above = 10% move too rare
MIN_AVG_VOL   = 200_000  # minimum 10-day avg volume
MIN_HISTORY   = 10       # minimum days of OHLCV needed

# T212 API
T212_API_KEY    = os.environ.get("T212_API_KEY", "")
T212_API_SECRET = os.environ.get("T212_API_SECRET", "")
T212_BASE       = "https://live.trading212.com/api/v0"

# RSI filter thresholds (from 1.8M signal backtest)
RSI_STRONG_MIN = 50
RSI_STRONG_MAX = 70
RSI_AVOID_MAX  = 40

# D drive paths — home PC only
D_TECH_DIR    = r"D:\GarAI\data\technicals"
# ─────────────────────────────────────────────────────────────────────────────


def load_tickers():
    for path in ["tickers.txt", os.path.join("..", "tickers.txt")]:
        if os.path.exists(path):
            with open(path) as f:
                tickers = [t.strip() for t in f if t.strip()]
            print(f"Loaded {len(tickers)} tickers from {path}")
            return tickers
    print("WARNING: tickers.txt not found")
    return []


def _safe_min_max(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0, 1.0
    mn, mx = float(s.min()), float(s.max())
    return (mn, mn + 1.0) if mx == mn else (mn, mx)


def _normalize(series, mn, mx):
    return (series - mn) / (mx - mn) if mx != mn else series * 0.0


# ── T212 price fetch ──────────────────────────────────────────────────────────

def fetch_t212_prices(tickers):
    """
    Fetch live premarket prices from T212 API for a list of tickers.
    Returns dict: {ticker: price} for tickers with valid prices.

    T212 gives us live prices for ISA-tradeable instruments — no rate limits,
    no silent None returns. Replaces the broken yfinance preMarketPrice approach.
    """
    if not T212_API_KEY:
        print("  T212_API_KEY not set — skipping T212 price fetch")
        return {}

    # T212 beta API uses HTTP Basic Auth: key as username, secret as password
    import base64
    if T212_API_SECRET:
        credentials = base64.b64encode(f"{T212_API_KEY}:{T212_API_SECRET}".encode()).decode()
        headers = {"Authorization": f"Basic {credentials}"}
    else:
        # Fallback: try key alone (some T212 accounts use single key)
        headers = {"Authorization": T212_API_KEY}
    prices  = {}

    # T212 instruments endpoint gives us ticker→instrumentId mapping
    try:
        resp = requests.get(
            f"{T212_BASE}/equity/metadata/instruments",
            headers=headers,
            timeout=30
        )
        if resp.status_code != 200:
            print(f"  T212 instruments error: {resp.status_code}")
            return {}
        instruments = resp.json()
    except Exception as e:
        print(f"  T212 instruments exception: {e}")
        return {}

    # Build ticker → T212 ticker mapping
    # T212 format is "TICKER_US_EQ" — strip suffix
    t212_map = {}
    for inst in instruments:
        raw    = inst.get("ticker", "")
        ticker = raw.split("_")[0] if "_" in raw else raw
        if ticker in set(tickers):
            t212_map[ticker] = raw

    print(f"  T212 matched {len(t212_map)}/{len(tickers)} tickers")

    if not t212_map:
        return {}

    # Fetch prices in batches via T212 market data endpoint
    # T212 /equity/history/value/{ticker} gives last price
    matched = list(t212_map.keys())
    fetched = 0

    for ticker in matched:
        t212_ticker = t212_map[ticker]
        try:
            r = requests.get(
                f"{T212_BASE}/equity/metadata/instruments/{t212_ticker}",
                headers=headers,
                timeout=10
            )
            if r.status_code == 200:
                data  = r.json()
                price = data.get("lastTraded") or data.get("buyPrice") or data.get("currentPrice")
                if price and float(price) > 0:
                    prices[ticker] = float(price)
                    fetched += 1
        except Exception:
            continue

        # Brief pause every 100 requests
        if fetched > 0 and fetched % 100 == 0:
            time.sleep(0.5)

    print(f"  T212 prices fetched: {fetched}")
    return prices


# ── OHLCV download ────────────────────────────────────────────────────────────

def _download_batch(tickers_batch, start_dt, end_dt):
    try:
        raw = yf.download(
            tickers_batch,
            start=start_dt,
            end=end_dt,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        return {}

    results = {}
    if len(tickers_batch) == 1:
        t = tickers_batch[0]
        if not raw.empty:
            results[t] = raw
        return results

    for t in tickers_batch:
        try:
            ticker_df = raw[t].dropna(how="all")
            if not ticker_df.empty:
                results[t] = ticker_df
        except (KeyError, TypeError):
            continue

    return results


def fetch_all_history(tickers, start_dt, end_dt):
    """Sequential with sleep — avoids yfinance rate limits."""
    batches  = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    all_data = {}

    for i, batch in enumerate(batches):
        try:
            result = _download_batch(batch, start_dt, end_dt)
            all_data.update(result)
        except Exception:
            pass
        if i > 0 and i % 5 == 0:
            time.sleep(1)

    return all_data


# ── D drive loaders (local only) ──────────────────────────────────────────────

def _load_d_drive_rsi(ticker):
    safe = ticker.replace(".", "-")
    path = os.path.join(D_TECH_DIR, f"{safe}_rsi.csv")
    if not os.path.exists(path):
        return None
    try:
        df     = pd.read_csv(path, index_col=0, parse_dates=True)
        col    = [c for c in df.columns if "rsi" in c.lower()]
        if not col:
            return None
        return round(float(df[col[0]].dropna().iloc[-1]), 1)
    except Exception:
        return None


def _load_d_drive_macd(ticker):
    safe = ticker.replace(".", "-")
    path = os.path.join(D_TECH_DIR, f"{safe}_macd.csv")
    if not os.path.exists(path):
        return None
    try:
        df        = pd.read_csv(path, index_col=0, parse_dates=True)
        macd_cols = [c for c in df.columns if "macd" in c.lower() and "signal" not in c.lower()]
        sig_cols  = [c for c in df.columns if "signal" in c.lower()]
        if not macd_cols or not sig_cols:
            return None
        return "bullish" if float(df[macd_cols[0]].dropna().iloc[-1]) > float(df[sig_cols[0]].dropna().iloc[-1]) else "bearish"
    except Exception:
        return None


# ── Snapshot writer ───────────────────────────────────────────────────────────

def save_snapshot(df, utc_hour):
    """
    Save a timestamped snapshot of the scored universe.
    Stored as output/premarket_HH.csv — one file per scan time per day.
    These accumulate through the morning so the dashboard can show
    score trends: rising = building momentum, fading = skip.
    """
    filename = f"output/premarket_{utc_hour:02d}.csv"
    df_snap  = df.copy()
    df_snap["snapshot_utc"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    df_snap.to_csv(filename, index=False)
    print(f"  Snapshot saved: {filename} ({len(df_snap)} tickers)")


def load_previous_snapshot(utc_hour):
    """
    Load the most recent earlier snapshot for score trend comparison.
    Returns None if no earlier snapshot exists today.
    """
    for earlier_hour in [utc_hour - 1, utc_hour - 2, utc_hour - 3]:
        if earlier_hour < 0:
            continue
        path = f"output/premarket_{earlier_hour:02d}.csv"
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if "snapshot_utc" in df.columns:
                    snap_time = pd.to_datetime(df["snapshot_utc"].iloc[0])
                    if snap_time.date() == datetime.utcnow().date():
                        return df.set_index("ticker")["score"]
            except Exception:
                continue
    return None


# ── Main builder ──────────────────────────────────────────────────────────────

def build_universe(target_date=None):
    tickers = load_tickers()

    if target_date is None:
        target_date = datetime.utcnow().date()

    utc_hour = datetime.utcnow().hour
    end_dt   = datetime.combine(target_date, datetime.min.time())
    start_dt = end_dt - timedelta(days=20)

    # Step 1: OHLCV history via yfinance
    print(f"Downloading OHLCV for {len(tickers)} tickers in batches of {BATCH_SIZE}...")
    hist_data = fetch_all_history(tickers, start_dt, end_dt)
    print(f"  → History returned for {len(hist_data)} tickers")

    # Step 2: Apply price/volume filters
    eligible     = []
    ohlcv_records = {}

    for t, hist in hist_data.items():
        try:
            if len(hist) < MIN_HISTORY:
                continue
            yesterday_close  = float(hist["Close"].iloc[-1])
            yesterday_volume = float(hist["Volume"].iloc[-1])

            if yesterday_close < MIN_PRICE or yesterday_close > MAX_PRICE:
                continue

            last_10       = hist.tail(10)
            avg_volume_10d = float(last_10["Volume"].mean())

            if avg_volume_10d < MIN_AVG_VOL:
                continue

            high_10d       = float(last_10["High"].max())
            low_10d        = float(last_10["Low"].min())
            atr_10d        = float((last_10["High"] - last_10["Low"]).mean())
            trend_5d       = int((hist["Close"].diff() > 0).tail(5).sum())
            breakout_score = (
                (yesterday_close - low_10d) / (high_10d - low_10d)
                if high_10d != low_10d else 0.0
            )
            volatility_score = atr_10d / yesterday_close if yesterday_close else 0.0

            eligible.append(t)
            ohlcv_records[t] = {
                "yesterday_close":  yesterday_close,
                "yesterday_volume": yesterday_volume,
                "avg_volume_10d":   avg_volume_10d,
                "trend_5d":         trend_5d,
                "breakout_score":   breakout_score,
                "atr_10d":          atr_10d,
                "volatility_score": volatility_score,
            }
        except Exception:
            continue

    print(f"  → {len(eligible)} tickers passed price/volume filters")

    # Step 3: Premarket prices via T212 API (primary)
    print(f"Fetching premarket prices via T212 API for {len(eligible)} tickers...")
    t212_prices = fetch_t212_prices(eligible)

    # Step 4: D drive RSI/MACD (local only)
    d_drive_available = os.path.exists(D_TECH_DIR)
    if d_drive_available:
        print("D drive technicals available — loading RSI/MACD...")
    else:
        print("D drive not available (GitHub Actions) — RSI/MACD will be None")

    # Step 5: Build records
    records = []
    for t in eligible:
        try:
            base = ohlcv_records[t]
            yc   = base["yesterday_close"]
            yv   = base["yesterday_volume"]
            av   = base["avg_volume_10d"]

            # Use T212 price if available, else fall back to yesterday's close
            pm_price  = t212_prices.get(t, None)
            has_gap   = pm_price is not None and pm_price > 0
            if not has_gap:
                pm_price = yc

            gap_pct        = (pm_price - yc) / yc if has_gap else 0.0
            rvol           = yv / av if av else 0.0
            premarket_rvol = 0.0  # T212 doesn't give premarket volume directly

            rsi_val  = _load_d_drive_rsi(t)  if d_drive_available else None
            macd_sig = _load_d_drive_macd(t) if d_drive_available else None

            records.append({
                "ticker":           t,
                "premarket_price":  round(pm_price, 4),
                "yesterday_close":  round(yc, 4),
                "gap_pct":          float(gap_pct),
                "rvol":             float(rvol),
                "premarket_rvol":   float(premarket_rvol),
                "avg_volume_10d":   int(av),
                "trend_5d":         int(base["trend_5d"]),
                "breakout_score":   float(base["breakout_score"]),
                "atr_10d":          float(base["atr_10d"]),
                "volatility_score": float(base["volatility_score"]),
                "rsi":              rsi_val,
                "macd_signal":      macd_sig,
                "price_source":     "t212" if has_gap else "prev_close",
            })
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Step 6: Normalise
    gap_min,   gap_max   = _safe_min_max(df["gap_pct"])
    rvol_min,  rvol_max  = _safe_min_max(df["rvol"])
    brk_min,   brk_max   = _safe_min_max(df["breakout_score"])
    vol_min,   vol_max   = _safe_min_max(df["volatility_score"])

    df["norm_gap"]        = _normalize(df["gap_pct"],        gap_min,  gap_max)
    df["norm_rvol"]       = _normalize(df["rvol"],           rvol_min, rvol_max)
    df["norm_pre_rvol"]   = df["norm_rvol"]
    df["norm_breakout"]   = _normalize(df["breakout_score"], brk_min,  brk_max)
    df["norm_volatility"] = _normalize(df["volatility_score"], vol_min, vol_max)
    df["norm_trend"]      = df["trend_5d"] / 5.0

    pm_raw = df["norm_gap"].clip(lower=0) + df["norm_rvol"].clip(lower=0)
    pm_min, pm_max = _safe_min_max(pm_raw)
    df["premarket_momentum"] = _normalize(pm_raw, pm_min, pm_max)

    # Step 7: RSI factor
    def rsi_factor(rsi):
        if rsi is None:      return 0.5
        if rsi < RSI_AVOID_MAX:  return 0.0
        elif rsi <= RSI_STRONG_MIN: return 0.3
        elif rsi <= RSI_STRONG_MAX: return 1.0
        elif rsi <= 80:      return 0.6
        else:                return 0.2

    df["rsi_factor"]  = df["rsi"].apply(rsi_factor)
    df["macd_factor"] = df["macd_signal"].map({"bullish": 1.0, "bearish": 0.0}).fillna(0.5)

    # Step 8: Score
    base_score = (
        5.0 * df["premarket_momentum"] +
        3.0 * df["norm_gap"] +
        2.0 * df["norm_breakout"] +
        1.0 * df["norm_rvol"] +
        1.0 * df["norm_trend"] +
        0.5 * df["norm_volatility"] +
        1.5 * df["rsi_factor"] +
        1.0 * df["macd_factor"]
    )

    score = base_score.copy()
    score[df["gap_pct"] < 0]  = 0.0
    score[df["gap_pct"] == 0] = score[df["gap_pct"] == 0].clip(upper=9.0)

    # Hard RSI block (only when D drive data available)
    if df["rsi"].notna().any():
        score[(df["rsi"].notna()) & (df["rsi"] < RSI_AVOID_MAX)] = 0.0

    df["score"] = score.round(4)

    # RSI label for dashboard
    def rsi_label(rsi):
        if rsi is None:         return "—"
        if rsi < 30:            return "oversold"
        elif rsi < RSI_AVOID_MAX:   return "weak"
        elif rsi <= RSI_STRONG_MIN: return "neutral"
        elif rsi <= RSI_STRONG_MAX: return "strong ✓"
        else:                   return "overbought"

    df["rsi_label"] = df["rsi"].apply(rsi_label)

    # Step 9: Score trend vs previous snapshot
    prev_scores = load_previous_snapshot(utc_hour)
    if prev_scores is not None:
        df["score_prev"]  = df["ticker"].map(prev_scores)
        df["score_trend"] = (df["score"] - df["score_prev"]).round(3)
        df["trend_dir"]   = df["score_trend"].apply(
            lambda x: "rising" if x > 0.3 else ("fading" if x < -0.3 else "steady")
            if pd.notna(x) else "new"
        )
    else:
        df["score_prev"]  = None
        df["score_trend"] = None
        df["trend_dir"]   = "new"

    df["scan_time_utc"] = datetime.utcnow().strftime("%H:%M:%S")
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # Step 10: Save snapshot
    save_snapshot(df, utc_hour)

    return df


if __name__ == "__main__":
    df = build_universe()
    if not df.empty:
        df.to_csv("output/universe.csv", index=False)
        print(f"\nUniverse written — {len(df)} tickers scored")

        # Show top candidates with T212 prices
        t212_count = (df["price_source"] == "t212").sum()
        print(f"Tickers with live T212 price: {t212_count}")

        top = df[df["score"] > 7].head(10)
        if not top.empty:
            print("\nTop candidates (score > 7):")
            cols = ["ticker","score","gap_pct","rvol","breakout_score","trend_dir","price_source"]
            print(top[cols].to_string(index=False))
        else:
            print("\nNo tickers above score 7 today")
    else:
        print("No data written")
