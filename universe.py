import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

os.makedirs("output", exist_ok=True)

# ── Tuning constants ──────────────────────────────────────────────────────────
BATCH_SIZE   = 100   # tickers per yf.download() call
MAX_WORKERS  = 5     # parallel threads for .info (premarket price) fetches
MIN_PRICE    = 1.50
MAX_PRICE    = 75.0
MIN_AVG_VOL  = 200_000
MIN_HISTORY  = 10
# ─────────────────────────────────────────────────────────────────────────────


def load_tickers():
    with open("tickers.txt", "r") as f:
        return [t.strip() for t in f if t.strip() and t.strip() != "Ticker"]


def _safe_min_max(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0, 1.0
    mn, mx = float(s.min()), float(s.max())
    if mx == mn:
        return mn, mn + 1.0
    return mn, mx


def _normalize(series, mn, mx):
    return (series - mn) / (mx - mn) if mx != mn else series * 0.0


# ── Step 1: bulk OHLCV download in batches ───────────────────────────────────

def _download_batch(tickers_batch, start_dt, end_dt):
    """
    Download OHLCV history for a list of tickers in one yf.download() call.
    Returns a dict: {ticker: DataFrame} for tickers that returned data.
    """
    try:
        raw = yf.download(
            tickers_batch,
            start=start_dt,
            end=end_dt,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception:
        return {}

    results = {}

    # Single ticker: yf.download returns a flat DataFrame, not grouped
    if len(tickers_batch) == 1:
        t = tickers_batch[0]
        if not raw.empty:
            results[t] = raw
        return results

    # Multiple tickers: top-level columns are ticker symbols
    for t in tickers_batch:
        try:
            ticker_df = raw[t].dropna(how="all")
            if not ticker_df.empty:
                results[t] = ticker_df
        except (KeyError, TypeError):
            continue

    return results


def fetch_all_history(tickers, start_dt, end_dt):
    """
    Split tickers into batches of BATCH_SIZE and download in parallel.
    Returns a dict: {ticker: DataFrame}
    """
    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    all_data = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_download_batch, b, start_dt, end_dt): b for b in batches}
        for future in as_completed(futures):
            try:
                all_data.update(future.result())
            except Exception:
                continue

    return all_data


# ── Step 2: premarket info fetches (parallel, one per ticker) ────────────────

def _fetch_premarket(ticker_str):
    """
    Fetch preMarketPrice and preMarketVolume for a single ticker via .info.
    Returns (ticker, price_or_None, volume).
    """
    try:
        info = yf.Ticker(ticker_str).info
        price  = info.get("preMarketPrice")
        volume = info.get("preMarketVolume") or 0
        return ticker_str, price, volume
    except Exception:
        return ticker_str, None, 0


def fetch_all_premarket(tickers):
    """
    Fetch premarket data for all tickers in parallel.
    Returns a dict: {ticker: (price_or_None, volume)}
    """
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_premarket, t): t for t in tickers}
        for future in as_completed(futures):
            try:
                t, price, vol = future.result()
                results[t] = (price, vol)
            except Exception:
                continue
    return results


# ── Main builder ──────────────────────────────────────────────────────────────

def build_universe(target_date=None):
    tickers = load_tickers()

    if target_date is None:
        target_date = datetime.utcnow().date()

    end_dt   = datetime.combine(target_date, datetime.min.time())
    start_dt = end_dt - timedelta(days=20)

    print(f"Downloading OHLCV for {len(tickers)} tickers in batches of {BATCH_SIZE}...")
    hist_data = fetch_all_history(tickers, start_dt, end_dt)
    print(f"  → History returned for {len(hist_data)} tickers")

    # First pass: apply price/volume filters using OHLCV data only
    # Build a list of tickers that pass before making expensive .info calls
    eligible = []
    ohlcv_records = {}

    for t, hist in hist_data.items():
        try:
            if len(hist) < MIN_HISTORY:
                continue

            yesterday_close  = float(hist["Close"].iloc[-1])
            yesterday_volume = float(hist["Volume"].iloc[-1])

            if yesterday_close < MIN_PRICE or yesterday_close > MAX_PRICE:
                continue

            last_10        = hist.tail(10)
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

    # Second pass: fetch premarket data only for eligible tickers
    print(f"Fetching premarket data for {len(eligible)} eligible tickers...")
    premarket_data = fetch_all_premarket(eligible)
    print(f"  → Premarket data returned for {len(premarket_data)} tickers")

    # Build final records
    records = []
    for t in eligible:
        try:
            base    = ohlcv_records[t]
            yc      = base["yesterday_close"]
            yv      = base["yesterday_volume"]
            avg_vol = base["avg_volume_10d"]

            pm_price, pm_volume = premarket_data.get(t, (None, 0))
            if not pm_price or pm_price <= 0:
                pm_price = yc

            gap_pct       = (pm_price - yc) / yc
            rvol          = yv / avg_vol if avg_vol else 0.0
            premarket_rvol = pm_volume / avg_vol if avg_vol else 0.0

            records.append({
                "ticker":           t,
                "premarket_price":  round(pm_price, 2),
                "yesterday_close":  round(yc, 2),
                "premarket_volume": int(pm_volume),
                "gap_pct":          float(gap_pct),
                "rvol":             float(rvol),
                "premarket_rvol":   float(premarket_rvol),
                "avg_volume_10d":   int(avg_vol),
                "trend_5d":         int(base["trend_5d"]),
                "breakout_score":   float(base["breakout_score"]),
                "atr_10d":          float(base["atr_10d"]),
                "volatility_score": float(base["volatility_score"]),
            })
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # ── Normalisation (min-max across today's universe) ───────────────────────
    gap_min,     gap_max     = _safe_min_max(df["gap_pct"])
    rvol_min,    rvol_max    = _safe_min_max(df["rvol"])
    pre_min,     pre_max     = _safe_min_max(df["premarket_rvol"])
    brk_min,     brk_max     = _safe_min_max(df["breakout_score"])
    vol_min,     vol_max     = _safe_min_max(df["volatility_score"])

    df["norm_gap"]        = _normalize(df["gap_pct"],          gap_min,  gap_max)
    df["norm_rvol"]       = _normalize(df["rvol"],             rvol_min, rvol_max)
    df["norm_pre_rvol"]   = _normalize(df["premarket_rvol"],   pre_min,  pre_max)
    df["norm_breakout"]   = _normalize(df["breakout_score"],   brk_min,  brk_max)
    df["norm_volatility"] = _normalize(df["volatility_score"], vol_min,  vol_max)
    df["norm_trend"]      = df["trend_5d"] / 5.0

    pm_raw = df["norm_gap"].clip(lower=0) + df["norm_pre_rvol"].clip(lower=0)
    pm_min, pm_max = _safe_min_max(pm_raw)
    df["premarket_momentum"] = _normalize(pm_raw, pm_min, pm_max)

    # ── Scoring ───────────────────────────────────────────────────────────────
    base_score = (
        5.0 * df["premarket_momentum"] +
        3.0 * df["norm_gap"] +
        2.0 * df["norm_breakout"] +
        1.0 * df["norm_rvol"] +
        1.0 * df["norm_trend"] +
        0.5 * df["norm_volatility"]
    )

    score = base_score.copy()
    score[df["gap_pct"] < 0]  = 0.0
    score[df["gap_pct"] == 0] = score[df["gap_pct"] == 0].clip(upper=9.0)

    df["score"] = score.round(4)

    # Timestamp — seconds included so every run always produces a git diff
    df["scan_time_utc"] = datetime.utcnow().strftime("%H:%M:%S")

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = build_universe()
    if not df.empty:
        df.to_csv("output/universe.csv", index=False)
        print(f"\nUniverse written — {len(df)} tickers scored")
        top5 = df[df["score"] > 9].head(5)
        if not top5.empty:
            print("\nTop candidates (score > 9):")
            print(top5[["ticker", "score", "gap_pct", "premarket_rvol",
                         "premarket_momentum", "breakout_score"]].to_string(index=False))
        else:
            print("\nNo tickers above score 9 today")
    else:
        print("No data written")
