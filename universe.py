import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
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

# RSI filter thresholds (from 1.8M signal backtest)
RSI_STRONG_MIN  = 50    # RSI above this = good signal
RSI_STRONG_MAX  = 70    # RSI above this = overbought risk
RSI_AVOID_MAX   = 40    # RSI below this = skip (39% WR in backtest)

# D drive paths — only available on home PC, not GitHub Actions
D_TECH_DIR = r"D:\GarAI\data\technicals"
# ─────────────────────────────────────────────────────────────────────────────


def _load_d_drive_rsi(ticker):
    """Load RSI from D drive — returns float or None."""
    safe = ticker.replace(".", "-")
    path = os.path.join(D_TECH_DIR, f"{safe}_rsi.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        rsi_cols = [c for c in df.columns if "rsi" in c.lower()]
        if not rsi_cols:
            return None
        return round(float(df[rsi_cols[0]].dropna().iloc[-1]), 1)
    except Exception:
        return None


def _load_d_drive_macd(ticker):
    """Load MACD signal from D drive — returns 'bullish', 'bearish', or None."""
    safe = ticker.replace(".", "-")
    path = os.path.join(D_TECH_DIR, f"{safe}_macd.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        macd_cols = [c for c in df.columns if "macd" in c.lower()
                     and "signal" not in c.lower()]
        sig_cols  = [c for c in df.columns if "signal" in c.lower()]
        if not macd_cols or not sig_cols:
            return None
        macd_val = float(df[macd_cols[0]].dropna().iloc[-1])
        sig_val  = float(df[sig_cols[0]].dropna().iloc[-1])
        return "bullish" if macd_val > sig_val else "bearish"
    except Exception:
        return None


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
    Fetch pre-market price for a single ticker.
    Uses yf.download with prepost=True — more reliable than .info preMarketPrice.
    Returns (ticker, price_or_None, volume).
    """
    try:
        import pytz
        from datetime import datetime, timedelta, date
        UTC = pytz.utc
        BST = pytz.timezone("Europe/London")

        # Download last 2 days with pre/post market data at 1m interval
        raw = yf.download(
            ticker_str,
            period="2d",
            interval="1m",
            prepost=True,
            progress=False,
            timeout=15,
        )

        if raw is None or raw.empty:
            return ticker_str, None, 0

        now_utc = datetime.now(UTC)
        # Pre-market = before 13:30 UTC (14:30 BST = NYSE open)
        pm_cutoff = now_utc.replace(hour=13, minute=30, second=0, microsecond=0)
        today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

        # Get today's pre-market candles
        pm = raw[(raw.index >= today_start) & (raw.index < pm_cutoff)]

        if pm.empty or pm["Volume"].sum() < 1000:
            return ticker_str, None, 0

        pm_price = float(pm["Close"].dropna().iloc[-1])
        pm_volume = int(pm["Volume"].sum())
        return ticker_str, pm_price, pm_volume

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
    d_drive_available = os.path.exists(D_TECH_DIR)
    if d_drive_available:
        print(f"D drive technicals available — loading RSI/MACD...")
    else:
        print(f"D drive not available (GitHub Actions) — RSI/MACD will be None")

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

            gap_pct        = (pm_price - yc) / yc
            rvol           = yv / avg_vol if avg_vol else 0.0
            premarket_rvol = pm_volume / avg_vol if avg_vol else 0.0

            # RSI and MACD from D drive (home PC only)
            rsi_val  = _load_d_drive_rsi(t)  if d_drive_available else None
            macd_sig = _load_d_drive_macd(t) if d_drive_available else None

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
                "rsi":              rsi_val,
                "macd_signal":      macd_sig,
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

    # ── RSI scoring factor ────────────────────────────────────────────────────
    # RSI 50-70 = ideal (score boost), 70-80 = acceptable, above 80 = penalty,
    # below 40 = penalty. None (D drive unavailable) = neutral (0.5).
    def rsi_factor(rsi):
        if rsi is None:
            return 0.5    # neutral when data not available
        if rsi < RSI_AVOID_MAX:
            return 0.0    # 39% WR in backtest — penalise heavily
        elif rsi <= RSI_STRONG_MIN:
            return 0.3    # below ideal but not terrible
        elif rsi <= RSI_STRONG_MAX:
            return 1.0    # sweet spot — full bonus
        elif rsi <= 80:
            return 0.6    # slightly overbought
        else:
            return 0.2    # very overbought — fade risk

    df["rsi_factor"] = df["rsi"].apply(rsi_factor)

    # ── MACD scoring factor ───────────────────────────────────────────────────
    # Bullish = bonus, bearish = penalty, None = neutral
    df["macd_factor"] = df["macd_signal"].map(
        {"bullish": 1.0, "bearish": 0.0}
    ).fillna(0.5)

    # ── Composite score ───────────────────────────────────────────────────────
    # Original 6 signals + RSI + MACD
    # RSI and MACD weighted at 1.5 each (meaningful but not dominant)
    # RSI is from D drive so only available at home — when None, neutral 0.5
    # doesn't distort relative rankings
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

    # Hard RSI filter — zero out signals with RSI below avoid threshold
    # Only applied when D drive data is available (rsi is not None)
    if df["rsi"].notna().any():
        score[(df["rsi"].notna()) & (df["rsi"] < RSI_AVOID_MAX)] = 0.0

    df["score"] = score.round(4)

    # Add RSI band label for dashboard display
    def rsi_label(rsi):
        if rsi is None:
            return "—"
        if rsi < 30:
            return "oversold"
        elif rsi < RSI_AVOID_MAX:
            return "weak"
        elif rsi <= RSI_STRONG_MIN:
            return "neutral"
        elif rsi <= RSI_STRONG_MAX:
            return "strong ✓"
        else:
            return "overbought"

    df["rsi_label"] = df["rsi"].apply(rsi_label)

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
