import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

os.makedirs("output", exist_ok=True)


def load_tickers():
    with open("tickers.txt", "r") as f:
        return [t.strip() for t in f if t.strip() and t.strip() != "Ticker"]


def _safe_min_max(series):
    """Return (min, max) with protection against constant or empty series."""
    if series is None or len(series) == 0:
        return 0.0, 1.0
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0, 1.0
    mn, mx = float(s.min()), float(s.max())
    if mx == mn:
        # avoid divide-by-zero; treat as flat feature
        return mn, mn + 1.0
    return mn, mx


def _normalize(series, mn, mx):
    return (series - mn) / (mx - mn) if mx != mn else series * 0.0


def build_universe(target_date=None):
    tickers = load_tickers()
    records = []

    if target_date is None:
        target_date = datetime.utcnow().date()

    end_dt = datetime.combine(target_date, datetime.min.time())
    start_dt = end_dt - timedelta(days=20)

    for t in tickers:
        try:
            ticker = yf.Ticker(t)

            # ── Historical daily data ────────────────────────────────────────
            hist = ticker.history(start=start_dt, end=end_dt)
            if hist.empty or len(hist) < 10:
                print(f"Skipping {t}: insufficient history")
                continue

            yesterday_close = float(hist["Close"].iloc[-1])
            yesterday_volume = float(hist["Volume"].iloc[-1])

            # Minimum price filter — penny stocks have wide spreads in premarket
            if yesterday_close < 1.0:
                print(f"Skipping {t}: price ${yesterday_close:.2f} below $1 minimum")
                continue

            last_10 = hist.tail(10)
            avg_volume_10d = float(last_10["Volume"].mean())
            high_10d = float(last_10["High"].max())
            low_10d = float(last_10["Low"].min())
            atr_10d = float((last_10["High"] - last_10["Low"]).mean())
            trend_5d = int((hist["Close"].diff() > 0).tail(5).sum())

            # ── Premarket data via .info dict ────────────────────────────────
            premarket_price = None
            premarket_volume = 0

            try:
                info = ticker.info
                premarket_price = info.get("preMarketPrice")
                premarket_volume = info.get("preMarketVolume") or 0
            except Exception:
                pass

            # Fallback: if no premarket price, use yesterday's close (gap = 0)
            if not premarket_price or premarket_price <= 0:
                premarket_price = yesterday_close

            # ── Feature engineering ──────────────────────────────────────────
            gap_pct = (premarket_price - yesterday_close) / yesterday_close
            rvol = yesterday_volume / avg_volume_10d if avg_volume_10d else 0
            premarket_rvol = premarket_volume / avg_volume_10d if avg_volume_10d else 0

            breakout_score = (
                (yesterday_close - low_10d) / (high_10d - low_10d)
                if high_10d != low_10d else 0
            )
            volatility_score = atr_10d / yesterday_close if yesterday_close else 0

            # Store raw features; scoring comes after we have the full universe
            records.append({
                "ticker":           t,
                "premarket_price":  round(premarket_price, 2),
                "premarket_volume": int(premarket_volume),
                "gap_pct":          float(gap_pct),
                "rvol":             float(rvol),
                "premarket_rvol":   float(premarket_rvol),
                "avg_volume_10d":   int(avg_volume_10d),
                "trend_5d":         int(trend_5d),
                "breakout_score":   float(breakout_score),
                "atr_10d":          float(atr_10d),
                "volatility_score": float(volatility_score),
            })

        except Exception as e:
            print(f"Skipping {t}: {e}")
            continue

    if not records:
        print("Warning: no records generated")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # ── Normalization (data‑science approach) ───────────────────────────────
    # We normalize per‑day across the universe so weights are meaningful.

    gap_min, gap_max = _safe_min_max(df["gap_pct"])
    rvol_min, rvol_max = _safe_min_max(df["rvol"])
    pre_rvol_min, pre_rvol_max = _safe_min_max(df["premarket_rvol"])
    brk_min, brk_max = _safe_min_max(df["breakout_score"])
    vol_min, vol_max = _safe_min_max(df["volatility_score"])

    # trend_5d is already bounded [0,5], so we just scale by 5
    df["norm_gap"] = _normalize(df["gap_pct"], gap_min, gap_max)
    df["norm_rvol"] = _normalize(df["rvol"], rvol_min, rvol_max)
    df["norm_premarket_rvol"] = _normalize(df["premarket_rvol"], pre_rvol_min, pre_rvol_max)
    df["norm_breakout"] = _normalize(df["breakout_score"], brk_min, brk_max)
    df["norm_volatility"] = _normalize(df["volatility_score"], vol_min, vol_max)
    df["norm_trend"] = df["trend_5d"] / 5.0

    # Premarket momentum proxy (for now: gap + premarket RVOL, normalized)
    # This is a lightweight approximation that still respects your edge:
    # strong gap + strong premarket volume.
    pm_raw = df["norm_gap"].clip(lower=0) + df["norm_premarket_rvol"].clip(lower=0)
    pm_min, pm_max = _safe_min_max(pm_raw)
    df["premarket_momentum"] = _normalize(pm_raw, pm_min, pm_max)

    # ── Scoring (Option A — momentum‑first) ────────────────────────────────
    base_score = (
        5.0 * df["premarket_momentum"] +
        3.0 * df["norm_gap"] +
        2.0 * df["norm_breakout"] +
        1.0 * df["norm_rvol"] +
        1.0 * df["norm_trend"] +
        0.5 * df["norm_volatility"]
    )

    # Long‑only rules:
    #  - Negative gap → score = 0
    #  - Zero gap → cap score at 9 (no‑story days shouldn't top the list)
    score = base_score.copy()

    score[df["gap_pct"] < 0] = 0.0
    score[(df["gap_pct"] == 0)] = score[(df["gap_pct"] == 0)].clip(upper=9.0)

    df["score"] = score.round(4)

    # Final ordering
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
            print(top5[["ticker", "score", "gap_pct", "rvol",
                         "premarket_rvol", "premarket_momentum",
                         "breakout_score"]].to_string(index=False))
        else:
            print("\nNo tickers above score 9 today")
    else:
        print("No data written")
