"""
Market Universe Generator — premarket scoring for The Game
==========================================================
Builds output/universe.csv: a ranked list of US equities scored for
premarket momentum. Consumed by The Game (via premarket_connector.py in
the private trading repo) and by the public Streamlit dashboard.

=============================================================
9 AUG 2026 — TWO CHANGES, BOTH ABOUT TRADEABILITY
=============================================================

(1) TRADEABILITY FLOORS ADDED (MIN_PRICE / MIN_AVG_DOLLAR_VOL).

    Before today there was no price floor, no liquidity floor and no
    market-cap filter anywhere in this file. Nothing stopped the system
    from ranking, surfacing and then trying to buy sub-penny stocks.

    It did exactly that. The Game's only two order attempts, ever:
        5 Aug — CALA at $0.0003, sized at 166,666 shares
        7 Aug — ACAN at $0.0002, sized at 250,000 shares
    Both rejected by T212 with "instrument-close-only-mode": T212 lets you
    SELL an illiquid instrument you already hold, but will not let you
    OPEN a new position in one.

    Names failing the floors are now written to output/universe_excluded.csv
    instead of the main file, so they can never reach a downstream consumer,
    but you can still see what was filtered and why.

(2) THE SCORE IS NOW NORMALISED WHEN THE GAP COMPONENT IS UNAVAILABLE.

    gap_pct has been exactly 0.0 for every ticker, every day, since this
    file was written — see the premarket note below. That meant gap_score
    was permanently 0, and the highest score the formula could physically
    produce was 7.0 (4 RSI + 0 gap + 2 rvol + 1 breakout).

    The Game's SCORE_THRESHOLD was also 7.0. The threshold equalled the
    ceiling. The only tickers that could ever qualify were literal perfect
    sevens, requiring RSI of exactly 100, rvol of 4.0+, AND a breakout_score
    of exactly 1.0, simultaneously. On the live dashboard you can see this:
    PRKA and ACAN scored exactly 7.0; BMBN scored 6.954 and was excluded
    purely because its rvol was 3.908 instead of 4.0.

    RSI of exactly 100 means zero down-closes in 14 sessions. On a liquid
    stock that essentially never happens. On thin sub-penny names it happens
    constantly — which is precisely why the only two candidates this system
    ever produced were untradeable junk.

    So: when the gap component is unavailable, the remaining components are
    rescaled so the achievable maximum is still 10.0 and a "7.0+" threshold
    keeps meaning "70% of the best score available today". Both numbers are
    written — score_raw (unscaled, comparable to history) and score
    (normalised, what everything downstream should use).

    Restore the gap component properly (two-pass premarket fetch) and
    GAP_AVAILABLE flips to True, the rescale switches itself off, and the
    7.0 threshold reverts to its original meaning with no other changes.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# ── Tradeability floors (9 Aug 2026) ────────────────────────────────────────
# A signal on something you cannot buy is not a signal.
MIN_PRICE           = 1.00        # no sub-$1 stocks. T212 close-only mode.
MIN_AVG_DOLLAR_VOL  = 1_000_000   # 10-day average daily traded value, USD.
                                  # Sub-penny and shell names fail this instantly.

# ── Premarket gap availability ──────────────────────────────────────────────
# Set to True ONLY once a real premarket price source is wired in. While it
# is False, gap_score is known-zero and the score is rescaled to compensate.
GAP_AVAILABLE = False

# Component ceilings — kept as named constants so the rescale below can be
# derived from them rather than hardcoded.
RSI_MAX      = 4.0
GAP_MAX      = 3.0
RVOL_MAX     = 2.0
BREAKOUT_MAX = 1.0
SCORE_MAX    = RSI_MAX + GAP_MAX + RVOL_MAX + BREAKOUT_MAX   # 10.0


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
# MAIN FUNCTION: Build universe for ANY date
# ---------------------------------------------------------
def build_universe(target_date=None):
    tickers = load_tickers()
    records = []
    excluded = []

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

            # ---------- 1b. TRADEABILITY GATE (9 Aug 2026) ----------
            # Runs BEFORE scoring. There is no point ranking something the
            # broker will refuse to sell us.
            avg_dollar_vol_10d = float(avg_volume_10d) * float(yesterday_close)
            reject_reason = None
            if not np.isfinite(yesterday_close) or yesterday_close < MIN_PRICE:
                reject_reason = f"price ${yesterday_close:.4f} below ${MIN_PRICE:.2f} floor"
            elif not np.isfinite(avg_dollar_vol_10d) or avg_dollar_vol_10d < MIN_AVG_DOLLAR_VOL:
                reject_reason = (f"10d avg dollar volume ${avg_dollar_vol_10d:,.0f} "
                                 f"below ${MIN_AVG_DOLLAR_VOL:,.0f} floor")

            if reject_reason:
                excluded.append({
                    "ticker": t,
                    "prev_close": round(float(yesterday_close), 6),
                    "avg_dollar_vol_10d": round(avg_dollar_vol_10d, 2),
                    "reason": reject_reason,
                })
                continue

            # ---------- 2. Trend strength ----------
            trend_5d = (hist["Close"].diff() > 0).tail(5).sum()

            # ---------- 3. RSI(14) ----------
            # Added after the win-rate sweep (10 Jul 2026) found RSI is by
            # far the strongest predictor of this model's actual outcomes —
            # RSI 90+ alone: 76.5% win rate vs ~50% baseline, a 26-point
            # swing.
            rsi_val = calc_rsi(hist["Close"])

            # ---------- 4. Premarket data ----------
            # KNOWN LIMITATION, still open as of 9 Aug 2026.
            #
            # `preMarketPrice` is not a fast_info field at all (it lives on
            # .info, where yfinance returns None for it silently anyway). So
            # this has ALWAYS fallen through to yesterday_close, which makes
            # gap_pct exactly 0.0 for every ticker, every day. The Game is a
            # premarket gap strategy that has never once measured a gap.
            #
            # Fixing the SOURCE properly means a second yfinance call per
            # ticker with prepost=True, which risks blowing the Actions job
            # time limit across the full universe. The scoped fix is a
            # two-pass approach: score the whole universe without gap, then
            # fetch real premarket bars for the top ~100 only.
            #
            # Until that lands, GAP_AVAILABLE stays False and the score
            # rescale below compensates so thresholds stay meaningful.
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

            # ---------- 6. Final score ----------
            # Each component is scaled to a bounded, comparable range before
            # weighting, so the weights actually mean what they say:
            #   rsi_score   0-4   (primary driver — matches the sweep)
            #   gap_score   0-3   (capped — a 10%+ gap already maxes this out)
            #   rvol_score  0-2
            #   breakout    0-1   (already 0-1 by construction)
            rsi_score = min((rsi_val or 0) / 100 * RSI_MAX, RSI_MAX)
            gap_score = min(abs(gap_pct) * 100 * 0.3, GAP_MAX)
            rvol_score = min(rvol * 0.5, RVOL_MAX) if rvol == rvol else 0  # NaN-safe
            breakout_component = breakout_score if breakout_score == breakout_score else 0

            score_raw = rsi_score + gap_score + rvol_score + breakout_component

            # ── Normalisation (9 Aug 2026) ──
            # With GAP_AVAILABLE False, gap_score is structurally 0, so the
            # achievable maximum is 7.0 rather than 10.0. Rescale so that a
            # 7.0 threshold still means "70% of what's achievable" instead of
            # "a literally perfect score".
            if GAP_AVAILABLE:
                score = score_raw
                score_scale = 1.0
            else:
                achievable_max = SCORE_MAX - GAP_MAX          # 7.0
                score_scale = SCORE_MAX / achievable_max      # 10/7 = 1.4286
                score = score_raw * score_scale

            # ---------- 7. Save record ----------
            records.append({
                "ticker": t,
                "score": round(score, 4),
                "score_raw": round(score_raw, 4),
                "score_scale": round(score_scale, 4),
                "gap_available": GAP_AVAILABLE,
                # premarket_price / prev_close — REQUIRED by
                # place_demo_orders_game.py to size a trade. Do not remove.
                "premarket_price": round(float(premarket_price), 4),
                "prev_close": round(float(yesterday_close), 4),
                "premarket_source": premarket_source,
                # Tradeability data — kept so downstream can re-check the
                # floors independently rather than trusting this file blindly.
                "avg_dollar_vol_10d": round(avg_dollar_vol_10d, 2),
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
    if not df.empty:
        df = df.sort_values("score", ascending=False)

    excluded_df = pd.DataFrame(excluded)
    return df, excluded_df


# ---------------------------------------------------------
# When run normally, generate today's universe CSV
# ---------------------------------------------------------
if __name__ == "__main__":
    df, excluded_df = build_universe()

    df.to_csv("output/universe.csv", index=False)
    print(f"Universe written to output/universe.csv — {len(df)} tradeable ticker(s)")

    if not excluded_df.empty:
        excluded_df.to_csv("output/universe_excluded.csv", index=False)
        print(f"Excluded {len(excluded_df)} ticker(s) on tradeability floors "
              f"(price < ${MIN_PRICE:.2f} or 10d avg dollar volume < "
              f"${MIN_AVG_DOLLAR_VOL:,.0f}) — see output/universe_excluded.csv")

    if not df.empty:
        print(f"Top score today: {df['score'].max():.2f} "
              f"(raw {df['score_raw'].max():.2f}, "
              f"gap component {'ON' if GAP_AVAILABLE else 'OFF — score rescaled'})")
