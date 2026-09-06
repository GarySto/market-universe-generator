"""
Market Universe Generator — premarket scoring for The Game
==========================================================
Builds output/universe.csv: a ranked list of US equities scored for
premarket momentum. Consumed by The Game (via premarket_connector.py in
the private trading repo) and by the public Streamlit dashboard.

=============================================================
6 SEP 2026 — REAL PREMARKET GAP, TWO-PASS FETCH
=============================================================

Root cause (diagnosed 9 Aug, fixed today): `ticker.fast_info` does not
actually carry a `preMarketPrice` field — it silently returns None, so
premarket_price has ALWAYS fallen back to yesterday_close, making
gap_pct exactly 0.0 for every ticker, every day. The Game is a
premarket-gap strategy that had never once measured a real gap.

Fix: after the pass-1 loop below scores the whole universe with no gap
component (unchanged), the TOP_N_FOR_GAP_FETCH highest-scoring tickers
each get ONE extra yfinance call: `ticker.history(period="1d",
interval="1m", prepost=True)`, which actually contains premarket bars.
The last available bar's close vs. yesterday's close is a real gap_pct.
Those rows are then rescored with the gap component included and
UNSCALED (gap_available=True, score_scale=1.0) — the achievable max
for them is genuinely 10.0 now, no rescue needed. Rows outside the top
N never get a real gap and keep the honest rescaled score exactly as
before (gap_available=False).

Why top-N and not the whole universe: fetching 1-minute prepost bars
for every ticker in the universe risked blowing the GitHub Actions job
time limit. Scoping to the top ~100 by no-gap score keeps the extra
cost to roughly 100 additional calls per run.

Expected effect: SCORE_THRESHOLD (7.0, set in place_demo_orders_game.py)
now means something real. Previously a "perfect" no-gap score (RSI 100,
rvol 4.0+, breakout 1.0) rescaled to exactly 10.0 and cleared 7.0 with
zero actual gap. Now clearing 7.0 requires a real, meaningful gap on
top of the other components — expect The Game to qualify a candidate
LESS often than before. That is the fix working, not a regression.

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

(2) THE SCORE IS NORMALISED WHEN THE GAP COMPONENT IS UNAVAILABLE.

    For any ticker outside today's top N (see 6 Sep note above), gap_pct
    is still not measured, so gap_score is 0 and the achievable max for
    that row is 7.0 rather than 10.0. The remaining components are
    rescaled so a "7.0+" threshold on those rows still means "70% of
    what's achievable for them" rather than "a literally perfect score".
    Both numbers are written — score_raw (unscaled) and score
    (normalised, what everything downstream should use).
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
# This now reflects the PASS-1 (no-gap) base calculation only. Rows in
# today's top N get a real gap fetched in pass 2 and their own
# per-row gap_available is overridden to True there — see build_universe().
# Leave this False: it governs the rescale for every row NOT in the
# top N, which never gets a real gap fetch and must stay honestly scaled.
GAP_AVAILABLE = False

# ── Two-pass real premarket gap (6 Sep 2026) ────────────────────────────────
TOP_N_FOR_GAP_FETCH = 100   # how many top-scoring (no-gap) tickers get a
                            # real premarket fetch in pass 2. Scoped to
                            # keep the extra yfinance calls bounded.

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


def fetch_real_premarket(ticker_symbol, prev_close):
    """
    PASS 2 ONLY — called for the top TOP_N_FOR_GAP_FETCH candidates by
    no-gap score. `fast_info` does not carry a real preMarketPrice field
    (see module docstring); `ticker.history(prepost=True)` does return
    actual extended-hours minute bars. Returns (premarket_price, source).

    Falls back to (prev_close, "prev_close_fallback") on any failure or
    empty result — same graceful degradation as pass 1, so a single bad
    fetch just means that one ticker keeps its rescaled no-gap score
    instead of crashing the whole run.
    """
    try:
        pm = yf.Ticker(ticker_symbol).history(
            period="1d", interval="1m", prepost=True
        )
        if pm.empty:
            return prev_close, "prev_close_fallback"
        last_price = float(pm["Close"].iloc[-1])
        if not np.isfinite(last_price) or last_price <= 0:
            return prev_close, "prev_close_fallback"
        return last_price, "live_premarket_1m"
    except Exception:
        return prev_close, "prev_close_fallback"


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

            # ---------- 4. Premarket data — PASS 1 (no real gap yet) -------
            # See module docstring, 6 Sep 2026 note. Pass 1 always uses the
            # yesterday_close fallback; pass 2 below (after this loop)
            # overwrites premarket_price/gap_pct for the top N tickers only
            # with a real fetch via fetch_real_premarket().
            premarket_price = yesterday_close
            premarket_source = "prev_close_fallback"
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
            # With GAP_AVAILABLE False, gap_score is structurally 0 here in
            # pass 1, so the achievable maximum is 7.0 rather than 10.0.
            # Rescale so that a 7.0 threshold still means "70% of what's
            # achievable" instead of "a literally perfect score". Pass 2
            # below overwrites this for the top N tickers once a real gap
            # is fetched.
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
                # Intermediate scaled components (6 Sep 2026) — kept so pass
                # 2 can rebuild score_raw exactly without recomputing from
                # already-rounded display values.
                "_rsi_score": rsi_score,
                "_rvol_score": rvol_score,
                "_breakout_component": breakout_component,
            })

        except Exception as e:
            print(f"Skipping {t}: error {e}")
            continue

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("score", ascending=False)

    # ---------------------------------------------------------
    # PASS 2 (6 Sep 2026): real premarket gap for the top candidates
    # ---------------------------------------------------------
    if not df.empty:
        top_slice = df.head(TOP_N_FOR_GAP_FETCH)
        print(f"Pass 2: fetching real premarket data for top "
              f"{len(top_slice)} candidate(s) by no-gap score...")

        for idx in top_slice.index:
            row = df.loc[idx]
            real_pm_price, pm_source = fetch_real_premarket(
                row["ticker"], row["prev_close"]
            )
            real_gap_pct = (real_pm_price - row["prev_close"]) / row["prev_close"]

            gap_score = min(abs(real_gap_pct) * 100 * 0.3, GAP_MAX)
            # Real gap now available for this row -> full 10.0 scale used
            # directly, no rescale needed.
            new_score_raw = (row["_rsi_score"] + gap_score
                              + row["_rvol_score"] + row["_breakout_component"])
            new_score = new_score_raw
            new_score_scale = 1.0

            df.loc[idx, "premarket_price"] = round(float(real_pm_price), 4)
            df.loc[idx, "premarket_source"] = pm_source
            df.loc[idx, "gap_pct"] = real_gap_pct
            df.loc[idx, "score_raw"] = round(new_score_raw, 4)
            df.loc[idx, "score"] = round(new_score, 4)
            df.loc[idx, "score_scale"] = round(new_score_scale, 4)
            df.loc[idx, "gap_available"] = True

        # Real gaps can reorder the top slice relative to itself (though
        # never bring in a row from outside it — that row never got a
        # fetch, see TOP_N_FOR_GAP_FETCH note above).
        df = df.sort_values("score", ascending=False)

        # Internal helper columns — not part of the public output contract.
        df = df.drop(columns=["_rsi_score", "_rvol_score", "_breakout_component"])

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
        n_real_gap = int(df["gap_available"].sum()) if "gap_available" in df else 0
        print(f"Top score today: {df['score'].max():.2f} "
              f"(raw {df['score_raw'].max():.2f}, "
              f"{n_real_gap} ticker(s) with a real premarket gap fetched)")
