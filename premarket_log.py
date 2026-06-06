"""
GarAI — Pre-market Data Logger
================================
Lives in: github.com/GarySto/market-universe-generator

Runs at 09:00 BST (08:00 UTC) each weekday via GitHub Actions,
BEFORE the market opens and before the main scanner runs.

What it captures (per ticker, per day):
  - Pre-market price and gap % vs previous close
  - Pre-market volume
  - RSI from D drive technicals (local) or skip if not available
  - MACD signal from D drive technicals (local) or skip
  - Regime context from market_events.csv

Output:
  output/premarket_latest.csv  — today's snapshot (read by Streamlit dashboard)
  D:\\GarAI\\data\\premarket\\   — daily archive for backtesting (home PC only)

Why this matters:
  - Builds the pre-market dataset needed for auction model hypotheses
  - Feeds the Streamlit dashboard with gap data before market opens
  - Pairs with intraday scanner: pre-market gap -> intraday continuation check
  - After 4+ weeks of data, enables pre-market -> intraday -> multi-day chain analysis

Data sources:
  - yfinance: pre-market prices (small targeted pulls, well within rate limits)
  - D drive technicals: RSI / MACD (local, no API calls)
  - market_events.csv: regime context (shared with intraday scanner)
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, date, timedelta
import pytz

BST = pytz.timezone("Europe/London")

# ── Config ─────────────────────────────────────────────────────────────────────
TICKERS_FILE  = "tickers.txt"
OUTPUT_DIR    = "output"
OUTPUT_FILE   = os.path.join(OUTPUT_DIR, "premarket_latest.csv")
EVENTS_FILE   = "market_events.csv"

# D drive paths — only available on home PC
# When running via GitHub Actions these will not exist — that's fine, we skip them
D_TECH_DIR    = r"D:\GarAI\data\technicals"
D_PREMARKET   = r"D:\GarAI\data\premarket"

# Scan config
MAX_TICKERS   = 500      # limit for yfinance to avoid rate limits
MIN_VOLUME    = 10_000   # minimum pre-market volume to log
GAP_THRESHOLD = 2.0      # minimum gap % to flag as notable


def load_tickers(max_tickers=MAX_TICKERS):
    for path in [TICKERS_FILE, os.path.join("..", TICKERS_FILE)]:
        if os.path.exists(path):
            with open(path) as f:
                tickers = [l.strip() for l in f if l.strip()]
            print(f"Loaded {len(tickers)} tickers from {path}")
            return tickers[:max_tickers]
    print("WARNING: tickers.txt not found — using fallback list")
    return ["AAPL", "NVDA", "MSFT", "TSLA", "AMZN", "META", "GOOGL", "AMD"]


def get_regime_today(events_file=EVENTS_FILE):
    """Return today's market regime from events CSV."""
    today = date.today()
    for path in [events_file, os.path.join("..", events_file)]:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, parse_dates=["date_start", "date_end"])
            active = df[
                (df["date_start"].dt.date <= today) &
                (df["date_end"].dt.date >= today)
            ]
            if not active.empty:
                row = active.iloc[-1]
                return str(row.get("regime", "Unknown")), str(row.get("event_name", "None"))
        except Exception:
            pass
    return "Unknown", "None"


def get_rsi_macd_local(ticker):
    """
    Read RSI and MACD from D drive technicals.
    Returns (rsi, macd_signal, source) where source is 'd_drive' or 'unavailable'.
    Only works on home PC — GitHub Actions will return None, None, 'unavailable'.
    """
    safe_t = ticker.replace(".", "-")

    rsi = None
    macd = None

    # RSI
    rsi_path = os.path.join(D_TECH_DIR, f"{safe_t}_rsi.csv")
    if os.path.exists(rsi_path):
        try:
            df = pd.read_csv(rsi_path, index_col=0, parse_dates=True)
            rsi_cols = [c for c in df.columns if "rsi" in c.lower()]
            if rsi_cols:
                rsi = round(float(df[rsi_cols[0]].dropna().iloc[-1]), 2)
        except Exception:
            pass

    # MACD
    macd_path = os.path.join(D_TECH_DIR, f"{safe_t}_macd.csv")
    if os.path.exists(macd_path):
        try:
            df = pd.read_csv(macd_path, index_col=0, parse_dates=True)
            macd_cols = [c for c in df.columns if "macd" in c.lower()
                         and "signal" not in c.lower()]
            sig_cols  = [c for c in df.columns if "signal" in c.lower()]
            if macd_cols and sig_cols:
                last_macd = float(df[macd_cols[0]].dropna().iloc[-1])
                last_sig  = float(df[sig_cols[0]].dropna().iloc[-1])
                macd = "bullish" if last_macd > last_sig else "bearish"
        except Exception:
            pass

    source = "d_drive" if (rsi is not None or macd is not None) else "unavailable"
    return rsi, macd, source


def fetch_premarket_batch(tickers):
    """
    Fetch pre-market data for a batch of tickers via yfinance.
    Returns a dict: {ticker: {gap_pct, pm_price, prev_close, pm_volume}}
    Uses period='2d' with prepost=True to get pre-market candles.
    Small batches to avoid rate limits.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed — run: pip install yfinance")
        return {}

    results = {}
    batch_size = 50  # safe batch size for yfinance

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            raw = yf.download(
                batch,
                period="2d",
                interval="1m",
                prepost=True,
                group_by="ticker",
                progress=False,
                timeout=30,
            )

            now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
            pm_cutoff = now_utc.replace(hour=13, minute=30, second=0)  # 13:30 UTC = 14:30 BST

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        df = raw
                    else:
                        df = raw[ticker] if ticker in raw.columns.get_level_values(0) else pd.DataFrame()

                    if df is None or df.empty:
                        continue

                    # Previous regular session close
                    regular = df[df.index < now_utc.replace(hour=0, minute=0)]
                    if regular.empty:
                        continue
                    prev_close = float(regular["Close"].dropna().iloc[-1])

                    # Pre-market candles for today
                    today_start = now_utc.replace(hour=0, minute=0, second=0)
                    pm = df[(df.index >= today_start) & (df.index < pm_cutoff)]
                    if pm.empty or pm["Volume"].sum() < MIN_VOLUME:
                        continue

                    pm_price  = float(pm["Close"].dropna().iloc[-1])
                    pm_volume = int(pm["Volume"].sum())
                    gap_pct   = (pm_price - prev_close) / prev_close * 100

                    results[ticker] = {
                        "pm_price":   round(pm_price, 4),
                        "prev_close": round(prev_close, 4),
                        "gap_pct":    round(gap_pct, 2),
                        "pm_volume":  pm_volume,
                    }
                except Exception:
                    continue

        except Exception as e:
            print(f"  Batch {i//batch_size + 1} error: {e}")

    return results


def run():
    now_bst = datetime.now(BST)
    today = date.today()

    print("=" * 60)
    print("GarAI — Pre-market Data Logger")
    print(f"Run time: {now_bst.strftime('%A %d %b %Y %H:%M BST')}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tickers = load_tickers()
    regime, event = get_regime_today()
    print(f"Market regime today: {regime} | Event: {event}")

    print(f"\nFetching pre-market data for {len(tickers)} tickers...")
    pm_data = fetch_premarket_batch(tickers)
    print(f"  {len(pm_data)} tickers with pre-market activity above {MIN_VOLUME:,} vol")

    rows = []
    for ticker, pm in pm_data.items():
        rsi, macd, tech_source = get_rsi_macd_local(ticker)
        rows.append({
            "date":           today.isoformat(),
            "scan_time":      now_bst.strftime("%H:%M BST"),
            "ticker":         ticker,
            "pm_price":       pm["pm_price"],
            "prev_close":     pm["prev_close"],
            "gap_pct":        pm["gap_pct"],
            "pm_volume":      pm["pm_volume"],
            "notable_gap":    abs(pm["gap_pct"]) >= GAP_THRESHOLD,
            "rsi":            rsi,
            "macd_signal":    macd,
            "tech_source":    tech_source,
            "market_regime":  regime,
            "event_name":     event,
        })

    if not rows:
        print("No pre-market activity found above threshold.")
        # Write empty file so downstream scripts don't crash
        pd.DataFrame(columns=[
            "date", "scan_time", "ticker", "pm_price", "prev_close",
            "gap_pct", "pm_volume", "notable_gap", "rsi", "macd_signal",
            "tech_source", "market_regime", "event_name"
        ]).to_csv(OUTPUT_FILE, index=False)
        return

    df = pd.DataFrame(rows).sort_values("gap_pct", ascending=False)

    # Save to output/ for Streamlit
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_FILE}")

    # Save to D drive archive (home PC only)
    if os.path.exists(os.path.dirname(D_PREMARKET)) or os.path.exists("D:\\"):
        os.makedirs(D_PREMARKET, exist_ok=True)
        archive_path = os.path.join(D_PREMARKET, f"premarket_{today.isoformat()}.csv")
        df.to_csv(archive_path, index=False)
        print(f"Archived to {archive_path}")

    # Print top 10 notable gaps
    notable = df[df["notable_gap"] == True].head(10)
    if not notable.empty:
        print(f"\nTop pre-market movers (gap >= {GAP_THRESHOLD}%):")
        for _, row in notable.iterrows():
            macd_str = f" | MACD {row['macd_signal']}" if row['macd_signal'] else ""
            rsi_str = f" | RSI {row['rsi']:.0f}" if row['rsi'] else ""
            print(f"  {row['ticker']:8s}  {row['gap_pct']:+6.2f}%  "
                  f"vol {row['pm_volume']:>8,}{rsi_str}{macd_str}")

    print(f"\nComplete. {len(df)} tickers logged.")


if __name__ == "__main__":
    run()
