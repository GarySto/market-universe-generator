"""
refresh_tickers_v2.py
======================
Uses T212 API as the primary filter, yfinance as secondary validation.

Flow:
1. T212 API → all ISA-tradeable instruments
2. Filter: USD stocks only, NYSE/NASDAQ exchanges only
3. Filter: remove ETFs, leveraged products, ADRs, penny stocks
4. yfinance validation: confirm price $2-$75, volume > 200k
5. Write clean tickers.txt

Run weekly via GitHub Actions (already in daily.yml schedule).
Run manually when you want a fresh list:
    py refresh_tickers_v2.py

Output:
    tickers.txt          — clean list for universe.py
    tickers_stats.txt    — breakdown of what was kept/removed
"""

import requests
import os
import time
import pandas as pd
import yfinance as yf
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
T212_API_KEY = os.environ.get("T212_API_KEY", "")
T212_BASE    = "https://live.trading212.com/api/v0"

PRICE_MIN    = 2.00     # below this = too volatile/spreads too wide
PRICE_MAX    = 75.00    # above this = 10% move too rare
VOL_MIN      = 200_000  # minimum 10-day avg volume

# Exchanges to keep (NYSE and NASDAQ only)
VALID_EXCHANGES = {'NYSE', 'NASDAQ', 'NASDAQ CM', 'NASDAQ GM', 'NASDAQ GS',
                   'NYSEArca', 'NYSE ARCA', 'NYSE MKT', 'BATS'}

# Known ETF/leveraged product tickers to exclude
ETF_EXCLUDE = {
    'SPY','QQQ','IWM','GLD','SLV','TLT','HYG','VXX','UVXY','SVXY',
    'SQQQ','TQQQ','SPXU','SPXL','LABD','LABU','FNGU','FNGD','SOXL','SOXS',
    'ARKK','ARKQ','ARKG','ARKF','ARKW','ARKX',
    'XLK','XLF','XLE','XLV','XLI','XLB','XLU','XLRE','XLY','XLP',
    'VTI','VOO','VEA','VWO','VNQ','VIG','VTV','VGT','VUG',
    'EEM','EFA','AGG','LQD','IEMG','IEF','IAU','GDX','GDXJ',
}

# Mega caps to exclude (above $75 filter catches most, but explicitly exclude)
MEGA_EXCLUDE = {
    'AAPL','MSFT','AMZN','GOOGL','GOOG','META','NVDA',
    'JPM','JNJ','UNH','XOM','V','MA','PG','HD',
}


def fetch_t212_instruments():
    """
    Fetch all tradeable instruments from T212 ISA API.
    Returns list of instrument dicts.
    """
    if not T212_API_KEY:
        print("WARNING: T212_API_KEY not set — skipping T212 filter")
        return []

    url = f"{T212_BASE}/equity/metadata/instruments"
    headers = {"Authorization": T212_API_KEY}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            instruments = resp.json()
            print(f"T212 API: {len(instruments)} instruments returned")
            return instruments
        else:
            print(f"T212 API error: {resp.status_code}")
            return []
    except Exception as e:
        print(f"T212 API exception: {e}")
        return []


def filter_t212_instruments(instruments):
    """
    Apply filters to T212 instrument list.
    Returns list of clean ticker strings.
    """
    tickers = []
    stats = {
        'total': len(instruments),
        'not_usd': 0,
        'not_stock': 0,
        'etf': 0,
        'mega_cap': 0,
        'adrs': 0,
        'foreign_suffix': 0,
        'spac_warrant': 0,
        'passed': 0,
    }

    for inst in instruments:
        # USD only (US stocks)
        if inst.get('currencyCode') != 'USD':
            stats['not_usd'] += 1
            continue

        # Stocks only (not ETFs/EFTs at T212 type level)
        inst_type = inst.get('type', '')
        if inst_type not in ('STOCK', 'STOCK_DIVIDEND_PAY'):
            stats['not_stock'] += 1
            continue

        # Get ticker — T212 uses format TICKER_US_EQ
        raw = inst.get('ticker', '')
        ticker = raw.split('_')[0] if '_' in raw else raw

        # Skip known ETFs
        if ticker in ETF_EXCLUDE:
            stats['etf'] += 1
            continue

        # Skip mega caps
        if ticker in MEGA_EXCLUDE:
            stats['mega_cap'] += 1
            continue

        # Skip ADRs (Y suffix)
        if ticker.endswith('Y') and len(ticker) > 3:
            stats['adrs'] += 1
            continue

        # Skip foreign (F suffix)
        if ticker.endswith('F') and len(ticker) > 3:
            stats['foreign_suffix'] += 1
            continue

        # Skip SPACs/warrants (numeric suffix)
        if ticker and ticker[-1].isdigit():
            stats['spac_warrant'] += 1
            continue

        # Skip single char or too long
        if len(ticker) < 2 or len(ticker) > 6:
            continue

        stats['passed'] += 1
        tickers.append(ticker)

    print(f"\nT212 filter results:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return sorted(set(tickers))


def validate_with_yfinance(tickers, batch_size=50):
    """
    Validate tickers with yfinance — check price $2-$75, volume > 200k.
    Returns (valid_tickers, removed_tickers).
    """
    print(f"\nValidating {len(tickers)} tickers with yfinance...")
    valid = []
    removed = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(
                batch,
                period="15d",
                progress=False,
                auto_adjust=True,
            )
            if data.empty:
                for t in batch:
                    removed.append((t, None, None, "no data"))
                continue

            closes  = data["Close"].iloc[-1]
            volumes = data["Volume"].tail(10).mean()

            for t in batch:
                try:
                    if len(batch) == 1:
                        price = float(closes.iloc[-1]) if hasattr(closes, 'iloc') else float(closes)
                        vol   = float(volumes.iloc[-1]) if hasattr(volumes, 'iloc') else float(volumes)
                    else:
                        price = float(closes[t]) if t in closes.index else None
                        vol   = float(volumes[t]) if t in volumes.index else None

                    if price is None or pd.isna(price):
                        removed.append((t, None, None, "no data"))
                    elif price < PRICE_MIN:
                        removed.append((t, round(price, 2), None, f"below ${PRICE_MIN}"))
                    elif price > PRICE_MAX:
                        removed.append((t, round(price, 2), None, f"above ${PRICE_MAX}"))
                    elif vol and not pd.isna(vol) and vol < VOL_MIN:
                        removed.append((t, round(price, 2), int(vol), f"low volume"))
                    else:
                        valid.append(t)
                except Exception:
                    removed.append((t, None, None, "parse error"))

        except Exception as e:
            for t in batch:
                removed.append((t, None, None, f"download error"))

        # Rate limit protection
        if i > 0 and i % 5 == 0:
            time.sleep(1)

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Progress: {i+batch_size}/{len(tickers)} checked, {len(valid)} valid so far")

    return valid, removed


def run():
    print("=" * 60)
    print("GarAI — Ticker Refresh (T212 + yfinance)")
    print(f"Run: {datetime.now().strftime('%A %d %b %Y %H:%M')}")
    print("=" * 60)

    # Step 1: T212 API
    instruments = fetch_t212_instruments()

    if instruments:
        # Step 2: Filter T212 list
        t212_tickers = filter_t212_instruments(instruments)
        print(f"\nAfter T212 filter: {len(t212_tickers)} tickers")
    else:
        # Fallback: load existing tickers.txt
        print("Falling back to existing tickers.txt")
        with open("tickers.txt") as f:
            t212_tickers = [l.strip() for l in f if l.strip()]

    # Step 3: yfinance validation
    valid, removed = validate_with_yfinance(t212_tickers)

    # Step 4: Write outputs
    with open("tickers.txt", "w") as f:
        for t in sorted(valid):
            f.write(t + "\n")

    with open("tickers_stats.txt", "w") as f:
        f.write(f"Ticker refresh: {datetime.now().strftime('%A %d %b %Y %H:%M')}\n")
        f.write(f"T212 instruments: {len(instruments)}\n")
        f.write(f"After T212 filter: {len(t212_tickers)}\n")
        f.write(f"After yfinance validation: {len(valid)}\n")
        f.write(f"Removed: {len(removed)}\n\n")
        f.write("REMOVED TICKERS:\n")
        for t, price, vol, reason in sorted(removed, key=lambda x: x[3]):
            f.write(f"  {t}: {reason} (price=${price}, vol={vol})\n")

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"  Final ticker count: {len(valid)}")
    print(f"  Written to: tickers.txt")
    print(f"  Stats: tickers_stats.txt")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()
