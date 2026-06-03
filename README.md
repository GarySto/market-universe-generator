# GarAI Momentum Scanner

I am not a developer. I want to be clear about that upfront. I'm a bloke in his early 40s who got curious about momentum trading, decided the best way to learn was to build something, and somehow ended up with a GitHub repository. My daughters remain unimpressed.

This project is a daily stock scanner that identifies momentum candidates on the US markets before they open. The goal is simple: find stocks that are moving with conviction in premarket, get in, take a 10% profit in the first 15 minutes of regular trading, and get out. Do that 98 times starting from £50 and the maths says I'll be a millionaire. The maths is correct. The execution is the hard bit.

---

## What it actually does

Every weekday, a GitHub Action runs four times between 08:00–13:00 UTC and rebuilds a ranked universe of stocks from a watchlist in `tickers.txt`. It pulls recent price and volume data via yfinance, computes a composite momentum score for each ticker, and writes the result to `output/universe.csv`. That CSV feeds a Streamlit dashboard which I can open around 13:15–13:30 BST to see what looks interesting before the NYSE and NASDAQ open at 14:30 BST.

No manual intervention needed day to day. Open browser, look at top of list, run the pre-trade check, decide whether to trade. That's the intent.

---

## How the scoring works

Each ticker gets scored across six signals, all min-max normalised across the universe before combining so that weights reflect true relative importance:

**Premarket momentum** — a composite of normalised gap % and normalised premarket RVOL, capturing stocks where both price and volume are building simultaneously before the open. Highest-weighted signal at 5×.

**Gap %** — how much the premarket price has moved versus yesterday's close. A stock gapping 5%+ with volume is the core signal. Weight: 3×.

**Breakout score** — where yesterday's close sits within the stock's 10-day high/low range. 0 = bottom, 1 = top. Values above 0.7 suggest the stock is pushing toward a breakout. Weight: 2×.

**RVOL** — yesterday's volume divided by the 10-day average. High RVOL means elevated interest. Weight: 1×.

**Trend (5d)** — how many of the last 5 trading days closed higher than the previous close. Confirmation signal. Weight: 1×.

**Volatility score** — ATR-based. Ensures stocks with no intraday range aren't prioritised. Weight: 0.5×.

The final formula:

```
score = 5 × premarket_momentum
      + 3 × norm_gap
      + 2 × norm_breakout
      + 1 × norm_rvol
      + 1 × norm_trend
      + 0.5 × norm_volatility
```

Maximum possible score: ~12.5. The dashboard flags tickers above 7 with a real premarket gap as Trade Today candidates.

**Long-only rule:** any ticker with a negative premarket gap receives a score of 0 and is excluded from Trade Today entirely.

**Price filter:** tickers below $1.50 or above $75 are excluded. Below $1.50 the spreads in premarket are too wide. Above $75 a 10% move in 15 minutes is rare without a major catalyst.

---

## Live trading record

As of 3 June 2026, three game trades and two side trades have been completed.

| Date | Ticker | Return | Score | Pre-trade | Lesson |
|------|--------|--------|-------|-----------|--------|
| 01 Jun | SPCE | +1.2% | 12.75 | 🟢 Green | Target touched but limit order didn't fill OTC — sell manually |
| 02 Jun | NAMM | +4.95% | 14.37 | 🟢 Green | Overrode system on hunch — partial rather than win |
| 03 Jun | BB | -8.49% | 10.09 | 🔴 Red | Early entry (12:44 vs 13:30 window). RED pre-trade ignored |
| 03 Jun | MRVL | -8.97% | 11.78 | 🔴 Red | Same pattern. RSI 34 at entry — oversold |
| 03 Jun | GME | TBC | 7.39 | 🔴 Red | Entered at 12:52 — before window. Live score 0.00 |

**The pattern so far:** every loss shares the same root cause — entry before 13:30 BST. The scanner correctly identified the stocks. The pre-trade check correctly flagged RED. The human entered early. RSI below 40 at entry time has correlated with every loss.

---

## Repository structure

```
market-universe-generator/
├── .github/
│   └── workflows/
│       └── daily.yml           # GitHub Actions — runs Mon–Fri at 08:00, 11:00, 12:00, 13:00 UTC
├── output/
│   └── universe.csv            # Generated daily — don't edit manually
├── universe.py                 # The scoring engine
├── streamlit_app.py            # The dashboard
├── send_alert.py               # Email alert script
├── tickers.txt                 # Watchlist — one ticker per line (~340 tickers)
├── trades.csv                  # Live trade log — feeds the My Trades tab
└── requirements.txt            # Python dependencies
```

---

## Running it yourself

**The dashboard (public):**
The live dashboard is hosted on Streamlit Cloud and reads the latest `output/universe.csv` directly from this repo.

**Locally:**
```bash
git clone https://github.com/GarySto/market-universe-generator.git
cd market-universe-generator
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

**Running the universe builder manually:**
```bash
python universe.py
```

---

## Reading the dashboard

The dashboard has four tabs.

**Scanner** — the full ranked universe from today's scan. Gap % vs RVOL scatter shows where the interesting stocks are (top-right = gapping up with high volume). A time-sensitive banner tells you exactly what to do at each hour of the day: when to just look, when to check, when to act, when the window has closed.

**Trade Today** — filtered to tickers scoring above 7 with a real positive premarket gap. Run the pre-trade confirmation check here at 14:00–14:15 BST:

- Enter the RSI from your broker's chart for each candidate (want 50–70; below 40 = avoid)
- Press Refresh to fetch live premarket data
- Traffic lights show 🟢 (gap and volume holding), 🟡 (one signal fading), or 🔴 (momentum gone)
- Only trade 🟢. No exceptions.

**Backtest** — re-scores the last 14 trading days using only data that would have been available at the time. Pick a day and a ticker to see the 1-minute candle chart from 12:00–15:00 BST with the simulated trade: entry at 13:30, +10% target line, market open at 14:30, hard exit at 14:45.

**My Trades** — tracks every real trade against the morning scanner scores. The scatter of score vs actual return is the key chart — over time it will validate or challenge the scoring weights.

---

## Daily routine

| Time (BST) | Action |
|------------|--------|
| 13:00 | Dashboard updates. Review top candidates. Check gap_pct and premarket_rvol are populated. |
| 13:15–13:30 | Compare candidates. Check RSI on T212 charts (want 50–70). Look for gap >5%, premarket_rvol building, trend 4-5, breakout >0.7. |
| 14:00–14:15 | Run pre-trade check. Enter RSI. Click Refresh. 🟢 = proceed. 🟡 = caution. 🔴 = skip. |
| 13:30–14:15 | Entry window. Buy in premarket via Trading 212 OTC. NOT before 13:30. NOT after 14:15. |
| At +10% | Sell manually — do not rely on OTC limit orders. |
| 14:45 | Hard exit. Close all positions regardless of P&L. No exceptions. |

---

## What's next

**Phase 4 — continuation strategy:** some stocks keep running past 14:45. The plan is a GitHub Action polling every 15-30 minutes during market hours and sending an alert if a morning pick is still showing strength. This is a nudge signal, not a real-time execution signal — GitHub Actions scheduling lag prevents that.

**Phase 5 — automated execution:** parked until Phase 4 is built and the strategy has a meaningful trade history. Alpaca Markets paper trading API is the planned starting point.

**Near-term:** accumulate 20-30 trades with full signal data (RSI, MACD, EMA) to validate whether the scoring weights reflect what actually predicts a winning trade.

---

## Tech stack

Python 3.10, yfinance, pandas, numpy, Streamlit, Altair, GitHub Actions, Outlook SMTP. Nothing exotic. Automation runs on GitHub's free tier, dashboard on Streamlit Cloud's free tier. Total infrastructure cost: £0. Total time invested: more than I'd like to admit.

---

## A note on the obvious disclaimer

This is not financial advice. This is a personal project by someone learning to code and learning to trade at the same time, which is probably not the most efficient way to do either. Please don't make financial decisions based on this.
