# Market Universe Generator

I am not a developer. I want to be clear about that upfront. I'm a bloke in his early 40s who got curious about momentum trading, decided the best way to learn was to build something, and somehow ended up with a GitHub repository. My daughters remain unimpressed.

This project is a daily stock scanner that identifies momentum candidates on the US markets before they open. The goal is simple: find stocks that are moving with conviction in premarket, get in, take a 10% profit in the first 15 minutes of regular trading, and get out. Do that 98 times starting from £50 and the maths says I'll be a millionaire. The maths is correct. The execution is the hard bit.

---

## What it actually does

Every weekday at 13:00 BST, a GitHub Action runs automatically and rebuilds a ranked universe of stocks from a watchlist I maintain in `tickers.txt`. It pulls recent price and volume data via yfinance, computes a momentum score for each ticker, and writes the result to `output/universe.csv`. That CSV feeds a Streamlit dashboard which I (and anyone I share the link with) can open around 13:15–13:30 to see what looks interesting before the NYSE and NASDAQ open at 14:30 BST.

No manual intervention needed day to day. Open browser, look at top of list, decide whether to trade. That's the intent.

---

## How the scoring works

Each ticker gets scored across five dimensions:

**Gap %** — how much the premarket price has moved versus yesterday's close. A stock gapping up 5%+ with volume behind it is the core signal I'm looking for.

**Relative volume (RVOL)** — yesterday's volume compared to its 10-day average. High RVOL means people are paying attention.

**Premarket RVOL** — same thing but for premarket volume specifically. This is the freshest signal.

**Breakout score** — where yesterday's close sits within the stock's 10-day high/low range. A score of 1 means it closed at the top of its recent range. A score of 0 means the bottom.

**Volatility score** — ATR-based. Higher volatility means more room to move, but also more risk. It's in the score because a stock that never moves isn't useful to this strategy even if everything else looks good.

There is also a sixth composite signal:

**Premarket momentum** — a combination of gap % and premarket RVOL, both normalised, that captures stocks where price and volume are building together before the open. This is the highest-weighted signal.

Before scoring, all signals are min-max normalised to a 0–1 scale across the full universe of tickers scanned that day. This matters: it means the weights actually control relative importance, rather than being dominated by whichever signal happens to have the largest raw numbers. Previously, RVOL (an unbounded ratio) was swamping gap % (a small fraction) despite the weights suggesting otherwise.

The final score:

```
score = 5 × premarket_momentum  (normalised gap + premarket RVOL)
      + 3 × norm_gap            (gap % normalised)
      + 2 × norm_breakout       (breakout score normalised)
      + 1 × norm_rvol           (RVOL normalised)
      + 1 × norm_trend          (trend_5d / 5)
      + 0.5 × norm_volatility   (volatility score normalised)
```

Maximum possible score: ~12.5. The dashboard flags tickers above 7 as worth investigating.

Long-only rules are enforced at the scoring stage: any ticker with a negative premarket gap receives a score of 0 and is excluded from Trade Today entirely. This prevents high-RVOL down-gappers from appearing as false positives.

The top of the list each morning is where I start looking. Whether I actually trade any of them is still a human decision — this tool narrows the field, it doesn't make the call for me.

---

## Repository structure

```
market-universe-generator/
├── .github/
│   └── workflows/
│       └── daily.yml        # GitHub Actions — runs Mon–Fri at 12:00 UTC (13:00 BST)
├── output/
│   └── universe.csv         # Generated daily — don't edit this manually
├── universe.py              # The scoring engine
├── streamlit_app.py         # The dashboard
├── tickers.txt              # Watchlist — one ticker per line
└── requirements.txt         # Python dependencies
```

---

## Running it yourself

**The dashboard (public):**
The live dashboard is hosted on Streamlit Cloud. It reads the latest `output/universe.csv` directly from this repo. If the GitHub Action has run that morning, the data will be fresh.

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

This generates a fresh `output/universe.csv` based on current data. Useful if you want to test outside of the scheduled run.

---

## Reading the dashboard

The dashboard has four tabs: Scanner, Trade Today, Backtest, and My Trades.

**Top 10 Momentum Tickers** — sorted by score descending. These are the first names worth looking at each morning.

**Gap % vs RVOL scatter** — top-right of this chart is where you want to be. High gap and high relative volume together is a much stronger signal than either on its own.

**Breakout score chart** — tickers near 1.0 are trading near the top of their recent range, which can mean momentum continuation or exhaustion depending on the broader context.

**Full universe table** — sortable. Useful if you want to filter or look beyond the top 10.

**Trade Today tab** — filters the morning scan to tickers scoring above 7 with a real positive premarket gap. Also includes a pre-trade confirmation check (run around 14:00–14:15 BST) that refetches live premarket data and gives a 🟢🟡🔴 traffic light for each candidate.

**Backtest tab** — re-scores each of the last 14 trading days using only data that would have been available at the time. Uses the same normalised scoring as the live scanner. Pick a day and a ticker to see the 1-minute candle chart and simulate the +10% / 14:45 BST exit strategy.

**My Trades tab** — tracks every real trade made using the scanner. Reads from `trades.csv` in the repo. Shows running bank, win rate, return distribution, and a scatter of score vs actual return.

One thing worth knowing: the `gap_pct` and `premarket_rvol` columns will show as zero outside of premarket hours. The Action runs at 13:00 BST specifically to catch live premarket data, so if you're looking at the dashboard at 9pm, those numbers won't mean much.

---

## Trading strategy

The current approach:

- Review the top candidates on the dashboard around 13:15–13:30 BST
- Look for tickers with a score above 7, gap_pct above ~0.05, premarket_rvol building, trend_5d of 4 or 5, and a strong breakout score
- Buy in premarket or at market open (14:30 BST)
- Target: +10% from entry
- Exit: take the profit, or cut at 15 minutes after open if target hasn't been hit

Starting capital: £50. Target: not having to do this 98 times perfectly.

Slippage and fees aren't yet modelled in the backtest. That's on the list.

---

## What's coming

The backtesting module is live and working — it re-scores the last 14 trading days with the same normalised formula as the live scanner, and shows 1-minute candle replays with the simulated +10% / 14:45 BST trade. Slippage and fees aren't yet modelled. That's on the list.

The My Trades tab is also live and tracks real trades against the scanner's morning scores — the scatter of score vs actual return will be the key chart for validating (or challenging) the model over time.

Next priorities: alerts (probably email), expanding the trade log to capture enough data to run proper weight optimisation, and possibly adding float and catalyst filters as separate pre-filters rather than blending them into the score.

---

## Tech stack

Python 3.10, yfinance, pandas, numpy, Streamlit, GitHub Actions. Nothing exotic. The automation runs on GitHub's free tier, the dashboard is hosted on Streamlit Cloud's free tier. Total infrastructure cost: £0. Total time invested: more than I'd like to admit.

---

## A note on the obvious disclaimer

This is not financial advice. This is a personal project by someone learning to code and learning to trade at the same time, which is probably not the most efficient way to do either. Please don't make financial decisions based on this.
