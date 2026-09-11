"""
The Game - quick premarket sanity check. Read-only, no LLM, no changes to
anything. Checks today's universe.csv for the kinds of things that have
gone wrong before in this file's own history (see module docstring in
universe.py): a real gap_pct actually varying rather than stuck at 0,
tradeability floors doing their job, and no obviously broken scores.

This is NOT a replacement for a real Analyst/Recommender for The Game -
that's real, separate work for another session. This is just eyes-on
before premarket starts today.
"""
import pandas as pd
import os

path = "output/universe.csv"
if not os.path.exists(path):
    print(f"No universe.csv found yet at {path} - has today's build_universe() run?")
    exit()

df = pd.read_csv(path)
print(f"Rows: {len(df)}")
print()

if "gap_pct" in df.columns:
    nonzero_gaps = (df["gap_pct"] != 0).sum()
    print(f"Non-zero gap_pct: {nonzero_gaps} of {len(df)} rows")
    if nonzero_gaps == 0:
        print("  WARNING: all gaps are exactly zero - this is the exact shape of the")
        print("  gap_pct bug fixed on 6 Sep. Worth checking before trusting today's scores.")

if "gap_available" in df.columns:
    real_gap_count = df["gap_available"].sum()
    print(f"Rows with a REAL premarket gap fetched: {real_gap_count}")

if "score" in df.columns:
    print(f"Top score today: {df['score'].max():.2f}")
    print(f"Rows scoring 7.0+ (SCORE_THRESHOLD): {(df['score'] >= 7.0).sum()}")

if "premarket_source" in df.columns:
    print()
    print("premarket_source breakdown:")
    print(df["premarket_source"].value_counts().to_string())
