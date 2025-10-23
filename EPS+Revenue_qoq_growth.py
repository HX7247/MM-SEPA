#!/usr/bin/env python3
"""
EPS+Revenue_qoq_growth.py

Compute EPS percentage difference (growth) for the last 5 quarters
from Yahoo Finance via yfinance, with robust fallbacks.

Usage:
    python EPS+Revenue_qoq_growth.py
    (then enter a ticker, e.g., AAPL)
"""

import warnings
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf

# ---------- Config ----------
MAX_OUTPUT_PERIODS = 5        # how many most recent % changes to show
GROWTH_MODE = "qoq"           # "qoq" (quarter-over-quarter) or "yoy" (same quarter last year)
# ----------------------------

def pct_change(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    """Standard % change. Returns None if previous is 0 or missing."""
    try:
        if current is None or previous is None:
            return None
        if previous == 0:
            return None
        return (current / previous - 1.0) * 100.0
    except Exception:
        return None

def extract_eps_series(tkr: yf.Ticker) -> pd.Series:
    """
    Try multiple sources to get a quarterly EPS series:
      1) get_earnings_dates(limit=...) -> 'EPS Actual'
      2) quarterly_earnings -> 'Earnings'
    Returns a Series indexed by quarter end (Timestamp), values=float EPS.
    """
    # ---- Primary: earnings dates feed (EPS Actual) ----
    eps_df = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            eps_df = tkr.get_earnings_dates(limit=40)  # recent quarters
        except Exception:
            eps_df = None

    if eps_df is not None and not eps_df.empty:
        # Find an EPS "actual/reported" column robustly
        candidates = [c for c in eps_df.columns
                      if "eps" in c.lower() and ("actual" in c.lower() or "reported" in c.lower())]
        if candidates:
            eps_col = candidates[0]
            # Ensure datetime index
            try:
                if "Earnings Date" in eps_df.columns:
                    eps_df = eps_df.set_index("Earnings Date")
            except Exception:
                pass
            eps_df.index = pd.to_datetime(eps_df.index, errors="coerce")
            eps_df = eps_df[eps_df.index.notna()]
            ser = pd.to_numeric(eps_df[eps_col], errors="coerce").dropna()
            if not ser.empty:
                # Group by calendar quarter to deduplicate and take last
                qidx = ser.index.to_period("Q")
                ser = ser.groupby(qidx).last()
                ser.index = ser.index.to_timestamp(how="end")
                return ser.sort_index()

    # ---- Fallback: quarterly_earnings (often present) ----
    qearn = getattr(tkr, "quarterly_earnings", None)
    if qearn is not None and not qearn.empty:
        qearn = qearn.sort_index()
        # Some yfinance versions use column 'Earnings' for EPS
        eps_col = None
        for c in ["Earnings", "EPS", "Reported EPS"]:
            if c in qearn.columns:
                eps_col = c
                break
        if eps_col is None:
            # Try a loose match
            matches = [c for c in qearn.columns if "earn" in c.lower() or "eps" in c.lower()]
            if matches:
                eps_col = matches[0]

        if eps_col:
            ser = pd.to_numeric(qearn[eps_col], errors="coerce")
            # Index is a quarter label (e.g., '2024-03-31'); coerce to datetime
            try:
                ser.index = pd.to_datetime(qearn.index)
            except Exception:
                ser.index = pd.to_datetime(ser.index, errors="coerce")
            ser = ser.dropna()
            if not ser.empty:
                ser.index = ser.index.to_period("Q").to_timestamp(how="end")
                return ser.sort_index()

    # Nothing found
    return pd.Series(dtype=float)

def compute_eps_growth(eps_series: pd.Series, mode: str = "qoq") -> pd.DataFrame:
    """
    Given an EPS series indexed by quarter end ascending, compute % difference.
    mode: "qoq" (shift 1) or "yoy" (shift 4).
    Returns DataFrame: columns [EPS, CompareTo, PctDiff], last MAX_OUTPUT_PERIODS rows.
    """
    s = eps_series.dropna().sort_index()
    if s.empty:
        return pd.DataFrame(columns=["EPS", "CompareTo", "PctDiff"])

    df = pd.DataFrame({"EPS": s})
    if mode.lower() == "yoy":
        df["CompareTo"] = df["EPS"].shift(4)
    else:
        df["CompareTo"] = df["EPS"].shift(1)

    df["PctDiff"] = [pct_change(cur, prv) for cur, prv in zip(df["EPS"], df["CompareTo"])]
    df = df.dropna(subset=["PctDiff"])
    # Round for display
    df["PctDiff"] = df["PctDiff"].astype(float).round(2)
    return df.tail(MAX_OUTPUT_PERIODS)

def format_table(ticker: str, mode: str, eps_growth: pd.DataFrame) -> str:
    title = f"\nTicker: {ticker.upper()} | EPS % Difference ({mode.upper()}) | Last {MAX_OUTPUT_PERIODS} quarters"
    lines = [title, "-" * len(title)]
    if eps_growth.empty:
        lines.append("No EPS data available for this ticker.")
        return "\n".join(lines)

    lines.append("\nQuarter End        EPS        Compare-To     % Diff")
    for idx, row in eps_growth.iterrows():
        lines.append(
            f"{idx.date()}    {row['EPS']:<10.4g} {row['CompareTo']:<12.4g} {row['PctDiff']:>8.2f}%"
        )
    return "\n".join(lines)

def main():
    print("Yahoo Finance — Quarterly EPS % Difference")
    print(f"Mode: {GROWTH_MODE.upper()}  (change GROWTH_MODE in the file to 'yoy' for year-over-year)\n")
    while True:
        try:
            ticker = input("Enter a ticker (or press Enter to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not ticker:
            break
        try:
            tkr = yf.Ticker(ticker)
            eps_series = extract_eps_series(tkr)
            eps_growth = compute_eps_growth(eps_series, mode=GROWTH_MODE)
            print(format_table(ticker, GROWTH_MODE, eps_growth))
            print()
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
