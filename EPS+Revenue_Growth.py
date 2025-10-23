#!/usr/bin/env python3
"""
Fetch Diluted EPS and Revenue quarter-over-quarter (or year-over-year) percentage growth
for the last 5 quarters using Yahoo Finance via the yfinance library.

This version prefers Diluted EPS (column names containing 'dilut') instead of
reported/actual EPS. If no diluted EPS column is found in the earnings table,
falls back to any EPS-like column only as a last resort.

Dependencies:
    pip install yfinance pandas python-dateutil
"""

import sys
import math
import warnings
from typing import Optional, Tuple

import pandas as pd
from dateutil.relativedelta import relativedelta
import yfinance as yf

# -------- Configuration --------
# Choose "qoq" for quarter-over-quarter or "yoy" for year-over-year comparisons.
GROWTH_MODE = "qoq"
MAX_OUTPUT_PERIODS = 5  # how many most-recent growth points to show
# --------------------------------

def _pct_change(current: float, previous: float) -> Optional[float]:
    """Safe percent change: (current/previous - 1)*100. Returns None when invalid."""
    try:
        if previous is None or current is None:
            return None
        if previous == 0:
            return None
        return (current / previous - 1.0) * 100.0
    except Exception:
        return None

def _prep_growth(series: pd.Series, mode: str = "qoq") -> pd.DataFrame:
    """
    Compute percentage growth for a time series indexed by datetime (ascending).
    mode: "qoq" -> prior period; "yoy" -> same quarter one year earlier (approx by shifting 4).
    Returns a DataFrame with columns: value, compare_to, pct_growth.
    """
    s = series.dropna().sort_index()
    if s.empty:
        return pd.DataFrame(columns=["value", "compare_to", "pct_growth"])

    df = pd.DataFrame({"value": s})
    if mode.lower() == "yoy":
        df["compare_to"] = df["value"].shift(4)
    else:
        df["compare_to"] = df["value"].shift(1)

    df["pct_growth"] = [
        _pct_change(cur, prv) for cur, prv in zip(df["value"], df["compare_to"])
    ]
    return df.dropna(subset=["pct_growth"])

def _extract_eps_series(tkr: yf.Ticker) -> pd.Series:
    """
    Try to obtain a quarterly Diluted or Basic EPS series from Yahoo Finance.
    Returns a Series indexed by quarter end (datetime), values are EPS (float).
    Priority: Diluted EPS -> Basic EPS -> fallback to EPS Actual.
    """
    # Try quarterly financials first
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qfin = tkr.quarterly_financials

    if qfin is not None and not qfin.empty:
        eps_row = None
        for candidate in ["Diluted EPS", "DilutedEPS", "EPS Diluted", "epsDiluted"]:
            if candidate in qfin.index:
                eps_row = candidate
                break

        if eps_row is None:
            for candidate in ["Basic EPS", "BasicEPS", "EPS Basic", "epsBasic"]:
                if candidate in qfin.index:
                    eps_row = candidate
                    break

        if eps_row:
            eps = pd.to_numeric(qfin.loc[eps_row], errors="coerce")
            eps.index = pd.to_datetime(eps.index, errors="coerce")
            eps = eps.dropna().sort_index()
            if not eps.empty:
                return eps

    # Fallback: if Diluted/Basic EPS not found, use legacy "EPS Actual"
    eps_df = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            eps_df = tkr.get_earnings_dates(limit=40)
        except Exception:
            pass

    if eps_df is None or eps_df.empty:
        try:
            eps_df = tkr.earnings_dates
        except Exception:
            eps_df = None

    if eps_df is None or eps_df.empty:
        return pd.Series(dtype=float)

    # Look for any EPS column if fallback is needed
    col_candidates = [c for c in eps_df.columns if "eps" in c.lower()]
    if not col_candidates:
        return pd.Series(dtype=float)

    eps_col = col_candidates[0]
    eps_df.index = pd.to_datetime(eps_df.index, errors="coerce")
    eps_df = eps_df[eps_df.index.notna()].sort_index()
    eps_series = pd.to_numeric(eps_df[eps_col], errors="coerce").dropna()

    # Group by quarter (last EPS per quarter)
    if not eps_series.empty:
        qidx = eps_series.index.to_period("Q")
        eps_series = eps_series.groupby(qidx).last()
        eps_series.index = eps_series.index.to_timestamp(how="end")

    return eps_series

def _extract_revenue_series(tkr: yf.Ticker) -> pd.Series:
    """
    Get quarterly revenue series from ticker.quarterly_financials.
    Returns a Series indexed by quarter end (datetime), values in currency units (float).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qfin = tkr.quarterly_financials

    if qfin is None or qfin.empty:
        return pd.Series(dtype=float)

    row = None
    for candidate in ["Total Revenue", "TotalRevenue", "totalRevenue", "Revenue", "Total revenue"]:
        if candidate in qfin.index:
            row = candidate
            break

    if row is None:
        rev_like = [idx for idx in qfin.index if "revenue" in str(idx).lower()]
        if rev_like:
            row = rev_like[0]

    if row is None:
        return pd.Series(dtype=float)

    rev = pd.to_numeric(qfin.loc[row], errors="coerce")
    rev.index = pd.to_datetime(rev.index, errors="coerce")
    rev = rev.dropna()
    rev = rev.sort_index()
    return rev

def fetch_growth_for_ticker(ticker: str, mode: str = GROWTH_MODE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two small tables (DataFrames):
      eps_growth_df: index=quarter end (timestamp), columns=[value, compare_to, pct_growth]
      rev_growth_df: index=quarter end (timestamp), columns=[value, compare_to, pct_growth]
    Each trimmed to the most-recent MAX_OUTPUT_PERIODS rows.
    """
    tk = yf.Ticker(ticker)

    # Diluted EPS
    eps_series = _extract_eps_series(tk)
    eps_growth_full = _prep_growth(eps_series, mode=mode)
    eps_growth = eps_growth_full.tail(MAX_OUTPUT_PERIODS)

    # Revenue
    rev_series = _extract_revenue_series(tk)
    rev_growth_full = _prep_growth(rev_series, mode=mode)
    rev_growth = rev_growth_full.tail(MAX_OUTPUT_PERIODS)

    # Round for display
    for df in (eps_growth, rev_growth):
        if not df.empty:
            df["value"] = df["value"].astype(float)
            df["compare_to"] = df["compare_to"].astype(float)
            df["pct_growth"] = df["pct_growth"].astype(float).round(2)

    return eps_growth, rev_growth

def format_output(ticker: str, eps_growth: pd.DataFrame, rev_growth: pd.DataFrame, mode: str) -> str:
    header = f"\nTicker: {ticker.upper()}  |  Growth mode: {mode.upper()}  |  Showing last {MAX_OUTPUT_PERIODS} quarters\n"
    lines = [header, "-" * len(header)]

    # Diluted EPS
    lines.append("\nDiluted EPS % Growth:")
    if eps_growth.empty:
        lines.append("  (No Diluted EPS data available.)")
    else:
        lines.append("  Quarter End        Diluted EPS   vs Prior     % Growth")
        for idx, row in eps_growth.iterrows():
            lines.append(
                f"  {idx.date()}   {row['value']:<10.4g} {row['compare_to']:<10.4g} {row['pct_growth']:>8.2f}%"
            )

    # Revenue
    lines.append("\nRevenue % Growth:")
    if rev_growth.empty:
        lines.append("  (No Revenue data available.)")
    else:
        lines.append("  Quarter End        Revenue         vs Prior         % Growth")
        for idx, row in rev_growth.iterrows():
            val = row["value"]
            prv = row["compare_to"]
            def fmt_money(x):
                if abs(x) >= 1e9:
                    return f"{x/1e9:.3g}B"
                if abs(x) >= 1e6:
                    return f"{x/1e6:.3g}M"
                return f"{x:.3g}"

            lines.append(
                f"  {idx.date()}   {fmt_money(val):<15} {fmt_money(prv):<15} {row['pct_growth']:>8.2f}%"
            )

    return "\n".join(lines)

def main():
    print("Yahoo Finance Diluted EPS & Revenue Growth (last 5 quarters)")
    print("Mode:", GROWTH_MODE.upper(), "(change GROWTH_MODE in the script to 'yoy' for year-over-year)\n")
    try:
        while True:
            ticker = input("Enter a ticker (or press Enter to quit): ").strip()
            if not ticker:
                break
            try:
                eps_g, rev_g = fetch_growth_for_ticker(ticker, mode=GROWTH_MODE)
                print(format_output(ticker, eps_g, rev_g, GROWTH_MODE))
                print()
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}\n")
    except KeyboardInterrupt:
        print("\nBye!")

if __name__ == "__main__":
    main()