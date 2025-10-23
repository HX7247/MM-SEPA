#!/usr/bin/env python3
"""
Fetch EPS (Diluted/Basic preferred) and Revenue quarter-over-quarter (or year-over-year)
percentage growth for the last 5 quarters using Yahoo Finance via the yfinance library.

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
GROWTH_MODE = "qoq"  # or "yoy"
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

# --- NEW: helper to find Diluted or Basic EPS rows robustly ---
def _find_eps_row(idx) -> Optional[str]:
    """
    Given a Pandas Index of row labels, return the best label for EPS.
    Preference: any label containing 'eps' and 'dilut' -> Diluted EPS;
                otherwise any label containing 'eps' and 'basic' -> Basic EPS.
    """
    lowers = {str(i): str(i).lower() for i in idx}
    diluted = [k for k, v in lowers.items() if "eps" in v and "dilut" in v]
    if diluted:
        return diluted[0]
    basic = [k for k, v in lowers.items() if "eps" in v and "basic" in v]
    if basic:
        return basic[0]
    return None

def _extract_eps_series(tkr: yf.Ticker) -> pd.Series:
    """
    Pull quarterly Diluted (preferred) or Basic EPS from Yahoo Finance via yfinance.
    Tries quarterly_income_stmt first (newer), then quarterly_financials (older).
    Falls back to earnings 'EPS Actual/Reported' only if neither is available.
    Returns a Series indexed by quarter end timestamps.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Some yfinance versions expose quarterly income statement as 'quarterly_income_stmt'
        qis = getattr(tkr, "quarterly_income_stmt", None)
        qfin = getattr(tkr, "quarterly_financials", None)

    # 1) Try quarterly_income_stmt
    if qis is not None and not qis.empty:
        row = _find_eps_row(qis.index)
        if row is not None:
            s = pd.to_numeric(qis.loc[row], errors="coerce")
            s.index = pd.to_datetime(s.index, errors="coerce")
            s = s.dropna().sort_index()
            if not s.empty:
                return s

    # 2) Fall back to quarterly_financials
    if qfin is not None and not qfin.empty:
        row = _find_eps_row(qfin.index)
        if row is not None:
            s = pd.to_numeric(qfin.loc[row], errors="coerce")
            s.index = pd.to_datetime(s.index, errors="coerce")
            s = s.dropna().sort_index()
            if not s.empty:
                return s

    # 3) Last-resort fallback: earnings calendar EPS (Actual/Reported)
    eps_df = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            eps_df = tkr.get_earnings_dates(limit=40)  # recent rows first
        except Exception:
            pass

    if eps_df is None or eps_df.empty:
        try:
            eps_df = tkr.earnings_dates
        except Exception:
            eps_df = None

    if eps_df is None or eps_df.empty:
        return pd.Series(dtype=float)

    # Use any EPS-like column available (e.g., 'EPS Actual', 'Reported EPS')
    col_candidates = [c for c in eps_df.columns if "eps" in c.lower()]
    if not col_candidates:
        return pd.Series(dtype=float)

    eps_col = col_candidates[0]

    # Ensure datetime index
    if "Earnings Date" in eps_df.columns:
        eps_df = eps_df.set_index("Earnings Date")
    elif eps_df.index.name and "date" in str(eps_df.index.name).lower():
        pass
    else:
        try:
            first_col = eps_df.columns[0]
            maybe_dt = pd.to_datetime(eps_df[first_col], errors="coerce")
            if maybe_dt.notna().any():
                eps_df = eps_df.set_index(maybe_dt)
            else:
                eps_df.index = pd.to_datetime(eps_df.index, errors="coerce")
        except Exception:
            eps_df.index = pd.to_datetime(eps_df.index, errors="coerce")

    eps_df.index = pd.to_datetime(eps_df.index, errors="coerce")
    eps_df = eps_df[eps_df.index.notna()].sort_index()

    eps_series = pd.to_numeric(eps_df[eps_col], errors="coerce").dropna()
    if eps_series.empty:
        return pd.Series(dtype=float)

    # One value per quarter (take the last in each quarter)
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

    # Yahoo often uses 'Total Revenue'; fall back to close matches.
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
    rev = rev.dropna().sort_index()
    return rev

def fetch_growth_for_ticker(ticker: str, mode: str = GROWTH_MODE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two small tables (DataFrames):
      eps_growth_df: index=quarter end (timestamp), columns=[value, compare_to, pct_growth]
      rev_growth_df: index=quarter end (timestamp), columns=[value, compare_to, pct_growth]
    Each trimmed to the most-recent MAX_OUTPUT_PERIODS rows.
    """
    tk = yf.Ticker(ticker)

    # EPS (Diluted or Basic preferred)
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

    # EPS
    lines.append("\nEPS % Growth (Diluted preferred, else Basic):")
    if eps_growth.empty:
        lines.append("  (No EPS growth available — need at least 2 quarters of EPS.)")
    else:
        lines.append("  Quarter End        EPS      vs Prior     % Growth")
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
    print("Yahoo Finance EPS (Diluted/Basic) & Revenue Growth (last 5 quarters)")
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
