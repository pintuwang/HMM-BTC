"""
MSTR Implied-Vol vs Realized-Vol Logger
========================================
Once per weekday (during market hours), records for the ~7-DTE weekly MSTR calls
at 10 / 15 / 20 / 25 % OTM:

  strike, bid, ask, mid, OI, volume, Yahoo's IV field,
  IV backed out from MID and from BID (bid = what a seller actually receives),
  30d and 10d realized vol (same formula as generate_hmm.py),
  and the ratio  IV / HV30  -> this IS the IV_MULT used in the covered-call backtest.

Output
  data/iv_log.csv        one row per (date, OTM level); re-runs on the same day overwrite
  data/iv_summary.json   latest reading + rolling means of the ratios per OTM level

Notes
  - Use iv_bid / hv30 as the conservative IV_MULT (you sell at the bid, not the mid).
  - Rows flagged low_price=True (mid < $0.20) have tick-size-dominated IVs; treat with care.
    Expect this mostly at 25% OTM when vol is low.
  - hv30 is TRAILING realized vol, matching the backtest. Close prices are logged too, so the
    forward-realized comparison (IV vs vol actually realized over the option's life)
    can be computed later from this file.
"""

import json
import os
import time
from datetime import datetime, timezone, timedelta
from math import log, sqrt, exp

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm

# ─────────────────────────────── CONFIG ───────────────────────────────
TICKER        = "MSTR"
OTM_LEVELS    = [0.10, 0.15, 0.20, 0.25]
TARGET_DTE    = 7            # pick the expiry closest to this many calendar days
MIN_DTE       = 3            # never pick an expiry closer than this
RISK_FREE     = 0.04
LOW_PRICE     = 0.20         # flag contracts with mid below this ($)
LOG_PATH      = "data/iv_log.csv"
SUMMARY_PATH  = "data/iv_summary.json"
RETRIES       = 3


# ─────────────────────────── BLACK-SCHOLES ────────────────────────────
def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def implied_vol(price, S, K, T, r):
    """Solve BS call price = price for sigma. NaN if no valid solution."""
    if price is None or not np.isfinite(price) or price <= 0 or T <= 0:
        return np.nan
    intrinsic = max(S - K * exp(-r * T), 0.0)
    if price <= intrinsic + 1e-6:
        return np.nan
    try:
        return brentq(lambda s: bs_call(S, K, T, r, s) - price, 1e-4, 10.0, xtol=1e-6)
    except ValueError:
        return np.nan


# ────────────────────────────── HELPERS ───────────────────────────────
def _num(x, default=0.0):
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def with_retries(fn, *args, **kwargs):
    last = None
    for i in range(RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:          # yfinance rate limits / transient errors
            last = e
            time.sleep(5 * (i + 1))
    raise last


def realized_vols(hist):
    ret = hist["Close"].pct_change()
    hv30 = float(ret.tail(30).std() * np.sqrt(252))
    hv10 = float(ret.tail(10).std() * np.sqrt(252))
    return hv30, hv10


def pick_expiry(expiries, today):
    """Expiry closest to TARGET_DTE calendar days, at least MIN_DTE away."""
    best, best_gap = None, None
    for e in expiries:
        dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
        if dte < MIN_DTE:
            continue
        gap = abs(dte - TARGET_DTE)
        if best is None or gap < best_gap:
            best, best_gap = e, gap
    return best


def years_to_expiry(expiry, now_utc):
    # Options expire at the 16:00 ET close; 20:00 UTC is close enough year-round for this purpose.
    exp_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(hour=20, tzinfo=timezone.utc)
    return max((exp_dt - now_utc).total_seconds(), 3600) / (365 * 24 * 3600)


def build_rows(spot, calls, expiry, now_utc, hv30, hv10):
    """Pure function (testable): one row per OTM level from a calls DataFrame."""
    T = years_to_expiry(expiry, now_utc)
    dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - now_utc.date()).days
    calls = calls.sort_values("strike")

    # ATM reference: strike nearest spot
    atm = calls.iloc[(calls["strike"] - spot).abs().argsort().iloc[0]]
    ab, aa = _num(atm.get("bid")), _num(atm.get("ask"))
    atm_mid = (ab + aa) / 2 if ab > 0 and aa > 0 else np.nan
    atm_iv = implied_vol(atm_mid, spot, float(atm["strike"]), T, RISK_FREE)

    rows = []
    for otm in OTM_LEVELS:
        target = spot * (1 + otm)
        above = calls[calls["strike"] >= target]
        if above.empty:
            rows.append(dict(otm_target=otm, note="no strike at/above target"))
            continue
        c = above.iloc[0]                                   # first listed strike at/above target
        K, bid, ask = float(c["strike"]), _num(c.get("bid")), _num(c.get("ask"))
        quoted = bid > 0 and ask > 0
        mid = (bid + ask) / 2 if quoted else np.nan
        iv_mid = implied_vol(mid, spot, K, T, RISK_FREE)
        iv_bid = implied_vol(bid, spot, K, T, RISK_FREE) if bid > 0 else np.nan
        rows.append(dict(
            otm_target   = otm,
            strike       = K,
            otm_actual   = round(K / spot - 1, 4),
            bid          = bid,
            ask          = ask,
            mid          = round(mid, 3) if quoted else np.nan,
            spread_pct   = round((ask - bid) / mid, 3) if quoted and mid > 0 else np.nan,
            open_interest= int(_num(c.get("openInterest"))),
            volume       = int(_num(c.get("volume"))),
            iv_yahoo     = round(_num(c.get("impliedVolatility"), np.nan), 4),
            iv_mid       = round(iv_mid, 4) if np.isfinite(iv_mid) else np.nan,
            iv_bid       = round(iv_bid, 4) if np.isfinite(iv_bid) else np.nan,
            ratio_mid    = round(iv_mid / hv30, 3) if np.isfinite(iv_mid) and hv30 > 0 else np.nan,
            ratio_bid    = round(iv_bid / hv30, 3) if np.isfinite(iv_bid) and hv30 > 0 else np.nan,
            low_price    = bool(quoted and mid < LOW_PRICE),
            note         = "" if quoted else "no two-sided quote",
        ))

    common = dict(
        date       = now_utc.strftime("%Y-%m-%d"),
        run_utc    = now_utc.strftime("%Y-%m-%d %H:%M"),
        spot       = round(spot, 2),
        expiry     = expiry,
        dte        = dte,
        hv30       = round(hv30, 4),
        hv10       = round(hv10, 4),
        atm_strike = float(atm["strike"]),
        atm_iv     = round(atm_iv, 4) if np.isfinite(atm_iv) else np.nan,
        atm_ratio  = round(atm_iv / hv30, 3) if np.isfinite(atm_iv) and hv30 > 0 else np.nan,
    )
    return [{**common, **r} for r in rows]


def update_log(rows, path):
    new = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old, new], ignore_index=True)
        df = df.drop_duplicates(subset=["date", "otm_target"], keep="last")
    else:
        df = new
    df = df.sort_values(["date", "otm_target"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def write_summary(df, path):
    out = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
           "days_logged": int(df["date"].nunique()), "levels": {}}
    latest_date = df["date"].max()
    for otm, g in df.groupby("otm_target"):
        g = g.sort_values("date")
        latest = g[g["date"] == latest_date]
        rb = g["ratio_bid"].dropna()
        out["levels"][f"{int(round(otm*100))}pct"] = {
            "latest_ratio_bid" : float(latest["ratio_bid"].iloc[0]) if len(latest) and pd.notna(latest["ratio_bid"].iloc[0]) else None,
            "latest_ratio_mid" : float(latest["ratio_mid"].iloc[0]) if len(latest) and pd.notna(latest["ratio_mid"].iloc[0]) else None,
            "mean_ratio_bid_all": round(float(rb.mean()), 3) if len(rb) else None,
            "mean_ratio_bid_20d": round(float(rb.tail(20).mean()), 3) if len(rb) else None,
            "n_valid"          : int(len(rb)),
        }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out


# ──────────────────────────────── MAIN ────────────────────────────────
def main():
    os.makedirs("data", exist_ok=True)
    now_utc = datetime.now(timezone.utc)
    tk = yf.Ticker(TICKER)

    hist = with_retries(tk.history, period="3mo", auto_adjust=True)
    if hist.empty:
        raise SystemExit("No price history returned")
    spot = float(hist["Close"].iloc[-1])      # during market hours: latest price
    hv30, hv10 = realized_vols(hist)

    expiries = with_retries(lambda: tk.options)
    expiry = pick_expiry(expiries, now_utc.date())
    if expiry is None:
        raise SystemExit("No suitable expiry found")
    calls = with_retries(tk.option_chain, expiry).calls

    rows = build_rows(spot, calls, expiry, now_utc, hv30, hv10)
    df = update_log(rows, LOG_PATH)
    summary = write_summary(df, SUMMARY_PATH)

    print(f"{TICKER} spot {spot:.2f} | expiry {expiry} | HV30 {hv30:.1%} | HV10 {hv10:.1%}")
    for r in rows:
        if "strike" not in r:
            print(f"  {r['otm_target']:.0%}: {r['note']}")
            continue
        print(f"  {r['otm_target']:.0%} OTM  K={r['strike']:<7} bid/ask {r['bid']:.2f}/{r['ask']:.2f}  "
              f"IV bid {r['iv_bid']}  mid {r['iv_mid']}  Yahoo {r['iv_yahoo']}  "
              f"ratio(bid) {r['ratio_bid']}  {'LOW PRICE' if r['low_price'] else ''} {r['note']}")
    print(f"  days logged: {summary['days_logged']}")


if __name__ == "__main__":
    main()
