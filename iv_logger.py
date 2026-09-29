"""
MSTR Implied-Vol vs Realized-Vol Logger — CALLS and PUTS
=========================================================
Once per US trading day (first run inside market hours wins), records the ~7-DTE weekly
MSTR options at 10 / 15 / 20 / 25 % out-of-the-money, on BOTH sides:

  OTM CALLS  -> strike at/ABOVE spot x (1 + otm)   -> data/iv_log_calls.csv
  OTM PUTS   -> strike at/BELOW spot x (1 - otm)   -> data/iv_log_puts.csv

Per option: strike, bid, ask, mid, spread, OI, volume, Yahoo's IV field,
IV backed out from the MID and from the BID (bid = what a seller actually receives),
30d / 10d realized vol (same formula as generate_hmm.py), and

  iv_hv_ratio_bid = IV at bid / HV30     <- the realistic IV_MULT for a SELLER

Summary of both sides: data/iv_summary.json  ({"calls": {...}, "puts": {...}})

Columns (identical in both CSVs):
  date             US trading date (New York)          session   live | pre_market | after_close
  option_type      call | put                           otm_target 0.10 / 0.15 / 0.20 / 0.25
  otm_actual       true distance of the listed strike from spot (always positive)
  atm_*            at-the-money reference for THIS option type
  low_price=True   mid < $0.20: tick size dominates the IV — treat with care

Notes
  - The covered-call break-evens (~1.09 @10%, ~1.10 @15%, ~1.00 @20%) apply to CALLS only.
    Puts have no break-even yet — that needs a cash-secured-put backtest.
  - One-time migration: an old data/iv_log.csv (calls only) is converted to
    data/iv_log_calls.csv with the new column names, then removed.
"""

import json
import os
import time
from datetime import datetime, timezone, timedelta
from math import log, sqrt, exp
from zoneinfo import ZoneInfo

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
LOG_PATHS     = {"call": "data/iv_log_calls.csv", "put": "data/iv_log_puts.csv"}
LEGACY_LOG    = "data/iv_log.csv"
SUMMARY_PATH  = "data/iv_summary.json"
RETRIES       = 3
ET            = ZoneInfo("America/New_York")
LIVE_START    = (9, 35)      # ET — skip the first 5 min (wide opening quotes)
LIVE_END      = (15, 55)     # ET — before the closing auction

LEGACY_RENAME = {"atm_ratio": "atm_iv_hv_ratio", "ratio_mid": "iv_hv_ratio_mid",
                 "ratio_bid": "iv_hv_ratio_bid"}
KEY           = ["date", "option_type", "otm_target"]


# ─────────────────────────── BLACK-SCHOLES ────────────────────────────
def bs_price(S, K, T, r, sigma, kind="call"):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if kind == "call":
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_call(S, K, T, r, sigma):          # kept for backward compatibility
    return bs_price(S, K, T, r, sigma, "call")


def implied_vol(price, S, K, T, r, kind="call"):
    """Solve BS price = price for sigma. NaN if no valid solution."""
    if price is None or not np.isfinite(price) or price <= 0 or T <= 0:
        return np.nan
    disc_k = K * exp(-r * T)
    intrinsic = max(S - disc_k, 0.0) if kind == "call" else max(disc_k - S, 0.0)
    if price <= intrinsic + 1e-6:
        return np.nan
    try:
        return brentq(lambda s: bs_price(S, K, T, r, s, kind) - price, 1e-4, 10.0, xtol=1e-6)
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


def market_session(now_utc):
    """'live' during regular US hours, 'pre_market' before them on a weekday, else 'after_close'.
    Exchange holidays aren't modelled (no quotes -> nothing is written, see main)."""
    et = now_utc.astimezone(ET)
    mins = et.hour * 60 + et.minute
    start, end = LIVE_START[0]*60 + LIVE_START[1], LIVE_END[0]*60 + LIVE_END[1]
    if et.weekday() < 5 and start <= mins <= end:
        return "live"
    if et.weekday() < 5 and mins < start:
        return "pre_market"
    return "after_close"


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


def _pick_strike(chain, spot, otm, kind):
    """First listed strike at/above spot*(1+otm) for calls, at/below spot*(1-otm) for puts."""
    if kind == "call":
        side = chain[chain["strike"] >= spot * (1 + otm)]
        return None if side.empty else side.iloc[0]
    side = chain[chain["strike"] <= spot * (1 - otm)]
    return None if side.empty else side.iloc[-1]


def build_rows(spot, chain, expiry, now_utc, hv30, hv10, kind="call"):
    """Pure function (testable): one row per OTM level for one option type."""
    T = years_to_expiry(expiry, now_utc)
    et_date = now_utc.astimezone(ET).date()
    dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - et_date).days
    chain = chain.sort_values("strike")

    atm = chain.iloc[(chain["strike"] - spot).abs().argsort().iloc[0]]
    ab, aa = _num(atm.get("bid")), _num(atm.get("ask"))
    atm_mid = (ab + aa) / 2 if ab > 0 and aa > 0 else np.nan
    atm_iv = implied_vol(atm_mid, spot, float(atm["strike"]), T, RISK_FREE, kind)

    common = dict(
        date            = et_date.strftime("%Y-%m-%d"),
        session         = market_session(now_utc),
        run_utc         = now_utc.strftime("%Y-%m-%d %H:%M"),
        spot            = round(spot, 2),
        expiry          = expiry,
        dte             = dte,
        hv30            = round(hv30, 4),
        hv10            = round(hv10, 4),
        atm_strike      = float(atm["strike"]),
        atm_iv          = round(atm_iv, 4) if np.isfinite(atm_iv) else np.nan,
        atm_iv_hv_ratio = round(atm_iv / hv30, 3) if np.isfinite(atm_iv) and hv30 > 0 else np.nan,
        option_type     = kind,
    )

    rows = []
    for otm in OTM_LEVELS:
        c = _pick_strike(chain, spot, otm, kind)
        if c is None:
            rows.append({**common, "otm_target": otm, "note": "no listed strike beyond target"})
            continue
        K, bid, ask = float(c["strike"]), _num(c.get("bid")), _num(c.get("ask"))
        quoted = bid > 0 and ask > 0
        mid = (bid + ask) / 2 if quoted else np.nan
        iv_mid = implied_vol(mid, spot, K, T, RISK_FREE, kind)
        iv_bid = implied_vol(bid, spot, K, T, RISK_FREE, kind) if bid > 0 else np.nan
        rows.append({**common,
            "otm_target"      : otm,
            "strike"          : K,
            "otm_actual"      : round(abs(K / spot - 1), 4),
            "bid"             : bid,
            "ask"             : ask,
            "mid"             : round(mid, 3) if quoted else np.nan,
            "spread_pct"      : round((ask - bid) / mid, 3) if quoted and mid > 0 else np.nan,
            "open_interest"   : int(_num(c.get("openInterest"))),
            "volume"          : int(_num(c.get("volume"))),
            "iv_yahoo"        : round(_num(c.get("impliedVolatility"), np.nan), 4),
            "iv_mid"          : round(iv_mid, 4) if np.isfinite(iv_mid) else np.nan,
            "iv_bid"          : round(iv_bid, 4) if np.isfinite(iv_bid) else np.nan,
            "iv_hv_ratio_mid" : round(iv_mid / hv30, 3) if np.isfinite(iv_mid) and hv30 > 0 else np.nan,
            "iv_hv_ratio_bid" : round(iv_bid / hv30, 3) if np.isfinite(iv_bid) and hv30 > 0 else np.nan,
            "low_price"       : bool(quoted and mid < LOW_PRICE),
            "note"            : "" if quoted else "no two-sided quote",
        })
    return rows


def migrate_legacy():
    """One-time: data/iv_log.csv (calls only, old column names) -> data/iv_log_calls.csv."""
    if not os.path.exists(LEGACY_LOG) or os.path.exists(LOG_PATHS["call"]):
        return
    old = pd.read_csv(LEGACY_LOG).rename(columns=LEGACY_RENAME)
    old["option_type"] = "call"
    if "session" not in old.columns:
        old["session"] = "after_close"      # all pre-migration runs were after the close
    old["session"] = old["session"].fillna("after_close")
    old = old[pd.to_numeric(old.get("bid"), errors="coerce").fillna(0) > 0]   # drop empty pre-market rows
    old.to_csv(LOG_PATHS["call"], index=False)
    os.remove(LEGACY_LOG)
    print(f"migrated {len(old)} call rows: {LEGACY_LOG} -> {LOG_PATHS['call']}")


def update_log(rows, path):
    new = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old, new], ignore_index=True)
        if "session" not in df.columns:
            df["session"] = np.nan
        df["_live"] = (df["session"] == "live").astype(int)
        df = df.sort_values(KEY + ["_live"], kind="stable")
        df = df.drop_duplicates(subset=KEY, keep="last").drop(columns="_live")
    else:
        df = new
    df = df.sort_values(KEY).reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def summarise_side(df):
    out = {}
    if df is None or df.empty:
        return out
    latest_date = df["date"].max()
    for otm, g in df.groupby("otm_target"):
        g = g.sort_values("date")
        latest = g[g["date"] == latest_date]
        rb = g["iv_hv_ratio_bid"].dropna()
        live_rb = g.loc[g["session"] == "live", "iv_hv_ratio_bid"].dropna()
        lv = lambda col: (float(latest[col].iloc[0])
                          if len(latest) and pd.notna(latest[col].iloc[0]) else None)
        out[f"{int(round(otm * 100))}pct_OTM"] = {
            "latest_date"                : latest_date,
            "latest_iv_hv_ratio_bid"     : lv("iv_hv_ratio_bid"),
            "latest_iv_hv_ratio_mid"     : lv("iv_hv_ratio_mid"),
            "mean_iv_hv_ratio_bid_all"   : round(float(rb.mean()), 3) if len(rb) else None,
            "mean_iv_hv_ratio_bid_20d"   : round(float(rb.tail(20).mean()), 3) if len(rb) else None,
            "mean_iv_hv_ratio_bid_live_only": round(float(live_rb.mean()), 3) if len(live_rb) else None,
            "n_valid"                    : int(len(rb)),
            "n_live"                     : int(len(live_rb)),
        }
    return out


def write_summary(dfs, path):
    out = {
        "generated_utc"      : datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "trading_days_logged": int(pd.concat([d["date"] for d in dfs.values() if d is not None]).nunique())
                               if any(d is not None for d in dfs.values()) else 0,
        "ratio_definition"   : "IV solved from the BID / 30-day realized vol (seller's view)",
        "calls"              : {"description": "OTM calls, strike ABOVE spot — covered-call side",
                                "breakeven_from_backtest": {"10pct_OTM": 1.09, "15pct_OTM": 1.10, "20pct_OTM": 1.00},
                                "levels": summarise_side(dfs.get("call"))},
        "puts"               : {"description": "OTM puts, strike BELOW spot — cash-secured-put side",
                                "breakeven_from_backtest": None,
                                "levels": summarise_side(dfs.get("put"))},
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def _print_rows(title, rows):
    print(f"  {title}")
    for r in rows:
        if "strike" not in r:
            print(f"    {r['otm_target']:.0%}: {r['note']}")
            continue
        print(f"    {r['otm_target']:.0%} OTM  K={r['strike']:<7} bid/ask {r['bid']:.2f}/{r['ask']:.2f}  "
              f"IV bid {r['iv_bid']}  mid {r['iv_mid']}  Yahoo {r['iv_yahoo']}  "
              f"IV/HV(bid) {r['iv_hv_ratio_bid']}  {'LOW PRICE' if r['low_price'] else ''} {r['note']}")


# ──────────────────────────────── MAIN ────────────────────────────────
def main():
    os.makedirs("data", exist_ok=True)
    migrate_legacy()
    now_utc = datetime.now(timezone.utc)
    session = market_session(now_utc)
    et_date = now_utc.astimezone(ET).strftime("%Y-%m-%d")

    def has_live_today(path):
        if not os.path.exists(path):
            return False
        d = pd.read_csv(path)
        return "session" in d.columns and ((d["date"] == et_date) & (d["session"] == "live")).any()

    if all(has_live_today(p) for p in LOG_PATHS.values()):
        print(f"{et_date}: live calls AND puts already logged — skipping ({session} run)")
        return

    tk = yf.Ticker(TICKER)
    hist = with_retries(tk.history, period="3mo", auto_adjust=True)
    if hist.empty:
        raise SystemExit("No price history returned")
    spot = float(hist["Close"].iloc[-1])      # during market hours: latest price
    hv30, hv10 = realized_vols(hist)

    expiries = with_retries(lambda: tk.options)
    expiry = pick_expiry(expiries, now_utc.astimezone(ET).date())
    if expiry is None:
        raise SystemExit("No suitable expiry found")
    chain = with_retries(tk.option_chain, expiry)

    rows = {"call": build_rows(spot, chain.calls, expiry, now_utc, hv30, hv10, "call"),
            "put" : build_rows(spot, chain.puts,  expiry, now_utc, hv30, hv10, "put")}
    if not any(r.get("bid", 0) > 0 for side in rows.values() for r in side):
        # Pre-market / holiday: Yahoo returns zero bids. Don't write empty rows.
        print(f"{et_date}: no bids on any target strike ({session}) — nothing logged")
        return

    dfs = {}
    for kind, path in LOG_PATHS.items():
        dfs[kind] = update_log(rows[kind], path)
    summary = write_summary(dfs, SUMMARY_PATH)

    print(f"{TICKER} spot {spot:.2f} | expiry {expiry} | HV30 {hv30:.1%} | HV10 {hv10:.1%} | session {session}")
    _print_rows("OTM CALLS (strike above spot)", rows["call"])
    _print_rows("OTM PUTS  (strike below spot)", rows["put"])
    print(f"  trading days logged: {summary['trading_days_logged']}")


if __name__ == "__main__":
    main()
