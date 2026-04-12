"""
BTC 3-State HMM + IVR + 5-Approach Backtest Scorer
====================================================
For each historical day, scores all 5 premium-maximisation approaches
and backtests which was most profitable using MSTR forward returns.

Approaches scored:
  1. Chop Transition   — sell put when Bear prob falling through 50%
  2. IVR Filter        — sell when HVR 50-70 (sweet spot)
  3. Covered Call Bear — sell covered call during Bear + high HVR
  4. Put Spread Trans  — sell put spread during Bear→Chop transition
  5. Calendar Premium  — sell longer DTE put when IV still elevated post-Bear

Update schedule: twice daily via GitHub Actions
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TICKER          = "BTC-USD"
MSTR_TICKER     = "MSTR"
LOOKBACK_DAYS   = 730
N_STATES        = 3
N_ITER          = 200
N_INIT          = 20
OUTPUT_PATH     = "data/hmm_output.json"
STATE_LABELS    = ["Bull", "Chop", "Bear"]
BACKTEST_DAYS   = 90     # how many days to backtest
FORWARD_DAYS    = 5      # look-forward window for outcome (weekly expiry)

IVR_LOW         = 30
IVR_SWEET_SPOT  = 50
IVR_HIGH        = 70
IVR_EXTREME     = 85
MA20_RECOVERY_PCT  = 4.0
MOMENTUM_RECOVERY  = 5.0
BEAR_SOFTENING     = 0.55


# ─────────────────────────────────────────────
# FETCH DATA
# ─────────────────────────────────────────────
def fetch_data(ticker: str, days: int):
    end   = datetime.utcnow()
    start = end - timedelta(days=days + 60)
    df    = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open","High","Low","Close","Volume"]].copy()
    df.dropna(inplace=True)
    return df


def fetch_mstr(days: int = 400):
    end   = datetime.utcnow()
    start = end - timedelta(days=days + 60)
    df    = yf.download(MSTR_TICKER, start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
    if df.empty:
        return None
    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].copy()
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
# HVR CALCULATION (rolling, for backtest)
# ─────────────────────────────────────────────
def calculate_rolling_hvr(mstr_df, lookback: int = 252) -> pd.DataFrame:
    """
    For each day, calculate HVR using only past data (no lookahead).
    Returns DataFrame with columns: hv_30d, hv_10d, hvr, vol_trend
    """
    if mstr_df is None or len(mstr_df) < 60:
        return None

    df        = mstr_df.copy()
    df["ret"] = df["Close"].pct_change()
    df["hv_30d"] = df["ret"].rolling(30).std() * np.sqrt(252) * 100
    df["hv_10d"] = df["ret"].rolling(10).std() * np.sqrt(252) * 100
    df.dropna(inplace=True)

    hvr_list = []
    for i in range(len(df)):
        start_idx = max(0, i - lookback + 1)
        window    = df["hv_30d"].iloc[start_idx : i + 1]
        curr      = df["hv_30d"].iloc[i]
        wmin, wmax = window.min(), window.max()
        if wmax == wmin:
            hvr = 50.0
        else:
            hvr = (curr - wmin) / (wmax - wmin) * 100
        hvr_list.append(round(min(max(hvr, 0), 100), 1))

    df["hvr"] = hvr_list

    # vol trend: 10d vs 30d
    df["vol_trend"] = np.where(df["hv_10d"] > df["hv_30d"] * 1.15, "rising",
                      np.where(df["hv_10d"] < df["hv_30d"] * 0.85, "falling", "stable"))
    return df


def get_hvr_snapshot(hvr_df, date) -> dict:
    """Get HVR metrics for a specific date (or nearest prior date)."""
    if hvr_df is None:
        return {"hvr": 50, "hv_30d": 60, "hv_10d": 60, "vol_trend": "stable",
                "premium_quality": "unknown", "ivr_label": "N/A", "sell_strategy": "N/A",
                "hv_52w_high": 100, "hv_52w_low": 20}
    try:
        row = hvr_df.loc[:date].iloc[-1]
    except Exception:
        row = hvr_df.iloc[-1]

    hvr = float(row["hvr"])
    hv_30d = float(row["hv_30d"])
    hv_10d = float(row["hv_10d"])
    vol_trend = str(row["vol_trend"])

    if hvr < IVR_LOW:
        label, quality, strategy = f"Low ({hvr:.0f})", "cheap", "Skip — premium too low"
    elif hvr < IVR_SWEET_SPOT:
        label, quality, strategy = f"Below Avg ({hvr:.0f})", "below_average", "Bull regime only"
    elif hvr < IVR_HIGH:
        label, quality, strategy = f"Elevated ({hvr:.0f})", "good", "Good zone — sell Chop/Bull"
    elif hvr < IVR_EXTREME:
        label, quality, strategy = f"High ({hvr:.0f})", "excellent", "Sell puts (Bull/Chop) OR covered calls (Bear)"
    else:
        label, quality, strategy = f"Extreme ({hvr:.0f})", "extreme", "Covered calls only"

    return {
        "hvr": hvr, "hv_30d": round(hv_30d, 1), "hv_10d": round(hv_10d, 1),
        "vol_trend": vol_trend, "ivr_label": label, "premium_quality": quality,
        "sell_strategy": strategy,
        "hv_52w_high": round(float(hvr_df["hv_30d"].tail(252).max()), 1),
        "hv_52w_low":  round(float(hvr_df["hv_30d"].tail(252).min()), 1),
    }


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def build_features(df) -> tuple:
    df = df.copy()
    df["return"]          = df["Close"].pct_change()
    df["realized_vol_7d"] = df["return"].rolling(7).std() * np.sqrt(252)
    df["ma20"]            = df["Close"].rolling(20).mean()
    df["ma20_distance"]   = (df["Close"] - df["ma20"]) / df["ma20"] * 100
    df["momentum_3d"]     = df["Close"].pct_change(3)
    df["vol_ratio"]       = df["Volume"] / df["Volume"].rolling(20).mean()
    df.dropna(inplace=True)
    df = df.tail(LOOKBACK_DAYS)
    feature_cols = ["return","realized_vol_7d","ma20_distance","momentum_3d","vol_ratio"]
    X_raw        = df[feature_cols].values
    scaler       = StandardScaler()
    X_scaled     = scaler.fit_transform(X_raw)
    return X_scaled, df, scaler, feature_cols


# ─────────────────────────────────────────────
# TRAIN HMM
# ─────────────────────────────────────────────
def train_hmm(X: np.ndarray) -> GaussianHMM:
    best_model, best_score = None, -np.inf
    for seed in range(N_INIT):
        m = GaussianHMM(n_components=N_STATES, covariance_type="full",
                        n_iter=N_ITER, random_state=seed, tol=1e-4)
        try:
            m.fit(X)
            s = m.score(X)
            if s > best_score:
                best_score, best_model = s, m
        except Exception:
            continue
    if best_model is None:
        raise RuntimeError("HMM failed to converge")
    return best_model


# ─────────────────────────────────────────────
# LABEL STATES
# ─────────────────────────────────────────────
def label_states(model, X, df):
    raw_states   = model.predict(X)
    mean_returns = {s: df["return"].values[raw_states==s].mean()
                    if (raw_states==s).sum() > 0 else 0.0
                    for s in range(N_STATES)}
    sorted_states = sorted(mean_returns, key=mean_returns.get, reverse=True)
    state_map     = {raw: lbl for raw, lbl in zip(sorted_states, STATE_LABELS)}
    return state_map, raw_states


# ─────────────────────────────────────────────
# 5-APPROACH SCORER  (core logic)
# ─────────────────────────────────────────────
def score_five_approaches(state: str, bear_prob: float, bear_prob_prev: float,
                           bear_prob_5d: list, chop_prob: float,
                           days_in_state: int, ma20_dist: float,
                           momentum_3d: float, hvr: float,
                           vol_trend: str) -> list:
    """
    Score each of the 5 approaches from 0–100.
    Higher = more appropriate for today's conditions.

    Returns list of dicts with name, score, rationale, action.
    """
    # Helper: is Bear prob declining?
    if len(bear_prob_5d) >= 3:
        bear_falling = bear_prob_5d[-1] < bear_prob_5d[0]
        bear_drop    = bear_prob_5d[0] - bear_prob_5d[-1]   # positive = improving
    else:
        bear_falling = bear_prob < bear_prob_prev
        bear_drop    = bear_prob_prev - bear_prob

    in_bear       = state == "Bear"
    in_chop       = state == "Chop"
    in_bull       = state == "Bull"
    was_bear      = (days_in_state <= 7 and in_chop)  # recently exited Bear
    transitioning = (in_chop or (in_bear and bear_falling and bear_prob < 0.60))

    # ── APPROACH 1: Chop Transition ─────────────────────────────────────
    a1 = 0
    a1_reasons = []

    if bear_falling:
        drop_score = min(40, int(bear_drop * 100))   # up to 40 pts for size of drop
        a1 += drop_score
        a1_reasons.append(f"Bear prob falling {bear_drop:.0%}")

    if 0.25 <= bear_prob <= 0.55:
        a1 += 25
        a1_reasons.append(f"Bear prob {bear_prob:.0%} in sweet spot (25–55%)")
    elif 0.55 < bear_prob <= 0.70:
        a1 += 10
        a1_reasons.append(f"Bear prob {bear_prob:.0%} still high — partial credit")

    if in_chop or (in_bear and bear_prob < 0.55):
        a1 += 20
        a1_reasons.append("Regime in Chop or Bear fading")

    if IVR_SWEET_SPOT <= hvr <= IVR_HIGH:
        a1 += 15
        a1_reasons.append(f"HVR {hvr:.0f} in sweet spot — premium still juicy")
    elif hvr > IVR_HIGH:
        a1 += 8
        a1_reasons.append(f"HVR {hvr:.0f} elevated but approaching extreme")

    a1 = min(100, a1)
    if a1 < 20:
        a1_action = "Not applicable — regime not transitioning"
    elif a1 < 50:
        a1_action = "Monitor — Bear showing early signs of fading"
    else:
        a1_action = f"Sell put 2 strikes below max pain — HVR {hvr:.0f} still elevated"

    # ── APPROACH 2: IVR Filter ───────────────────────────────────────────
    a2 = 0
    a2_reasons = []

    # IVR zone scoring
    if IVR_SWEET_SPOT <= hvr < IVR_HIGH:
        a2 += 50
        a2_reasons.append(f"HVR {hvr:.0f} in ideal 50–70 sweet spot")
    elif IVR_HIGH <= hvr < IVR_EXTREME:
        a2 += 35
        a2_reasons.append(f"HVR {hvr:.0f} high — premium excellent but add regime check")
    elif IVR_LOW <= hvr < IVR_SWEET_SPOT:
        a2 += 20
        a2_reasons.append(f"HVR {hvr:.0f} below sweet spot — premium thin")
    elif hvr >= IVR_EXTREME:
        a2 += 10
        a2_reasons.append(f"HVR {hvr:.0f} extreme — puts too risky at this level")
    else:
        a2 += 5
        a2_reasons.append(f"HVR {hvr:.0f} too low — not worth selling")

    # Regime compatibility
    if in_bull:
        a2 += 30
        a2_reasons.append("Bull regime — IVR filter most effective")
    elif in_chop and bear_prob < 0.35:
        a2 += 25
        a2_reasons.append("Chop with low Bear risk — good IVR play")
    elif transitioning:
        a2 += 15
        a2_reasons.append("Transitioning — IVR filter provides entry discipline")
    elif in_bear:
        a2 += 0
        a2_reasons.append("Bear regime — IVR filter says wait regardless of score")

    # Vol trend bonus
    if vol_trend == "falling":
        a2 += 20
        a2_reasons.append("Vol falling — IV crush starting, sell now before premium drops")
    elif vol_trend == "stable":
        a2 += 10
        a2_reasons.append("Vol stable — premium holding")

    a2 = min(100, a2)
    if in_bear:
        a2 = min(a2, 25)   # cap in Bear regardless of HVR
        a2_reasons.append("Capped — Bear regime overrides IVR filter for puts")

    if a2 < 30:
        a2_action = "Skip — IVR conditions not met"
    elif a2 < 60:
        a2_action = f"Possible — HVR {hvr:.0f} acceptable but check regime first"
    else:
        a2_action = f"Strong — HVR {hvr:.0f} in sell zone, regime supports entry"

    # ── APPROACH 3: Covered Call (Bear) ──────────────────────────────────
    a3 = 0
    a3_reasons = []

    if in_bear:
        a3 += 40
        a3_reasons.append(f"Bear regime day {days_in_state} — ideal for covered calls")
        if days_in_state >= 5:
            a3 += 15
            a3_reasons.append("Established Bear — call premium maximised")
    elif in_chop and was_bear:
        a3 += 20
        a3_reasons.append("Recently exited Bear — call premium still elevated")
    else:
        a3 += 5
        a3_reasons.append("Not in Bear — covered call suboptimal")

    if hvr >= IVR_HIGH:
        a3 += 30
        a3_reasons.append(f"HVR {hvr:.0f} — call premium excellent")
    elif hvr >= IVR_SWEET_SPOT:
        a3 += 20
        a3_reasons.append(f"HVR {hvr:.0f} — call premium good")
    else:
        a3 += 5
        a3_reasons.append(f"HVR {hvr:.0f} — call premium thin")

    if bear_prob >= 0.60:
        a3 += 15
        a3_reasons.append(f"Bear prob {bear_prob:.0%} — downside protection confirmed")

    a3 = min(100, a3)
    if a3 < 30:
        a3_action = "Not applicable — not in Bear regime"
    elif a3 < 60:
        a3_action = "Consider covered call if you hold MSTR shares"
    else:
        a3_action = f"Sell covered call 3–5% OTM — Bear day {days_in_state}, HVR {hvr:.0f}"

    # ── APPROACH 4: Put Spread Transition ────────────────────────────────
    a4 = 0
    a4_reasons = []

    if transitioning:
        a4 += 35
        a4_reasons.append("Regime transitioning — spread reduces tail risk")

    if vol_trend == "falling":
        a4 += 30
        a4_reasons.append("Vol falling — spread holds value better than naked put in IV crush")
    elif vol_trend == "stable":
        a4 += 15
        a4_reasons.append("Vol stable — spread still adds downside protection")

    if 0.20 <= bear_prob <= 0.50:
        a4 += 20
        a4_reasons.append(f"Bear prob {bear_prob:.0%} — spread width provides buffer")
    elif bear_prob > 0.50:
        a4 += 10
        a4_reasons.append(f"Bear prob still {bear_prob:.0%} — spread essential over naked put")

    if in_bull:
        a4 += 0   # no need for spread in Bull — naked put fine
        a4_reasons.append("Bull regime — naked put preferable to spread")
    elif in_chop:
        a4 += 15
        a4_reasons.append("Chop — spread protects against false Bull signal")

    a4 = min(100, a4)
    if a4 < 25:
        a4_action = "Not needed — regime clear enough for naked put or skip entirely"
    elif a4 < 55:
        a4_action = "Consider spread — sells $125P / buys $115P for downside protection"
    else:
        a4_action = "Use put spread — sell upper, buy lower 2 strikes down. IV crush risk managed"

    # ── APPROACH 5: Calendar Premium ─────────────────────────────────────
    a5 = 0
    a5_reasons = []

    if was_bear or (in_bear and bear_falling and bear_prob < 0.50):
        a5 += 35
        a5_reasons.append("Bear ending — longer DTE options still have elevated IV")

    if hvr >= IVR_SWEET_SPOT:
        a5 += 25
        a5_reasons.append(f"HVR {hvr:.0f} — 30–45 DTE options priced attractively")

    if vol_trend == "stable":
        a5 += 20
        a5_reasons.append("Vol stable — 30–45 DTE premium will decay without immediate crush")
    elif vol_trend == "falling":
        a5 += 10
        a5_reasons.append("Vol falling — calendar viable but close early (10–14 days)")
    elif vol_trend == "rising":
        a5 += 5
        a5_reasons.append("Vol rising — calendar risky, avoid")

    if bear_falling and 0.20 <= bear_prob <= 0.55:
        a5 += 20
        a5_reasons.append("Regime improving — 30 DTE gives time for recovery while keeping premium")

    a5 = min(100, a5)
    if a5 < 25:
        a5_action = "Not applicable — regime/vol conditions don't support calendar"
    elif a5 < 55:
        a5_action = "Possible — sell 30 DTE put while IV still elevated, plan to close at 50% profit"
    else:
        a5_action = "Sell 30–45 DTE put while IV elevated post-Bear — close after 10–14 days"

    # ── Normalise so scores are relative percentages ─────────────────────
    scores = [a1, a2, a3, a4, a5]
    total  = sum(scores)
    if total == 0:
        pcts = [20.0] * 5
    else:
        pcts = [round(s / total * 100, 1) for s in scores]

    approaches = [
        {
            "id"       : 1,
            "name"     : "Chop Transition",
            "score"    : a1,
            "pct"      : pcts[0],
            "action"   : a1_action,
            "reasons"  : a1_reasons,
            "best_when": "Bear prob falling 78%→45%, HVR 50–70",
        },
        {
            "id"       : 2,
            "name"     : "IVR Filter",
            "score"    : a2,
            "pct"      : pcts[1],
            "action"   : a2_action,
            "reasons"  : a2_reasons,
            "best_when": "HVR 50–70 + Bull or Chop regime",
        },
        {
            "id"       : 3,
            "name"     : "Covered Call (Bear)",
            "score"    : a3,
            "pct"      : pcts[2],
            "action"   : a3_action,
            "reasons"  : a3_reasons,
            "best_when": "Bear regime + HVR > 70 + hold MSTR shares",
        },
        {
            "id"       : 4,
            "name"     : "Put Spread (Transition)",
            "score"    : a4,
            "pct"      : pcts[3],
            "action"   : a4_action,
            "reasons"  : a4_reasons,
            "best_when": "Bear→Chop transition + vol falling (IV crush imminent)",
        },
        {
            "id"       : 5,
            "name"     : "Calendar Premium",
            "score"    : a5,
            "pct"      : pcts[4],
            "action"   : a5_action,
            "reasons"  : a5_reasons,
            "best_when": "Bear just ended, HVR still elevated, sell 30–45 DTE",
        },
    ]

    # Sort by score descending — top recommended first
    approaches.sort(key=lambda x: x["score"], reverse=True)
    approaches[0]["recommended"] = True
    for a in approaches[1:]:
        a["recommended"] = False

    return approaches


# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────
def run_backtest(df_btc, log_posteriors, raw_states, state_map,
                 mstr_df, hvr_df) -> dict:
    """
    For each of the last BACKTEST_DAYS days:
    1. Compute 5-approach scores
    2. Look forward FORWARD_DAYS to see if MSTR went up or down
    3. Map outcome to whether the recommended approach was correct
    4. Calculate win rate and avg return per approach
    """
    if mstr_df is None or len(mstr_df) < FORWARD_DAYS + BACKTEST_DAYS:
        return {"available": False, "reason": "Insufficient MSTR data"}

    bull_state = [k for k, v in state_map.items() if v == "Bull"][0]
    chop_state = [k for k, v in state_map.items() if v == "Chop"][0]
    bear_state = [k for k, v in state_map.items() if v == "Bear"][0]

    results   = []
    n         = len(df_btc)
    tail      = min(BACKTEST_DAYS + FORWARD_DAYS, n - 5)

    approach_stats = {i: {"wins": 0, "losses": 0, "skips": 0, "returns": []} for i in range(1, 6)}

    for idx in range(tail, n - FORWARD_DAYS):
        date      = df_btc.index[idx]
        date_str  = date.strftime("%Y-%m-%d")

        # Regime at this day
        prob_vec   = log_posteriors[idx]
        bear_prob  = float(prob_vec[bear_state])
        bull_prob  = float(prob_vec[bull_state])
        chop_prob  = float(prob_vec[chop_state])
        raw_s      = raw_states[idx]
        state      = state_map[raw_s]

        # Days in current state
        days_in = 0
        for k in range(idx, max(0, idx - 30), -1):
            if raw_states[k] == raw_s:
                days_in += 1
            else:
                break

        # Bear prob previous day
        bear_prob_prev = float(log_posteriors[idx-1][bear_state]) if idx > 0 else bear_prob

        # Bear prob last 5 days
        bp5 = [float(log_posteriors[max(0,idx-4+j)][bear_state]) for j in range(5)]

        # BTC features
        ma20_dist  = float(df_btc["ma20_distance"].iloc[idx])
        momentum   = float(df_btc["momentum_3d"].iloc[idx]) * 100

        # HVR at this date
        ivr_snap  = get_hvr_snapshot(hvr_df, date)
        hvr       = ivr_snap["hvr"]
        vol_trend = ivr_snap["vol_trend"]

        # Score approaches
        approaches = score_five_approaches(
            state=state, bear_prob=bear_prob, bear_prob_prev=bear_prob_prev,
            bear_prob_5d=bp5, chop_prob=chop_prob, days_in_state=days_in,
            ma20_dist=ma20_dist, momentum_3d=momentum,
            hvr=hvr, vol_trend=vol_trend
        )

        # MSTR forward return (5 days)
        try:
            mstr_dates  = mstr_df.index
            future_idx  = mstr_dates.searchsorted(date)
            if future_idx + FORWARD_DAYS >= len(mstr_df):
                continue
            price_now   = float(mstr_df["Close"].iloc[future_idx])
            price_fwd   = float(mstr_df["Close"].iloc[future_idx + FORWARD_DAYS])
            fwd_return  = (price_fwd - price_now) / price_now * 100
            mstr_up     = fwd_return >= 0
        except Exception:
            continue

        # Map outcome to each approach's correctness
        # Approach 1,2,4,5 → success if MSTR goes UP (put expires OTM)
        # Approach 3 (covered call) → success if MSTR stays flat or goes DOWN (call expires OTM)
        day_result = {
            "date"       : date_str,
            "state"      : state,
            "bear_prob"  : round(bear_prob, 3),
            "hvr"        : hvr,
            "fwd_return" : round(fwd_return, 2),
            "mstr_up"    : mstr_up,
            "top_approach": approaches[0]["id"],
            "scores"     : {a["id"]: a["score"] for a in approaches},
            "pcts"       : {a["id"]: a["pct"]   for a in approaches},
        }

        for a in approaches:
            aid     = a["id"]
            score   = a["score"]

            # Only count the approach if it was actually recommended (score > 40)
            if score < 40:
                approach_stats[aid]["skips"] += 1
                continue

            if aid == 3:
                # Covered call wins if MSTR flat/down (call not exercised)
                correct = not mstr_up
            else:
                # Put-based strategies win if MSTR goes up (put expires OTM)
                correct = mstr_up

            if correct:
                approach_stats[aid]["wins"] += 1
            else:
                approach_stats[aid]["losses"] += 1

            approach_stats[aid]["returns"].append(fwd_return if aid != 3 else -fwd_return)

        results.append(day_result)

    # Summarise stats per approach
    approach_names = {
        1: "Chop Transition",
        2: "IVR Filter",
        3: "Covered Call (Bear)",
        4: "Put Spread (Transition)",
        5: "Calendar Premium",
    }

    summary = []
    for aid in range(1, 6):
        stats  = approach_stats[aid]
        w, l   = stats["wins"], stats["losses"]
        total  = w + l
        wr     = round(w / total * 100, 1) if total > 0 else None
        rets   = stats["returns"]
        avg_ret = round(float(np.mean(rets)), 2) if rets else None
        summary.append({
            "id"          : aid,
            "name"        : approach_names[aid],
            "wins"        : w,
            "losses"      : l,
            "skips"       : stats["skips"],
            "win_rate_pct": wr,
            "avg_fwd_return": avg_ret,
            "sample_size" : total,
        })

    # Sort by win rate
    summary.sort(key=lambda x: (x["win_rate_pct"] or 0), reverse=True)

    return {
        "available"         : True,
        "backtest_days"     : BACKTEST_DAYS,
        "forward_window"    : FORWARD_DAYS,
        "total_observations": len(results),
        "summary"           : summary,
        "daily"             : results[-30:],  # last 30 days for chart
    }


# ─────────────────────────────────────────────
# COMPUTE OUTPUTS
# ─────────────────────────────────────────────
def compute_outputs(model, X, df, state_map, raw_states, ivr_data,
                    mstr_df, hvr_df) -> dict:

    log_posteriors    = model.predict_proba(X)
    latest_probs_raw  = log_posteriors[-1]
    latest_state_raw  = raw_states[-1]
    latest_state_name = state_map[latest_state_raw]

    bull_state = [k for k, v in state_map.items() if v == "Bull"][0]
    chop_state = [k for k, v in state_map.items() if v == "Chop"][0]
    bear_state = [k for k, v in state_map.items() if v == "Bear"][0]

    bull_prob  = float(latest_probs_raw[bull_state])
    chop_prob  = float(latest_probs_raw[chop_state])
    bear_prob  = float(latest_probs_raw[bear_state])
    confidence = float(max(bull_prob, chop_prob, bear_prob))

    days_in_state = 0
    for s in reversed(raw_states):
        if s == latest_state_raw:
            days_in_state += 1
        else:
            break

    prev_probs_raw      = log_posteriors[-2] if len(log_posteriors) > 1 else latest_probs_raw
    prev_bear_prob      = float(prev_probs_raw[bear_state])
    bear_probs_5d       = [float(log_posteriors[-(5-i)][bear_state]) for i in range(min(5, len(log_posteriors)))]
    regime_change_alert = (bear_prob >= 0.30 and prev_bear_prob < 0.30)
    state_changed       = (raw_states[-1] != raw_states[-2]) if len(raw_states) > 1 else False

    if len(bear_probs_5d) >= 3:
        bear_trend = "improving" if bear_probs_5d[-1] < bear_probs_5d[0] else \
                     "deteriorating" if bear_probs_5d[-1] > bear_probs_5d[0] else "stable"
    else:
        bear_trend = "stable"

    latest_close   = float(df["Close"].iloc[-1])
    latest_ma20    = float(df["ma20"].iloc[-1])
    ma20_dist_pct  = round((latest_close - latest_ma20) / latest_ma20 * 100, 2)
    momentum_pct   = round(float(df["momentum_3d"].iloc[-1]) * 100, 2)
    latest_date    = df.index[-1].strftime("%Y-%m-%d")

    # Score 5 approaches for TODAY
    today_approaches = score_five_approaches(
        state=latest_state_name, bear_prob=bear_prob,
        bear_prob_prev=prev_bear_prob, bear_prob_5d=bear_probs_5d,
        chop_prob=chop_prob, days_in_state=days_in_state,
        ma20_dist=ma20_dist_pct, momentum_3d=momentum_pct,
        hvr=ivr_data.get("hvr", 50), vol_trend=ivr_data.get("vol_trend","stable")
    )

    # Recommendation from top-scoring approach
    top = today_approaches[0]
    recommendation = get_recommendation(
        state=latest_state_name, bear_prob=bear_prob, chop_prob=chop_prob,
        days_in_state=days_in_state, ma20_dist_pct=ma20_dist_pct,
        momentum_pct=momentum_pct, bear_trend=bear_trend, ivr_data=ivr_data
    )

    # History (last 90 days)
    history = []
    tail    = min(90, len(df))
    for i, (date, raw_s) in enumerate(zip(df.index[-tail:], raw_states[-tail:])):
        probs = log_posteriors[-(tail - i)]
        history.append({
            "date"         : date.strftime("%Y-%m-%d"),
            "state"        : state_map[raw_s],
            "bull_prob"    : round(float(probs[bull_state]), 4),
            "chop_prob"    : round(float(probs[chop_state]), 4),
            "bear_prob"    : round(float(probs[bear_state]), 4),
            "close"        : round(float(df["Close"].iloc[-(tail-i)]), 2),
            "ma20_distance": round(float(df["ma20_distance"].iloc[-(tail-i)]), 2),
        })

    # Transition matrix
    raw_trans    = model.transmat_
    ordered      = [bull_state, chop_state, bear_state]
    trans_matrix = [[round(float(raw_trans[i][j]), 4) for j in ordered] for i in ordered]

    # Backtest
    print("📈 Running 5-approach backtest...")
    backtest = run_backtest(df, log_posteriors, raw_states, state_map, mstr_df, hvr_df)

    return {
        "generated_at"          : datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "latest_date"           : latest_date,
        "btc_close"             : round(latest_close, 2),
        "btc_ma20"              : round(latest_ma20, 2),
        "btc_ma20_distance_pct" : ma20_dist_pct,
        "realized_vol_7d"       : round(float(df["realized_vol_7d"].iloc[-1]) * 100, 2),
        "momentum_3d_pct"       : momentum_pct,
        "bear_prob_trend"       : bear_trend,
        "bear_probs_5d"         : [round(p, 4) for p in bear_probs_5d],
        "current_state"         : latest_state_name,
        "bull_probability"      : round(bull_prob, 4),
        "chop_probability"      : round(chop_prob, 4),
        "bear_probability"      : round(bear_prob, 4),
        "state_confidence"      : round(confidence, 4),
        "days_in_current_state" : int(days_in_state),
        "regime_change_alert"   : bool(regime_change_alert or state_changed),
        "transition_matrix"     : trans_matrix,
        "history"               : history,
        "ivr"                   : ivr_data,
        "approaches_today"      : today_approaches,
        "backtest"              : backtest,
        "recommendation"        : recommendation,
    }


# ─────────────────────────────────────────────
# RECOMMENDATION ENGINE (unchanged from v2)
# ─────────────────────────────────────────────
def get_recommendation(state, bear_prob, chop_prob, days_in_state,
                        ma20_dist_pct, momentum_pct, bear_trend, ivr_data):
    hvr             = ivr_data.get("hvr") or 50
    premium_quality = ivr_data.get("premium_quality", "unknown")
    vol_trend       = ivr_data.get("vol_trend", "stable")

    ma20_recovering = (ma20_dist_pct >= MA20_RECOVERY_PCT and
                       momentum_pct  >= MOMENTUM_RECOVERY and
                       bear_prob     <  BEAR_SOFTENING and
                       bear_trend    == "improving")

    ivr_note = (f"⚠️ HVR {hvr:.0f} — premium cheap"              if premium_quality=="cheap"       else
                f"✅ HVR {hvr:.0f} — premium elevated"            if premium_quality in ("good","excellent") else
                f"🚨 HVR {hvr:.0f} — extreme, covered calls only" if premium_quality=="extreme"     else
                f"HVR {hvr:.0f}")

    if state == "Bear" and ma20_recovering:
        signal, action = "YELLOW", "Bear recovering — sell put spread or far OTM put"
        strike_adj     = "2–3 strikes below max pain (put spread preferred)"
        caution        = f"Bear prob {bear_prob:.0%} improving. {ivr_note}"
        cc             = "✅ Also consider covered call 3–5% OTM"

    elif state == "Bear" and days_in_state > 5 and hvr >= IVR_HIGH:
        signal, action = "ORANGE", "No new puts — sell covered calls to monetize Bear premium"
        strike_adj     = "Covered call 3–5% OTM on MSTR shares"
        caution        = f"Bear day {days_in_state}. {ivr_note}"
        cc             = f"✅ BEST BEAR PLAY: HVR {hvr:.0f} = excellent call premium"

    elif state == "Bear" and days_in_state > 5:
        signal, action = "ORANGE", "No new puts — covered call viable if you hold shares"
        strike_adj     = "Covered call 4–6% OTM"
        caution        = f"Bear day {days_in_state}. {ivr_note}"
        cc             = f"🟡 Covered call viable — HVR {hvr:.0f}"

    elif state == "Bear":
        signal, action = "ORANGE", "Fresh Bear — manage existing, no new puts"
        strike_adj     = "Close or roll existing positions down"
        caution        = f"Fresh Bear day {days_in_state}. {ivr_note}"
        cc             = "🟡 Covered call — wait 2–3 days to confirm Bear"

    elif state == "Chop" and bear_prob < 0.35 and bear_trend == "improving":
        if hvr >= IVR_SWEET_SPOT:
            signal, action = "YELLOW", "Regime improving — sell put spread at elevated premium"
            strike_adj     = "2 strikes below max pain — put spread preferred"
            cc             = "🟡 Covered call still viable as secondary"
        else:
            signal, action = "YELLOW", "Improving but premium thin — wait for better HVR"
            strike_adj     = "Wait for HVR > 50"
            cc             = "❌ Covered call — premium too low"
        caution = f"Chop transitioning. {ivr_note}"

    elif state == "Chop" and bear_prob >= 0.35:
        signal, action = "ORANGE", "Chop with elevated Bear risk — far OTM only"
        strike_adj     = "3+ strikes below max pain"
        caution        = f"Bear prob {bear_prob:.0%} still high. {ivr_note}"
        cc             = "🟡 Covered call if HVR > 60"

    elif state == "Chop":
        signal, action = "YELLOW", "Chop — sell puts 2 strikes below max pain"
        strike_adj     = "2 strikes below max pain"
        caution        = f"Magnet unreliable in Chop. {ivr_note}"
        cc             = "❌ Skip covered call in Chop"

    elif state == "Bull" and bear_prob < 0.15 and hvr >= IVR_SWEET_SPOT:
        signal, action = "GREEN", "Safe to sell puts at max pain — elevated premium"
        strike_adj     = "At max pain strike"
        caution        = f"Monitor Bear prob > 20%. {ivr_note}"
        cc             = "❌ Skip covered call — want upside exposure in Bull"

    elif state == "Bull" and bear_prob < 0.15:
        signal, action = "GREEN", "Safe to sell puts at max pain"
        strike_adj     = "At max pain strike"
        caution        = ivr_note
        cc             = "❌ Skip covered call in Bull"

    else:
        signal, action = "YELLOW", "Bull weakening — 1 strike below max pain"
        strike_adj     = "1 strike below max pain"
        caution        = f"Bull losing momentum. {ivr_note}"
        cc             = "🟡 Consider covered call as hedge"

    return {"signal": signal, "action": action, "strike_adj": strike_adj,
            "caution": caution, "covered_call": cc, "ivr_note": ivr_note}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    import os
    os.makedirs("data", exist_ok=True)

    print("📥 Fetching BTC-USD data...")
    df = fetch_data(TICKER, LOOKBACK_DAYS)
    print(f"   {len(df)} rows fetched")

    print("📊 Fetching MSTR data (IVR + backtest)...")
    mstr_df = fetch_mstr(days=400)
    hvr_df  = calculate_rolling_hvr(mstr_df)
    ivr_data = get_hvr_snapshot(hvr_df, df.index[-1])
    if ivr_data["hvr"] is not None:
        print(f"   MSTR HVR: {ivr_data['hvr']:.1f} — {ivr_data['ivr_label']}")

    print("🔧 Engineering features...")
    X, df_feat, scaler, feature_cols = build_features(df)
    print(f"   Training on {len(X)} days")

    print(f"🧠 Training HMM ({N_INIT} seeds)...")
    model = train_hmm(X)
    print(f"   Log-likelihood: {model.score(X):.2f}")

    print("🏷️  Labelling states...")
    state_map, raw_states = label_states(model, X, df_feat)
    print(f"   State map: {state_map}")

    print("📊 Computing outputs + backtest...")
    output = compute_outputs(model, X, df_feat, state_map, raw_states,
                             ivr_data, mstr_df, hvr_df)

    print(f"💾 Writing to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    rec = output["recommendation"]
    print(f"\n✅ Regime: {output['current_state']} "
          f"(Bull {output['bull_probability']:.0%} / "
          f"Chop {output['chop_probability']:.0%} / "
          f"Bear {output['bear_probability']:.0%})")
    print(f"   Signal: {rec['signal']} — {rec['action']}")
    print(f"\n📋 Today's 5-Approach Scores:")
    for a in output["approaches_today"]:
        star = " ★ RECOMMENDED" if a["recommended"] else ""
        print(f"   A{a['id']} {a['name']:<28} score={a['score']:3d}  pct={a['pct']:5.1f}%{star}")

    if output["backtest"]["available"]:
        print(f"\n📈 Backtest Win Rates (last {BACKTEST_DAYS} days, {FORWARD_DAYS}d forward):")
        for s in output["backtest"]["summary"]:
            wr = f"{s['win_rate_pct']}%" if s["win_rate_pct"] is not None else "N/A"
            print(f"   A{s['id']} {s['name']:<28} WR={wr:>6}  n={s['sample_size']}")


if __name__ == "__main__":
    main()
