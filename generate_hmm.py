"""
BTC 3-State HMM Regime Detector + IVR Tracker
===============================================
Fetches BTC-USD daily OHLCV + MSTR historical vol from Yahoo Finance,
trains a 3-state Gaussian HMM (Bull / Chop / Bear),
calculates MSTR Historical Volatility Rank (HVR) as IVR proxy,
and writes results to data/hmm_output.json for the dashboard.

Update schedule: twice daily via GitHub Actions
  - 08:00 ET (pre-market)
  - 16:30 ET (post-close)
"""

import json
import warnings
import numpy as np
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

# IVR thresholds
IVR_LOW         = 30    # below = cheap premium, skip selling
IVR_SWEET_SPOT  = 50    # above = elevated premium, good to sell
IVR_HIGH        = 70    # above = expensive premium, best selling zone
IVR_EXTREME     = 85    # above = panic premium, too risky for puts

# MA20 override thresholds
MA20_RECOVERY_PCT   = 4.0   # BTC > 4% above MA20 = recovering
MOMENTUM_RECOVERY   = 5.0   # 3-day momentum > 5% = strong bounce
BEAR_SOFTENING      = 0.55  # Bear prob must be below 55% for full override


# ─────────────────────────────────────────────
# STEP 1 — FETCH DATA
# ─────────────────────────────────────────────
def fetch_data(ticker: str, days: int):
    """Download daily OHLCV from Yahoo Finance."""
    end   = datetime.utcnow()
    start = end - timedelta(days=days + 60)
    df    = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
# STEP 1B — FETCH MSTR FOR IVR
# ─────────────────────────────────────────────
def fetch_mstr_for_ivr(days: int = 365):
    """
    Fetch MSTR daily data to compute Historical Volatility Rank.
    HVR = proxy for IVR since Yahoo Finance lacks historical IV data.
    Uses 252 trading days (1 year) for the rank calculation.
    """
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


def calculate_hvr(mstr_df) -> dict:
    """
    Calculate MSTR Historical Volatility Rank (HVR) — IVR proxy.

    Steps:
    1. Calculate 30-day rolling realized vol (annualized) for each day
    2. Take the last 252 trading days of those vol readings
    3. HVR = rank of today's vol within that 1-year range

    Also calculates:
    - Current 30-day HV
    - 52-week HV high and low
    - 10-day HV (short-term vol)
    - Vol trend: is vol rising or falling?
    """
    if mstr_df is None or len(mstr_df) < 60:
        return {
            "hvr"              : None,
            "hv_30d"           : None,
            "hv_10d"           : None,
            "hv_52w_high"      : None,
            "hv_52w_low"       : None,
            "vol_trend"        : "unknown",
            "ivr_label"        : "N/A",
            "premium_quality"  : "unknown",
            "sell_strategy"    : "Insufficient data",
        }

    df = mstr_df.copy()

    # Daily returns
    df["return"] = df["Close"].pct_change()

    # 30-day realized vol (annualized) — rolling
    df["hv_30d"] = df["return"].rolling(30).std() * np.sqrt(252) * 100   # as %

    # 10-day realized vol (annualized) — for short-term trend
    df["hv_10d"] = df["return"].rolling(10).std() * np.sqrt(252) * 100

    df.dropna(inplace=True)

    if len(df) < 30:
        return {
            "hvr"              : None,
            "hv_30d"           : None,
            "hv_10d"           : None,
            "hv_52w_high"      : None,
            "hv_52w_low"       : None,
            "vol_trend"        : "unknown",
            "ivr_label"        : "N/A",
            "premium_quality"  : "unknown",
            "sell_strategy"    : "Insufficient data",
        }

    # Use last 252 trading days for 52-week range
    lookback = min(252, len(df))
    hv_series = df["hv_30d"].iloc[-lookback:]

    current_hv  = float(df["hv_30d"].iloc[-1])
    hv_10d      = float(df["hv_10d"].iloc[-1])
    hv_52w_high = float(hv_series.max())
    hv_52w_low  = float(hv_series.min())

    # HVR calculation
    if hv_52w_high == hv_52w_low:
        hvr = 50.0
    else:
        hvr = (current_hv - hv_52w_low) / (hv_52w_high - hv_52w_low) * 100

    hvr = round(min(max(hvr, 0), 100), 1)

    # Vol trend — is short-term vol rising vs 30-day?
    if hv_10d > current_hv * 1.15:
        vol_trend = "rising"     # short-term vol elevated vs 30d = vol expanding
    elif hv_10d < current_hv * 0.85:
        vol_trend = "falling"    # short-term vol below 30d = vol contracting
    else:
        vol_trend = "stable"

    # IVR label and premium quality
    if hvr < IVR_LOW:
        ivr_label       = f"Low ({hvr:.0f})"
        premium_quality = "cheap"
        sell_strategy   = "Skip selling — premium too low for risk taken"
    elif hvr < IVR_SWEET_SPOT:
        ivr_label       = f"Below Average ({hvr:.0f})"
        premium_quality = "below_average"
        sell_strategy   = "Sell only in confirmed Bull regime at max pain"
    elif hvr < IVR_HIGH:
        ivr_label       = f"Elevated ({hvr:.0f})"
        premium_quality = "good"
        sell_strategy   = "Good selling zone — sell in Chop transition or Bull"
    elif hvr < IVR_EXTREME:
        ivr_label       = f"High ({hvr:.0f})"
        premium_quality = "excellent"
        sell_strategy   = "Excellent premium — sell puts (Bull/Chop) OR covered calls (Bear)"
    else:
        ivr_label       = f"Extreme ({hvr:.0f})"
        premium_quality = "extreme"
        sell_strategy   = "Extreme premium — covered calls only, puts too risky"

    return {
        "hvr"              : hvr,
        "hv_30d"           : round(current_hv, 1),
        "hv_10d"           : round(hv_10d, 1),
        "hv_52w_high"      : round(hv_52w_high, 1),
        "hv_52w_low"       : round(hv_52w_low, 1),
        "vol_trend"        : vol_trend,
        "ivr_label"        : ivr_label,
        "premium_quality"  : premium_quality,
        "sell_strategy"    : sell_strategy,
    }


# ─────────────────────────────────────────────
# STEP 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
def build_features(df) -> tuple:
    """
    Compute 5 features from daily OHLCV.
    Drop first 20 rows (Option A) to avoid NaN from MA/vol lookback.
    """
    import pandas as pd

    df = df.copy()

    df["return"]          = df["Close"].pct_change()
    df["realized_vol_7d"] = df["return"].rolling(7).std() * np.sqrt(252)
    df["ma20"]            = df["Close"].rolling(20).mean()
    df["ma20_distance"]   = (df["Close"] - df["ma20"]) / df["ma20"] * 100
    df["momentum_3d"]     = df["Close"].pct_change(3)
    df["vol_ratio"]       = df["Volume"] / df["Volume"].rolling(20).mean()

    df.dropna(inplace=True)
    df = df.tail(LOOKBACK_DAYS)

    feature_cols = ["return", "realized_vol_7d", "ma20_distance", "momentum_3d", "vol_ratio"]
    X_raw        = df[feature_cols].values
    scaler       = StandardScaler()
    X_scaled     = scaler.fit_transform(X_raw)

    return X_scaled, df, scaler, feature_cols


# ─────────────────────────────────────────────
# STEP 3 — TRAIN HMM
# ─────────────────────────────────────────────
def train_hmm(X: np.ndarray) -> GaussianHMM:
    best_model  = None
    best_score  = -np.inf

    for seed in range(N_INIT):
        model = GaussianHMM(
            n_components    = N_STATES,
            covariance_type = "full",
            n_iter          = N_ITER,
            random_state    = seed,
            tol             = 1e-4,
        )
        try:
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("HMM failed to converge on all seeds")

    return best_model


# ─────────────────────────────────────────────
# STEP 4 — LABEL STATES
# ─────────────────────────────────────────────
def label_states(model: GaussianHMM, X: np.ndarray, df) -> dict:
    raw_states   = model.predict(X)
    mean_returns = {}
    for s in range(N_STATES):
        mask = raw_states == s
        mean_returns[s] = df["return"].values[mask].mean() if mask.sum() > 0 else 0.0

    sorted_states = sorted(mean_returns, key=mean_returns.get, reverse=True)
    state_map     = {raw: label for raw, label in zip(sorted_states, STATE_LABELS)}

    return state_map, raw_states


# ─────────────────────────────────────────────
# STEP 5 — COMPUTE OUTPUTS
# ─────────────────────────────────────────────
def compute_outputs(model: GaussianHMM, X: np.ndarray,
                    df, state_map: dict, raw_states: np.ndarray,
                    ivr_data: dict) -> dict:

    log_posteriors    = model.predict_proba(X)
    latest_probs_raw  = log_posteriors[-1]
    latest_state_raw  = raw_states[-1]
    latest_state_name = state_map[latest_state_raw]

    bull_state = [k for k, v in state_map.items() if v == "Bull"][0]
    chop_state = [k for k, v in state_map.items() if v == "Chop"][0]
    bear_state = [k for k, v in state_map.items() if v == "Bear"][0]

    bull_prob         = float(latest_probs_raw[bull_state])
    chop_prob         = float(latest_probs_raw[chop_state])
    bear_prob         = float(latest_probs_raw[bear_state])
    state_confidence  = float(max(bull_prob, chop_prob, bear_prob))

    days_in_state = 0
    for s in reversed(raw_states):
        if s == latest_state_raw:
            days_in_state += 1
        else:
            break

    prev_probs_raw      = log_posteriors[-2] if len(log_posteriors) > 1 else latest_probs_raw
    prev_bear_prob      = float(prev_probs_raw[bear_state])
    regime_change_alert = (bear_prob >= 0.30 and prev_bear_prob < 0.30)
    state_changed       = (raw_states[-1] != raw_states[-2]) if len(raw_states) > 1 else False

    # Bear probability trend (last 5 days)
    bear_probs_5d = [float(log_posteriors[-(5-i)][bear_state]) for i in range(min(5, len(log_posteriors)))]
    if len(bear_probs_5d) >= 3:
        bear_trend = "improving" if bear_probs_5d[-1] < bear_probs_5d[0] else "deteriorating" if bear_probs_5d[-1] > bear_probs_5d[0] else "stable"
    else:
        bear_trend = "stable"

    # Historical state sequence (last 90 days)
    history = []
    tail    = min(90, len(df))
    dates   = df.index[-tail:]
    for i, (date, raw_s) in enumerate(zip(dates, raw_states[-tail:])):
        probs = log_posteriors[-(tail - i)]
        history.append({
            "date"          : date.strftime("%Y-%m-%d"),
            "state"         : state_map[raw_s],
            "bull_prob"     : round(float(probs[bull_state]), 4),
            "chop_prob"     : round(float(probs[chop_state]), 4),
            "bear_prob"     : round(float(probs[bear_state]), 4),
            "close"         : round(float(df["Close"].iloc[-(tail - i)]), 2),
            "ma20_distance" : round(float(df["ma20_distance"].iloc[-(tail - i)]), 2),
        })

    # Transition matrix
    raw_trans    = model.transmat_
    ordered      = [bull_state, chop_state, bear_state]
    trans_matrix = [[round(float(raw_trans[i][j]), 4) for j in ordered] for i in ordered]

    latest_close    = float(df["Close"].iloc[-1])
    latest_ma20     = float(df["ma20"].iloc[-1])
    latest_vol_7d   = float(df["realized_vol_7d"].iloc[-1])
    latest_momentum = float(df["momentum_3d"].iloc[-1])
    latest_date     = df.index[-1].strftime("%Y-%m-%d")
    ma20_dist_pct   = round((latest_close - latest_ma20) / latest_ma20 * 100, 2)
    momentum_pct    = round(latest_momentum * 100, 2)

    # Recommendation with IVR + MA20 override
    recommendation = get_recommendation(
        state         = latest_state_name,
        bear_prob     = bear_prob,
        chop_prob     = chop_prob,
        days_in_state = days_in_state,
        ma20_dist_pct = ma20_dist_pct,
        momentum_pct  = momentum_pct,
        bear_trend    = bear_trend,
        ivr_data      = ivr_data,
    )

    output = {
        "generated_at"          : datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "latest_date"           : latest_date,
        "btc_close"             : round(latest_close, 2),
        "btc_ma20"              : round(latest_ma20, 2),
        "btc_ma20_distance_pct" : ma20_dist_pct,
        "realized_vol_7d"       : round(latest_vol_7d * 100, 2),
        "momentum_3d_pct"       : momentum_pct,
        "bear_prob_trend"       : bear_trend,
        "bear_probs_5d"         : [round(p, 4) for p in bear_probs_5d],
        "current_state"         : latest_state_name,
        "bull_probability"      : round(bull_prob, 4),
        "chop_probability"      : round(chop_prob, 4),
        "bear_probability"      : round(bear_prob, 4),
        "state_confidence"      : round(state_confidence, 4),
        "days_in_current_state" : int(days_in_state),
        "regime_change_alert"   : bool(regime_change_alert or state_changed),
        "transition_matrix"     : trans_matrix,
        "history"               : history,
        "ivr"                   : ivr_data,
        "recommendation"        : recommendation,
    }

    return output


# ─────────────────────────────────────────────
# STEP 6 — RECOMMENDATION ENGINE
# (with IVR + MA20 override + covered call logic)
# ─────────────────────────────────────────────
def get_recommendation(state: str, bear_prob: float, chop_prob: float,
                        days_in_state: int, ma20_dist_pct: float,
                        momentum_pct: float, bear_trend: str,
                        ivr_data: dict) -> dict:

    hvr             = ivr_data.get("hvr") or 50
    premium_quality = ivr_data.get("premium_quality", "unknown")
    vol_trend       = ivr_data.get("vol_trend", "stable")

    # ── MA20 Recovery Override ──
    # Bear regime but BTC clearly recovering above MA20 with strong momentum
    ma20_recovering = (ma20_dist_pct >= MA20_RECOVERY_PCT and
                       momentum_pct >= MOMENTUM_RECOVERY and
                       bear_prob < BEAR_SOFTENING and
                       bear_trend == "improving")

    # ── IVR context string ──
    if premium_quality == "cheap":
        ivr_note = f"⚠️ HVR {hvr:.0f} — premium cheap, reduce size"
    elif premium_quality in ("good", "excellent"):
        ivr_note = f"✅ HVR {hvr:.0f} — premium elevated, good selling opportunity"
    elif premium_quality == "extreme":
        ivr_note = f"🚨 HVR {hvr:.0f} — extreme premium, covered calls only"
    else:
        ivr_note = f"HVR {hvr:.0f} — normal premium"

    # ── Decision tree ──

    # BEAR + recovering strongly (MA20 override)
    if state == "Bear" and ma20_recovering:
        if hvr >= IVR_SWEET_SPOT:
            signal     = "YELLOW"
            action     = "Bear regime but BTC recovering — sell put spread or far OTM put"
            strike_adj = "2–3 strikes below max pain (put spread preferred)"
            caution    = f"Bear prob {bear_prob:.0%} improving — not fully safe yet. {ivr_note}"
            covered_call = "✅ Also consider covered call 3–5% OTM to lock in elevated premium"
        else:
            signal     = "YELLOW"
            action     = "Bear recovering but premium thin — hold existing, no new puts yet"
            strike_adj = "Wait for HVR > 50 before selling"
            caution    = f"Bear prob {bear_prob:.0%} improving but {ivr_note}"
            covered_call = "🟡 Covered call marginal — premium too low"

    # BEAR + established + high IVR → covered calls shine
    elif state == "Bear" and days_in_state > 5:
        if hvr >= IVR_HIGH:
            signal     = "ORANGE"
            action     = "No new puts — sell covered calls to monetize Bear premium"
            strike_adj = "Covered call 3–5% OTM on MSTR shares held"
            caution    = f"Bear day {days_in_state} — max pain broken. {ivr_note}"
            covered_call = f"✅ BEST BEAR PLAY: Sell covered call — HVR {hvr:.0f} gives excellent call premium"
        elif hvr >= IVR_SWEET_SPOT:
            signal     = "ORANGE"
            action     = "No new puts — covered call viable if you hold shares"
            strike_adj = "Covered call 4–6% OTM"
            caution    = f"Bear day {days_in_state}. {ivr_note}"
            covered_call = f"🟡 Covered call viable — HVR {hvr:.0f} is decent"
        else:
            signal     = "RED"
            action     = "Close all short puts — premium too thin to sell calls"
            strike_adj = "No new positions"
            caution    = f"Bear day {days_in_state} and {ivr_note}"
            covered_call = "❌ Skip — premium not worth the risk"

    # BEAR + fresh
    elif state == "Bear" and days_in_state <= 5:
        signal     = "ORANGE"
        action     = "Fresh Bear — manage existing positions, no new puts"
        strike_adj = "Close or roll existing positions down"
        caution    = f"Fresh Bear regime day {days_in_state} — likely more downside. {ivr_note}"
        covered_call = "🟡 Covered call possible but wait 2–3 days to confirm Bear"

    # CHOP + bear prob falling + good IVR → sweet spot
    elif state == "Chop" and bear_prob < 0.35 and bear_trend == "improving":
        if hvr >= IVR_SWEET_SPOT:
            signal     = "YELLOW"
            action     = "Regime improving — sell put spread at elevated premium"
            strike_adj = "2 strikes below max pain — put spread preferred over naked put"
            caution    = f"Chop transitioning out of Bear — {ivr_note}"
            covered_call = "🟡 Covered call still viable as secondary strategy"
        else:
            signal     = "YELLOW"
            action     = "Regime improving but wait for better premium"
            strike_adj = "2 strikes below max pain, reduced size"
            caution    = f"Chop with low premium — {ivr_note}"
            covered_call = "❌ Skip covered call — premium not worth it"

    # CHOP + bear prob still elevated
    elif state == "Chop" and bear_prob >= 0.35:
        signal     = "ORANGE"
        action     = "Chop with elevated Bear risk — far OTM puts only"
        strike_adj = "3+ strikes below max pain"
        caution    = f"Bear prob {bear_prob:.0%} still high. {ivr_note}"
        covered_call = "🟡 Covered call viable if HVR > 60"

    # CHOP + normal
    elif state == "Chop" and bear_prob < 0.25:
        signal     = "YELLOW"
        action     = "Chop — sell puts 2 strikes below max pain"
        strike_adj = "2 strikes below max pain"
        caution    = f"Magnet unreliable in Chop. {ivr_note}"
        covered_call = "❌ Skip covered call in Chop — wait for Bull"

    # BULL + strong + good premium
    elif state == "Bull" and bear_prob < 0.15:
        if hvr >= IVR_SWEET_SPOT:
            signal     = "GREEN"
            action     = "Safe to sell puts at max pain — elevated premium"
            strike_adj = "At max pain strike"
            caution    = f"Monitor if Bear prob rises above 20%. {ivr_note}"
            covered_call = "❌ Skip covered call in Bull — you want upside exposure"
        elif hvr < IVR_LOW:
            signal     = "GREEN"
            action     = "Safe but premium thin — reduce size or skip"
            strike_adj = "At max pain but half normal size"
            caution    = f"Bull regime safe but {ivr_note}"
            covered_call = "❌ Skip — too cheap"
        else:
            signal     = "GREEN"
            action     = "Safe to sell puts at max pain"
            strike_adj = "At max pain strike"
            caution    = f"{ivr_note}"
            covered_call = "❌ Skip covered call — upside open in Bull"

    # BULL + weakening
    elif state == "Bull" and bear_prob >= 0.15:
        signal     = "YELLOW"
        action     = "Bull weakening — sell 1 strike below max pain"
        strike_adj = "1 strike below max pain"
        caution    = f"Bull losing momentum. {ivr_note}"
        covered_call = "🟡 Consider covered call as hedge against Bull → Chop flip"

    else:
        signal     = "YELLOW"
        action     = "Mixed signals — reduce size"
        strike_adj = "2 strikes below max pain"
        caution    = ivr_note
        covered_call = "🟡 Evaluate based on regime direction"

    return {
        "signal"      : signal,
        "action"      : action,
        "strike_adj"  : strike_adj,
        "caution"     : caution,
        "covered_call": covered_call,
        "ivr_note"    : ivr_note,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    import os
    os.makedirs("data", exist_ok=True)

    print("📥 Fetching BTC-USD data from Yahoo Finance...")
    df = fetch_data(TICKER, LOOKBACK_DAYS)
    print(f"   {len(df)} rows fetched")

    print("📊 Fetching MSTR data for IVR calculation...")
    mstr_df  = fetch_mstr_for_ivr(days=365)
    ivr_data = calculate_hvr(mstr_df)
    if ivr_data["hvr"] is not None:
        print(f"   MSTR HVR: {ivr_data['hvr']:.1f} — {ivr_data['ivr_label']}")
        print(f"   30d HV: {ivr_data['hv_30d']}%  |  Vol trend: {ivr_data['vol_trend']}")
    else:
        print("   ⚠️ Could not calculate HVR")

    print("🔧 Engineering features (Option A — drop first 20 rows)...")
    X, df_feat, scaler, feature_cols = build_features(df)
    print(f"   Training on {len(X)} days of data")

    print(f"🧠 Training 3-state HMM ({N_INIT} random seeds)...")
    model = train_hmm(X)
    print(f"   Best log-likelihood: {model.score(X):.2f}")

    print("🏷️  Labelling states (Bull / Chop / Bear)...")
    state_map, raw_states = label_states(model, X, df_feat)
    print(f"   State map: {state_map}")

    print("📊 Computing outputs...")
    output = compute_outputs(model, X, df_feat, state_map, raw_states, ivr_data)

    print(f"💾 Writing to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    rec = output["recommendation"]
    print(f"\n✅ Done — Current regime: {output['current_state']} "
          f"(Bull {output['bull_probability']:.0%} / "
          f"Chop {output['chop_probability']:.0%} / "
          f"Bear {output['bear_probability']:.0%})")
    print(f"   Bear trend: {output['bear_prob_trend']}")
    print(f"   BTC vs MA20: {output['btc_ma20_distance_pct']:+.1f}%  |  "
          f"3d momentum: {output['momentum_3d_pct']:+.1f}%")
    print(f"   Signal: {rec['signal']} — {rec['action']}")
    print(f"   Covered call: {rec['covered_call']}")
    print(f"   Regime alert: {output['regime_change_alert']}")


if __name__ == "__main__":
    main()
