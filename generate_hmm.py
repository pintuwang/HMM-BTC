"""
BTC 3-State HMM Regime Detector
================================
Fetches BTC-USD daily OHLCV from Yahoo Finance,
trains a 3-state Gaussian HMM (Bull / Chop / Bear),
and writes results to data/hmm_output.json for the dashboard.

Update schedule: twice daily via GitHub Actions
  - 08:00 ET (pre-market)
  - 16:30 ET (post-close)

Author: auto-generated for pintuwang/HMM-BTC
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
TICKER        = "BTC-USD"
LOOKBACK_DAYS = 730          # 2 years of training data
N_STATES      = 3
N_ITER        = 200
N_INIT        = 20           # run 20 random seeds, keep best
OUTPUT_PATH   = "data/hmm_output.json"
STATE_LABELS  = ["Bull", "Chop", "Bear"]   # re-mapped after training by mean return


# ─────────────────────────────────────────────
# STEP 1 — FETCH DATA
# ─────────────────────────────────────────────
def fetch_data(ticker: str, days: int) -> dict:
    """Download daily OHLCV from Yahoo Finance."""
    end   = datetime.utcnow()
    start = end - timedelta(days=days + 30)   # extra 30 days for MA warmup
    df    = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
# STEP 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
def build_features(df) -> tuple:
    """
    Compute 5 features from daily OHLCV.
    Drop first 20 rows (Option A) to avoid NaN from MA/vol lookback.

    Returns:
        features_scaled : np.ndarray  shape (n, 5)  — model input
        feature_df      : DataFrame                 — for inspection / JSON export
        scaler          : fitted StandardScaler
    """
    import pandas as pd

    df = df.copy()

    # Feature 1 — Daily return
    df["return"] = df["Close"].pct_change()

    # Feature 2 — 7-day realized volatility (annualized)
    df["realized_vol_7d"] = df["return"].rolling(7).std() * np.sqrt(252)

    # Feature 3 — Distance from 20-day MA (normalized %)
    df["ma20"]          = df["Close"].rolling(20).mean()
    df["ma20_distance"] = (df["Close"] - df["ma20"]) / df["ma20"] * 100

    # Feature 4 — 3-day momentum (funding rate proxy)
    df["momentum_3d"] = df["Close"].pct_change(3)

    # Feature 5 — Volume ratio vs 20-day average
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # Option A — drop first 20 rows (NaN from rolling windows)
    df.dropna(inplace=True)

    # Keep only the last LOOKBACK_DAYS rows for training
    df = df.tail(LOOKBACK_DAYS)

    feature_cols = ["return", "realized_vol_7d", "ma20_distance", "momentum_3d", "vol_ratio"]
    X_raw = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_scaled, df, scaler, feature_cols


# ─────────────────────────────────────────────
# STEP 3 — TRAIN HMM
# ─────────────────────────────────────────────
def train_hmm(X: np.ndarray) -> GaussianHMM:
    """
    Train GaussianHMM with N_INIT random seeds.
    Returns the model with the highest log-likelihood.
    """
    best_model  = None
    best_score  = -np.inf

    for seed in range(N_INIT):
        model = GaussianHMM(
            n_components      = N_STATES,
            covariance_type   = "full",
            n_iter            = N_ITER,
            random_state      = seed,
            tol               = 1e-4,
        )
        try:
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score  = score
                best_model  = model
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("HMM failed to converge on all seeds")

    return best_model


# ─────────────────────────────────────────────
# STEP 4 — LABEL STATES
# ─────────────────────────────────────────────
def label_states(model: GaussianHMM, X: np.ndarray, df) -> dict:
    """
    Predict state sequence and label Bull/Chop/Bear
    by sorting states on mean daily return (feature index 0).

    Returns mapping: raw_state_int → "Bull"/"Chop"/"Bear"
    """
    raw_states = model.predict(X)

    # Mean return per raw state (feature 0 = return, but we use unscaled df)
    mean_returns = {}
    for s in range(N_STATES):
        mask = raw_states == s
        if mask.sum() > 0:
            mean_returns[s] = df["return"].values[mask].mean()
        else:
            mean_returns[s] = 0.0

    # Sort: highest return → Bull(0), middle → Chop(1), lowest → Bear(2)
    sorted_states = sorted(mean_returns, key=mean_returns.get, reverse=True)
    state_map = {raw: label for raw, label in zip(sorted_states, STATE_LABELS)}

    return state_map, raw_states


# ─────────────────────────────────────────────
# STEP 5 — COMPUTE OUTPUTS
# ─────────────────────────────────────────────
def compute_outputs(model: GaussianHMM, X: np.ndarray,
                    df, state_map: dict, raw_states: np.ndarray) -> dict:
    """
    Build the full output JSON for the dashboard.
    """
    # Posterior probabilities for the full sequence
    log_posteriors = model.predict_proba(X)   # shape (n, 3)

    # Latest day
    latest_probs_raw  = log_posteriors[-1]   # raw state ordering
    latest_state_raw  = raw_states[-1]
    latest_state_name = state_map[latest_state_raw]

    # Re-map probabilities to Bull/Chop/Bear ordering
    bull_state = [k for k, v in state_map.items() if v == "Bull"][0]
    chop_state = [k for k, v in state_map.items() if v == "Chop"][0]
    bear_state = [k for k, v in state_map.items() if v == "Bear"][0]

    bull_prob = float(latest_probs_raw[bull_state])
    chop_prob = float(latest_probs_raw[chop_state])
    bear_prob = float(latest_probs_raw[bear_state])

    state_confidence = float(max(bull_prob, chop_prob, bear_prob))

    # Days in current state (consecutive streak)
    days_in_state = 0
    for s in reversed(raw_states):
        if s == latest_state_raw:
            days_in_state += 1
        else:
            break

    # Regime change alert: bear_prob crossed above 30% from previous day
    prev_probs_raw    = log_posteriors[-2] if len(log_posteriors) > 1 else latest_probs_raw
    prev_bear_prob    = float(prev_probs_raw[bear_state])
    regime_change_alert = (bear_prob >= 0.30 and prev_bear_prob < 0.30)
    state_changed       = (raw_states[-1] != raw_states[-2]) if len(raw_states) > 1 else False

    # Historical state sequence (last 90 days for chart)
    history = []
    tail = min(90, len(df))
    dates = df.index[-tail:]
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

    # Transition matrix (re-mapped to Bull/Chop/Bear order)
    raw_trans = model.transmat_
    ordered   = [bull_state, chop_state, bear_state]
    trans_matrix = []
    for i in ordered:
        row = []
        for j in ordered:
            row.append(round(float(raw_trans[i][j]), 4))
        trans_matrix.append(row)

    # Current BTC price info
    latest_close     = float(df["Close"].iloc[-1])
    latest_ma20      = float(df["ma20"].iloc[-1])
    latest_vol_7d    = float(df["realized_vol_7d"].iloc[-1])
    latest_momentum  = float(df["momentum_3d"].iloc[-1])
    latest_date      = df.index[-1].strftime("%Y-%m-%d")

    # Options strategy recommendation
    recommendation = get_recommendation(latest_state_name, bear_prob, chop_prob, days_in_state)

    output = {
        "generated_at"        : datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "latest_date"         : latest_date,
        "btc_close"           : round(latest_close, 2),
        "btc_ma20"            : round(latest_ma20, 2),
        "btc_ma20_distance_pct": round((latest_close - latest_ma20) / latest_ma20 * 100, 2),
        "realized_vol_7d"     : round(latest_vol_7d * 100, 2),   # as percentage
        "momentum_3d_pct"     : round(latest_momentum * 100, 2),
        "current_state"       : latest_state_name,
        "bull_probability"    : round(bull_prob, 4),
        "chop_probability"    : round(chop_prob, 4),
        "bear_probability"    : round(bear_prob, 4),
        "state_confidence"    : round(state_confidence, 4),
        "days_in_current_state": int(days_in_state),
        "regime_change_alert" : bool(regime_change_alert or state_changed),
        "transition_matrix"   : trans_matrix,
        "history"             : history,
        "recommendation"      : recommendation,
    }

    return output


# ─────────────────────────────────────────────
# STEP 6 — OPTIONS RECOMMENDATION ENGINE
# ─────────────────────────────────────────────
def get_recommendation(state: str, bear_prob: float,
                        chop_prob: float, days_in_state: int) -> dict:
    """
    Translate regime into actionable options guidance.
    """
    if state == "Bull" and bear_prob < 0.15:
        signal     = "GREEN"
        action     = "Safe to sell puts at or near max pain"
        strike_adj = "At max pain strike"
        caution    = "Monitor if Bear prob rises above 20%"

    elif state == "Bull" and bear_prob >= 0.15:
        signal     = "YELLOW"
        action     = "Sell puts but 1 strike below max pain"
        strike_adj = "1 strike below max pain"
        caution    = "Bull weakening — tighten stops"

    elif state == "Chop" and bear_prob < 0.25:
        signal     = "YELLOW"
        action     = "Reduce size, sell puts 2 strikes below pain"
        strike_adj = "2 strikes below max pain"
        caution    = "Magnet unreliable in Chop — watch BTC $MA20"

    elif state == "Chop" and bear_prob >= 0.25:
        signal     = "ORANGE"
        action     = "Minimal exposure only — far OTM puts if any"
        strike_adj = "3+ strikes below max pain"
        caution    = f"Bear prob at {bear_prob:.0%} — regime flip risk"

    elif state == "Bear" and days_in_state <= 5:
        signal     = "ORANGE"
        action     = "No new short puts — manage existing positions"
        strike_adj = "Close or roll existing positions down"
        caution    = "Fresh Bear regime — likely more downside"

    else:  # Bear, established
        signal     = "RED"
        action     = "Close all short puts immediately"
        strike_adj = "No new positions"
        caution    = f"Bear regime day {days_in_state} — max pain magnet broken"

    return {
        "signal"    : signal,
        "action"    : action,
        "strike_adj": strike_adj,
        "caution"   : caution,
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
    output = compute_outputs(model, X, df_feat, state_map, raw_states)

    print(f"💾 Writing to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Done — Current regime: {output['current_state']} "
          f"(Bull {output['bull_probability']:.0%} / "
          f"Chop {output['chop_probability']:.0%} / "
          f"Bear {output['bear_probability']:.0%})")
    print(f"   Signal: {output['recommendation']['signal']} — {output['recommendation']['action']}")
    print(f"   Regime alert: {output['regime_change_alert']}")


if __name__ == "__main__":
    main()
