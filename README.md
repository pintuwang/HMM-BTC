## PENDING VALIDATION
Date noted: April 14, 2026
Hypothesis: BTC spot vs BTC pain is PRIMARY predictor of whether
MSTR closes within $3 of max pain (stronger signal than HMM regime).
Evidence so far: 4/4 expiries confirmed (small sample — needs more data)
Re-validate: Week of May 12, 2026 (4 weeks from now)
Add 4 new expiry dates: Apr 17, Apr 24, May 1, May 8
Feed to Claude with screenshots of Chart 3, Chart 4, and HMM chart.



# BTC HMM · IVR · 5-Strategy Scorer
### A regime-aware options strategy dashboard for MSTR short put and covered call decisions

Live dashboard: **https://pintuwang.github.io/hmm-btc/**

---

# PART 1 — HOW TO USE (Simple Guide)

## What This Page Does

This dashboard answers one question every trading day:

> **"Given what BTC is doing right now, which options strategy on MSTR gives me the best risk-adjusted premium?"**

It combines three signals into one unified view:
- **BTC market regime** (is BTC in a Bull, Chop, or Bear phase?)
- **MSTR options premium quality** (is IV expensive or cheap right now?)
- **5 strategy scores** (which of 5 specific approaches fits today's conditions best?)

It also **backtests** each strategy against the last 90 days so you can see historical win rates — not just theoretical recommendations.

---

## The Simple 3-Step Workflow

### Step 1 — Check the Traffic Light Signal
At the top of the page you will see a coloured signal:

| Signal | Colour | What To Do |
|--------|--------|------------|
| **GREEN** | 🟢 | Safe to sell puts near max pain — conditions ideal |
| **YELLOW** | 🟡 | Sell with caution — go 1–2 strikes below max pain |
| **ORANGE** | 🟠 | No new puts — manage existing or sell covered calls only |
| **RED** | 🔴 | Close or roll existing short puts immediately |

This is your **go/no-go** decision for the day. If it says ORANGE or RED, do not open new short put positions regardless of how attractive the premium looks.

---

### Step 2 — Check the Top-Ranked Strategy (★ BEST TODAY)
Below the signal, the **5-Strategy Scorer** ranks all 5 approaches for today's conditions. The one marked **★ BEST TODAY** is the highest-scoring strategy given the current regime + IV environment.

Each strategy has a **% score** showing its relative suitability. For example:
```
A3: Covered Call (Bear)     68%  ★ BEST TODAY
A1: Chop Transition         14%
A4: Put Spread (Transition)  9%
A2: IVR Filter               6%
A5: Calendar Premium         3%
```
This means today is a Bear regime day — selling covered calls on your MSTR shares is the most appropriate action, not selling puts.

---

### Step 3 — Check the Backtest Win Rate
Below the scorer is the **Backtest table** showing historical win rates for each strategy over the past 90 days. This tells you which strategies have actually worked recently on MSTR — not just which look good in theory.

Use the win rate to size your confidence:
- **Win rate > 65%** → high conviction, normal position size
- **Win rate 50–65%** → moderate conviction, reduce size by 25%
- **Win rate < 50%** → low conviction, skip or go very small

---

## Daily Routine (takes 2 minutes)

**Before market open (8 AM ET / 8 PM SGT):**
1. Open https://pintuwang.github.io/hmm-btc/
2. Read the traffic light — GREEN/YELLOW = can trade, ORANGE/RED = wait
3. Note the ★ BEST TODAY strategy
4. Check HVR gauge — is premium elevated (above 50)?
5. Make your decision

**After market close (4:30 PM ET / 4:30 AM SGT):**
- Page auto-updates with new data
- Check if regime changed (bear trend sparkbars show direction)
- Adjust any open positions if signal changed

---

## Quick Reference — What Each Signal Means for Your MSTR Positions

| Situation | Signal | Action |
|-----------|--------|--------|
| Bull regime, HVR 50–70 | 🟢 GREEN | Sell puts at max pain |
| Chop, Bear prob falling | 🟡 YELLOW | Sell put spread 2 strikes below pain |
| Bear, HVR > 70, hold shares | 🟠 ORANGE | Sell covered calls instead |
| Bear, fresh (day 1–5) | 🟠 ORANGE | Do nothing, manage existing |
| Bear, established (day 6+) | 🔴 RED | Close short puts, sell covered calls |
| Bear recovering (BTC +5% above MA20) | 🟡 YELLOW | Put spread, not naked put |

---

---

# PART 2 — DEEP DIVE (How It Works)

## Architecture Overview

The system has 4 layers that feed into each other:

```
Yahoo Finance API
│
├── BTC-USD daily OHLCV (2 years)
│   └── Feature Engineering (5 inputs)
│       └── 3-State HMM → Bull / Chop / Bear + probabilities
│
├── MSTR daily OHLCV (1 year)
│   └── Rolling 30-day realized vol
│       └── Historical Vol Rank (HVR) = IVR proxy
│
└── Combined → 5-Approach Scorer
                └── Backtest Engine (90 days history)
                    └── JSON output → GitHub Pages dashboard
```

Data updates **twice daily on weekdays** via GitHub Actions:
- **12:00 UTC (8 AM ET)** — pre-market update captures overnight BTC moves
- **20:30 UTC (4:30 PM ET)** — post-close update uses official daily closes

---

## Section 1 — BTC 3-State HMM Regime

### What It Is
A **Hidden Markov Model (HMM)** is a statistical model that detects hidden states from observable data. In this case:
- The **hidden states** are market regimes: Bull, Chop, Bear
- The **observable data** are daily BTC price features

The model learns which combinations of features tend to cluster together, then classifies each day into the most probable regime.

### Data Source
- Ticker: `BTC-USD` via Yahoo Finance
- Frequency: Daily OHLCV
- Training window: **730 days (2 years)** rolling

### The 5 Input Features

| Feature | Calculation | What It Captures |
|---------|-------------|------------------|
| **Daily Return** | `(close - prev_close) / prev_close` | Direction and magnitude of price move |
| **7-Day Realized Vol** | Rolling 7-day std of returns × √252 | Short-term volatility level (annualized) |
| **MA20 Distance** | `(close - 20d MA) / 20d MA × 100` | Trend structure — positive = above trend |
| **3-Day Momentum** | `(close - close[3d ago]) / close[3d ago]` | Short-term directional momentum (funding rate proxy) |
| **Volume Ratio** | `today volume / 20d avg volume` | Conviction behind the move |

All features are normalized using StandardScaler before being fed to the HMM so no single feature dominates.

### Model Specification

| Parameter | Value | Reason |
|-----------|-------|--------|
| n_components | 3 | Bull / Chop / Bear |
| covariance_type | full | Features are correlated — Bear has different vol structure than Bull |
| n_iter | 200 | Enough EM iterations to converge |
| n_init | 20 | 20 random seeds, best log-likelihood kept — avoids local optima |
| Training data | Last 730 days | Covers at least one full BTC cycle |

### State Labelling
After training, states are labelled by **mean daily return** of each cluster:
- Highest mean return → **Bull**
- Middle mean return → **Chop**
- Lowest mean return → **Bear**

This ensures consistent labelling across monthly retrains (raw state numbers may swap but economic labels stay stable).

### Output Per Day
The model outputs **3 probabilities** (not just one state):
```json
{
  "current_state": "Bear",
  "bull_probability": 0.08,
  "chop_probability": 0.14,
  "bear_probability": 0.78
}
```
A Bear at 78% confidence is very different from Bear at 45% with Chop at 40%. Both show "Bear" but the second is far less certain — the probabilities give you the full picture.

### Bear Trend Indicator
The 5-day mini sparkbar chart shows whether Bear probability is **improving (↓)**, **deteriorating (↑)**, or **stable (→)**. This is critical for the MA20 override and for approach scoring — a Bear at 78% that was 90% yesterday is very different from one that was 60% yesterday.

### Transition Matrix
Shows the probability of moving between regimes day-to-day. A typical MSTR/BTC matrix looks like:
```
              → Bull   → Chop   → Bear
From Bull  [  0.85     0.13     0.02  ]
From Chop  [  0.15     0.70     0.15  ]
From Bear  [  0.05     0.20     0.75  ]
```
Key insight: **Bull→Bear direct is rare (2%)** — there is almost always a Chop phase in between. This means the Chop state is your early warning before a Bear regime establishes.

### MA20 Override
The recommendation engine softens from ORANGE → YELLOW when all 4 conditions are simultaneously true:
1. BTC is more than **4% above its 20-day MA**
2. 3-day momentum is above **+5%**
3. Bear probability is below **55%**
4. Bear trend is **improving**

This handles the scenario where the HMM is still showing Bear (due to model inertia) but price has clearly started recovering — the most relevant case being the Apr 11 2026 BTC bounce from $66K to $73K.

---

## Section 2 — MSTR Historical Volatility Rank (HVR)

### What It Is and Why Not True IVR
True **IV Rank (IVR)** requires historical implied volatility data — what the options market was pricing on each past day. Yahoo Finance does not store this. Instead, the system uses **Historical Volatility Rank (HVR)** as a proxy.

HVR measures where **current 30-day realized volatility** sits within its own 1-year range:

```
HVR = (Current 30d HV  −  52-week lowest 30d HV)
      ÷
      (52-week highest 30d HV  −  52-week lowest 30d HV)
      × 100
```

For MSTR specifically, realized vol and implied vol are highly correlated (both driven by BTC moves) so HVR captures 85–90% of what true IVR would tell you.

### Data Source
- Ticker: `MSTR` via Yahoo Finance
- Frequency: Daily close prices
- Window: **252 trading days (1 year)** for the rank calculation
- Calculated rolling (no lookahead bias in backtest)

### HVR Zones and Meaning

| HVR Range | Zone | Meaning | Recommended Action |
|-----------|------|---------|-------------------|
| 0–30 | Cheap | Premium unusually low | Skip selling — not worth the risk |
| 30–50 | Below Average | Normal-ish premium | Sell only in confirmed Bull regime |
| 50–70 | Elevated (Sweet Spot) | Good premium, manageable risk | Sell puts in Chop/Bull transition |
| 70–85 | High | Excellent premium | Sell puts (Bull/Chop) OR covered calls (Bear) |
| 85–100 | Extreme | Maximum premium, maximum risk | Covered calls only — puts too dangerous |

### The IVR Problem This Solves
The fundamental tension in regime-based options trading: the safest time to sell puts (Bull regime) has the lowest premium because IV has already collapsed. The highest premium occurs during Bear regimes when selling puts is most dangerous.

HVR tracks where you are in that cycle so you can:
1. **Avoid selling when premium is too cheap** (HVR < 30)
2. **Identify the sweet spot** — Chop transition with HVR 50–70 (elevated from Bear but regime improving)
3. **Redirect to covered calls** when HVR is extreme (Bear + HVR > 70)

### Vol Trend
The 10-day HV vs 30-day HV comparison tells you if premium is expanding or contracting:
- **Rising** (10d HV > 30d HV × 1.15) → vol expanding, put spreads preferred over naked puts
- **Falling** (10d HV < 30d HV × 0.85) → vol contracting (IV crush starting), sell now before premium drops
- **Stable** → premium holding steady

---

## Section 3 — The 5 Strategies Explained

### A1 — Chop Transition (Sell Put at Declining Bear Prob)

**Core Idea:** Don't wait for full Bull confirmation. The Chop regime with Bear probability actively falling is the premium sweet spot — IV is still elevated from the Bear period but directional risk is clearly reducing.

**Ideal Conditions:**
- Bear probability: falling from >60% down through 30–50%
- Regime: transitioning to Chop or early Chop
- HVR: 50–70 (still juicy from Bear period)

**Entry Signal:**
```
Bear prob: 78% → 60% → 45% → 30%
                         ↑
                    SELL HERE
```

**Scoring Inputs:**
- Size of Bear prob drop over 5 days (up to 40 points)
- Bear prob currently in 25–55% range (25 points)
- Regime in Chop or fading Bear (20 points)
- HVR in sweet spot (15 points)

**Strike Selection:** 2 strikes below current max pain — extra buffer since regime not fully confirmed.

**Why It Works:** By entering during the transition (not waiting for Bull), you capture the residual IV premium from the Bear regime while directional risk has already started normalising. Waiting for full Bull confirmation means IV has crushed and premiums are 40–60% lower.

---

### A2 — IVR Filter (Sell When HVR in Sweet Spot)

**Core Idea:** Use HVR as the primary filter — sell options when premium is objectively elevated, regardless of regime signals. The IVR zone determines whether the premium justifies the risk.

**Ideal Conditions:**
- HVR: 50–70 (ideal), 70–85 (good but add regime check)
- Regime: Bull or Chop with low Bear risk
- Vol trend: stable or falling (premium holding or about to compress)

**The IVR Table:**
```
HVR 0–30:    Skip entirely — premium doesn't justify delta risk
HVR 30–50:   Bull regime only — premium acceptable only with full safety
HVR 50–70:   Ideal zone — sell in Bull or Chop
HVR 70–85:   Excellent premium — verify regime before selling puts
HVR 85–100:  Covered calls only — put risk too high at this IV level
```

**Scoring Inputs:**
- HVR zone position (up to 50 points)
- Regime compatibility (up to 30 points)
- Vol trend (up to 20 points)
- Note: capped at 25 in Bear regime regardless of HVR

**Why It Works:** Prevents selling cheap premium (low HVR in Bull) and prevents dangerous put selling at extreme premium levels (high HVR in Bear). Filters out the worst entries in both directions.

---

### A3 — Covered Call in Bear (Monetize High IV Without Directional Risk)

**Core Idea:** During Bear regime you hold MSTR shares but cannot safely sell puts. Instead, sell covered calls — collect the elevated IV premium without additional downside exposure. This pre-funds the lower-premium put trades in Bull regime.

**Requires:** Holding 100+ MSTR shares per contract.

**Ideal Conditions:**
- Regime: Bear (established, day 5+)
- HVR: above 70 (excellent call premium)
- Bear probability: above 60% (downside confirmation)

**Mechanics:**
```
Bear regime:
  Own 100 MSTR shares
  Sell 1x covered call 3–5% OTM (e.g. MSTR at $120, sell $126C)
  Collect $4–6 premium
  If MSTR stays below $126 → keep premium + shares ✅
  If MSTR spikes above $126 → shares called away at $126 (miss upside)
```

**Strike Selection:** 3–5% OTM — wide enough to survive MSTR's typical weekly noise but close enough for meaningful premium. Avoid selling too close (1–2% OTM) — MSTR can spike 5–8% in a day.

**Scoring Inputs:**
- In Bear regime (40 points)
- Days in Bear ≥ 5 (15 additional points — established Bear = calls safer)
- HVR > 70 (30 points)
- Bear prob > 60% (15 points)

**The Wheel Effect:** Over a full Bull/Bear cycle:
- Bear phases: collect $4–6/week in call premium
- Bull phases: collect $2–3/week in put premium
- Combined: $6–9/week vs $2–3/week from puts alone

---

### A4 — Put Spread in Transition (Protect Against IV Crush)

**Core Idea:** When transitioning from Bear to Bull, the vol crush (IV falling rapidly) damages naked puts faster than spreads. Selling a put spread (sell upper strike, buy lower strike) maintains better value during IV compression because the bought put loses value faster than the sold put.

**Ideal Conditions:**
- Vol trend: falling (IV crush underway)
- Regime: Bear→Chop or early Chop
- Bear probability: 20–50%

**Mechanics:**
```
Instead of: Sell $125P for $3.00
            Max loss = $125 per share

Use spread: Sell $125P for $3.00
            Buy  $115P for $1.50
            Net credit = $1.50
            Max loss = $10 (spread width) vs $125 naked
```

**Why Spread Holds Value Better:**
During IV crush, the shorter-dated/lower-delta bought put loses value faster than the sold put. The spread's value is therefore more stable — you don't lose as much when IV falls 20 points vs a naked put.

**Scoring Inputs:**
- Regime transitioning (35 points)
- Vol trend falling (30 points)
- Bear prob 20–50% (20 points)
- In Chop regime (15 points)
- Penalised in Bull (spread unnecessary when regime is clear)

**When NOT to Use:** Confirmed Bull with HVR < 50 — naked put is better because the spread's width reduction isn't worth it when you have full directional safety.

---

### A5 — Calendar Premium (Sell Longer DTE Post-Bear)

**Core Idea:** IV mean-reverts slower in longer-dated options than in weeklies. After a Bear regime ends, the 30–45 DTE options still carry elevated IV for 1–2 weeks while the weeklies have already crushed. Sell the 30–45 DTE put, then close after 10–14 days when you've captured 50–60% of the premium.

**Ideal Conditions:**
- Bear regime just ended (now in Chop, was Bear within last 7 days)
- HVR still elevated (50+)
- Vol trend: stable (premium not yet compressing rapidly)

**Mechanics:**
```
Bear ends → MSTR at $120 → HVR still 65
Sell May $110P (30 DTE) for $5.00  (vs weekly $100P for $1.50)
Wait 10–14 days
Close at $2.50 (50% of premium captured)
Time in trade: 10–14 days vs full 30 days
Annualised return: higher than holding to expiry
```

**Why It Works:** At the end of a Bear regime, market makers still have elevated vol priced into longer dates because they're uncertain if Bear is truly over. This mis-pricing is the edge — you're selling IV that is higher than what the regime transition justifies.

**Scoring Inputs:**
- Recently exited Bear (35 points)
- HVR ≥ 50 (25 points)
- Vol trend stable (20 points) — if vol falling, close sooner
- Bear prob falling and in 20–55% range (20 points)

**Risk:** If Bear resumes, you have 30–45 days of exposure vs weekly's 5–7 days. Always use a stop: close if Bear prob returns above 60%.

---

## Section 4 — Backtest Engine

### What It Tests
For each of the last **90 trading days**, the system:
1. Calculates the regime state, probabilities, HVR, and vol trend **using only past data** (no lookahead bias)
2. Scores all 5 approaches as if it were that day
3. Looks **5 trading days forward** at MSTR closing price
4. Determines if each strategy would have been successful

### Win Condition Per Strategy

| Strategy | Win Condition | Rationale |
|----------|---------------|-----------|
| A1 Chop Transition | MSTR up in 5 days | Put expires OTM |
| A2 IVR Filter | MSTR up in 5 days | Put expires OTM |
| A3 Covered Call | MSTR flat or down in 5 days | Call expires OTM |
| A4 Put Spread | MSTR up in 5 days | Upper put expires OTM |
| A5 Calendar | MSTR up in 5 days | Put expires OTM |

### Skip Logic
A day is **skipped** (not counted in win/loss) if the approach scores below 40 — meaning it was not applicable that day. This prevents counting days where the approach was correctly identified as unsuitable.

### Limitations to Be Aware Of
- Uses MSTR price direction as proxy for put profitability — doesn't model exact premium P&L
- 5-day forward window assumes weekly expiry horizon — monthly positions not captured
- 90-day backtest window may be short for statistical significance (especially for A5 which is rare)
- Does not account for bid/ask spread or assignment risk on weekends
- Win rate is a directional win rate — a 1% up move counts the same as a 20% up move

### How to Interpret the Results
Sample output:
```
A3 Covered Call     WR=72%  avg ret=+3.2%  n=18
A1 Chop Transition  WR=68%  avg ret=+2.8%  n=22
A2 IVR Filter       WR=61%  avg ret=+1.9%  n=31
A4 Put Spread       WR=55%  avg ret=+1.1%  n=19
A5 Calendar         WR=50%  avg ret=+0.8%  n=8
```
A3 winning 72% of 18 cases is meaningful. A5 winning 50% of 8 cases is not statistically reliable — use it with caution until sample size grows.

---

## Section 5 — GitHub Actions Automation

The system uses **GitHub Actions** (free, runs in the cloud) to update data twice daily without any manual intervention.

### Workflow File: `.github/workflows/update.yml`

```yaml
Schedule:
  - 12:00 UTC (8 AM ET)  → pre-market update
  - 20:30 UTC (4:30 PM ET) → post-close update
  - Weekdays only (Mon–Fri)
  - Manual trigger available in GitHub Actions tab

Steps each run:
  1. Checkout repository code
  2. Install Python 3.11
  3. Install: yfinance, hmmlearn, scikit-learn, numpy, pandas
  4. Run generate_hmm.py
  5. Commit updated data/hmm_output.json back to repo
  6. GitHub Pages auto-serves the updated JSON to the dashboard
```

### Manual Update
Go to **github.com/pintuwang/hmm-btc → Actions → Update BTC HMM Regime → Run workflow** to force an immediate refresh at any time.

---

## Data Flow Summary

```
1. Yahoo Finance downloads BTC-USD + MSTR daily prices
2. Features calculated: return, realized_vol, MA20_distance, momentum_3d, vol_ratio
3. HMM trained on BTC features → state sequence + daily probabilities
4. MSTR 30d rolling vol calculated → HVR (IVR proxy)
5. 5-approach scorer runs for today's readings
6. Backtest scorer runs over last 90 days
7. All results written to data/hmm_output.json
8. index.html reads JSON on page load → renders dashboard
```

---

## Suggested Future Features

Based on the architecture above, these additions are feasible and would improve signal quality:

1. **MSTR max pain integration** — pull MSTR options OI from Unusual Whales API to calculate max pain and magnet score per expiry, combining with regime for a full combined traffic light
2. **VIX HMM overlay** — combine this BTC HMM with the existing VIX9D HMM (pintuwang.github.io/HMM) into one unified signal
3. **Expiry-aware approach scoring** — adjust strategy scores based on days to next expiry (A5 Calendar only makes sense with >20 DTE available)
4. **Position tracker** — input your current strike and expiry, get a roll recommendation based on live regime + price
5. **BTC pain level** — overlay BTC max pain from Deribit options chain to see if BTC is near its own institutional magnet
6. **Alert system** — GitHub Actions sends a Telegram or email alert when regime changes or Bear prob crosses 30%
7. **Monthly retraining automation** — add a monthly GitHub Actions trigger to retrain the HMM on the latest 2-year window

---

## File Structure

```
hmm-btc/
├── generate_hmm.py          # Main script — HMM + IVR + scorer + backtest
├── index.html               # Dashboard — reads JSON and renders all panels
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── data/
│   └── hmm_output.json      # Auto-generated — do not edit manually
└── .github/
    └── workflows/
        └── update.yml       # GitHub Actions schedule
```

---

## Quick Setup (for new machine)

```bash
# 1. Clone repo
git clone https://github.com/pintuwang/hmm-btc
cd hmm-btc

# 2. Install dependencies (use Anaconda Prompt on Windows)
pip install yfinance hmmlearn scikit-learn numpy pandas

# 3. Generate initial data
python generate_hmm.py

# 4. Push data file
git add data/hmm_output.json
git commit -m "data: update HMM output"
git push

# 5. View dashboard
# Open https://pintuwang.github.io/hmm-btc/
# (or locally: python -m http.server 8000 then open localhost:8000)
```

---

*Not financial advice. Past backtest performance does not guarantee future results.*
*Data: Yahoo Finance (BTC-USD, MSTR). Updated pre-market and post-close, weekdays.*
