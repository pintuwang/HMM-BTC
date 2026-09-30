# MSTR Option-Premium Monitor — Strategy Guide

*How the IV/HV logger works, what it is for, and how to use it.*
*Last updated: 2026-09-30*

---

## 1. The idea in one paragraph

This system does **not** try to predict where MSTR goes next. Two earlier attempts to do that (max pain in BTC2, BTC regimes in HMM-BTC) found no reliable directional edge once their bugs were fixed. What is left is a simpler, measurable question:

> **Are MSTR options priced richer than MSTR actually moves?**

If they are, selling options (covered calls, cash-secured puts) earns that overpricing over time. If they are not, selling options just sells MSTR's upside (or buys its crashes) too cheaply. The logger records the answer every trading day; after enough days, the average tells you whether selling premium on MSTR is worth doing at all, and at which strikes.

---

## 2. How we got here — what was tested and what it showed

| Tool | Question it asked | Result |
|---|---|---|
| **BTC2 max-pain dashboard** | Does BTC vs its max pain predict MSTR vs its max pain? | Directionally consistent *at expiry*, but contemporaneous — pain lags price, so it isn't known at entry. Chart bug fixed (it graded against post-expiry pain). **Descriptive, not predictive.** |
| **HMM-BTC regime scorer** | Do Bull/Chop/Bear regimes pick the right option strategy? | After fixing lookahead bias and adding baselines, **no strategy beat its no-signal baseline** beyond noise (edges swing ±4–8 pts between refits). |
| **Covered-call P&L backtest** (Colab, Jun 2022 → Sep 2026) | Do weekly covered calls beat simply holding MSTR? | **Depends entirely on how richly options are priced** — see below. |

### The backtest result that drives everything

Weekly covered calls, sold every week, compounded return vs buy & hold (+614%):

| Assumed IV ÷ realized vol (`IV_MULT`) | 10% OTM | 15% OTM | 20% OTM |
|---|---|---|---|
| 1.00 (options priced at realized vol) | +262% ❌ | +301% ❌ | +616% ≈ |
| 1.15 | +973% ✅ | +823% ✅ | +1,054% ✅ |
| **Break-even** | **≈ 1.09** | **≈ 1.10** | **≈ 1.00** |

The verdict flips on a ~10% difference in option pricing. The backtest cannot say which side of break-even reality is on — **only real option prices can.** That is what the logger measures.

Cross-check: the MSTY ETF (a real covered-call fund on MSTR since Feb 2024) roughly matched MSTR's total return to mid-2026 — consistent with a ratio near break-even, and inconsistent with options being hugely overpriced.

---

## 3. Key concepts

### HV — historical (realized) volatility: *what MSTR actually did*
Standard deviation of the last 30 daily returns, annualised by √252.
`hv30 = 1.05` means MSTR has been swinging at 105%/year ≈ ±6.6% per day.

### IV — implied volatility: *what the option market is charging for*
The one volatility that makes Black-Scholes reproduce an option's market price. The logger solves for it from the **bid** (what a seller actually receives), not Yahoo's IV field (unreliable on weeklies).

### The ratio — the single number that matters
```
iv_hv_ratio_bid = IV at the bid ÷ HV30
```
This is exactly the backtest's `IV_MULT`.

**Worked example (2026-09-28, 20% OTM call):** MSTR $157.14 → Oct 2 $190 call, bid $0.20.
IV that prices it at $0.20 = 98.1%. HV30 = 105.1%. Ratio = **0.93** → you are paid for 93% of the swinginess MSTR has actually shown. Below the 20% OTM break-even of 1.00 → **selling underpaid**.

### Skew
Each strike has its own IV. On MSTR, far OTM calls carry higher IV than at-the-money (e.g. 70% ATM → 90% at 25% OTM). That is why every OTM level gets its own ratio, and why calls and puts must be judged separately.

---

## 4. The decision rules

### Covered calls (you hold 100 shares)

Use the **20-day average** of `iv_hv_ratio_bid` for **live** readings (`mean_iv_hv_ratio_bid_live_only` in `iv_summary.json`), never a single day.

| Call-side 20-day average | Action |
|---|---|
| 10–15% OTM ratio **above ~1.10** | Closer strikes are justified — more premium, and the backtest says it beat holding |
| 20% OTM ratio **at or above ~1.00** | Sell 20% OTM weekly calls — roughly matches holding plus steadier income |
| 20% OTM ratio **clearly below 1.00** | **Don't sell calls for income** — you'd be selling upside too cheaply |

**The exception (no edge required):** if you would genuinely be happy to sell your shares at a price, a call at that strike is a *paid limit order*. The ratio then only tells you whether the payment is fair, not whether to do it.

### Cash-secured puts

**No rule yet.** Put ratios are being logged, but there is no put break-even — selling puts risks buying MSTR in a crash rather than giving up upside, so it needs its own backtest (planned, see §8). **Do not apply the call break-evens to puts.**

### What *not* to use for the decision
- **HMM regime label / ★ BEST TODAY ranking** — no measurable edge; labels flip with model refits.
- **Max pain level** — lags price.
- **A single day's ratio** — noisy; wait for the 20-day average.

Still useful as context: **HVR** (is premium rich or cheap vs the past year) and the **7-day regime forecast** (a reminder that a regime has ~1-in-3 odds of flipping before a weekly expires).

---

## 5. How the logger runs

### Files (all in `data/`)

| File | Contents |
|---|---|
| `iv_log_calls.csv` | OTM **calls**, strike **above** spot — covered-call side |
| `iv_log_puts.csv` | OTM **puts**, strike **below** spot — cash-secured-put side |
| `iv_summary.json` | Latest ratios + averages, in separate `"calls"` and `"puts"` sections |

Code: `iv_logger.py` (repo root) · Schedule: `.github/workflows/iv_logger.yml`

### What it records each trading day
- The weekly expiry closest to **7 calendar days** out (at least 3 days away).
- For calls and puts at **10 / 15 / 20 / 25% OTM**: the first listed strike at/beyond the target, its bid/ask, IV (from bid and mid), and the ratio.
- An at-the-money IV and ratio for each side as a reference.

### Timing and sessions
- GitHub delays scheduled runs by hours, so the workflow fires **every hour 12:07–19:07 UTC** on weekdays.
- Each row is tagged `session`: `live` (9:35–15:55 New York), `pre_market`, or `after_close`.
- **Pre-market runs exit immediately.** The **first live run of the day wins**; later runs that day skip.
- An `after_close` row is kept only as a fallback and is replaced if a live row arrives the same day.
- Rows are dated by the **US trading day**, not UTC.
- If no strike has a bid (pre-market, holidays), **nothing is written**.

**Live window in Singapore time:** 9:35 pm – 3:55 am SGT (US summer time); **10:35 pm – 4:55 am SGT after 1 Nov 2026**.

---

## 6. How to use it

### Daily — nothing
It runs on its own. Optionally glance at `iv_summary.json`.

### Weekly (optional, ~2 minutes)
1. Open `data/iv_summary.json` → `calls.levels.20pct_OTM`.
2. Check `n_live` is growing by ~5 per week (if not, the schedule is failing — see §9).
3. Note `mean_iv_hv_ratio_bid_live_only` against the break-even.

### Before selling a covered call
1. Read the **20% OTM live-only average** (and 10–15% if you're considering closer strikes).
2. Apply the table in §4.
3. If it says "don't sell", only sell at a strike you'd happily exit at.
4. In your broker, check the actual bid on the strike — the logger's reading may be a day old.

### At milestones
Send the repo link to Claude and ask for the corresponding analysis (§8).

---

## 7. Column reference (both CSVs)

| Column | Meaning |
|---|---|
| `date` | US trading date (New York) |
| `session` | `live` / `pre_market` / `after_close` — **analyse `live` rows** |
| `run_utc` | When the run actually happened |
| `spot` | MSTR price at run time |
| `expiry`, `dte` | Option expiry and calendar days to it |
| `hv30`, `hv10` | 30- and 10-day realized vol (annualised, decimal) |
| `atm_strike`, `atm_iv`, `atm_iv_hv_ratio` | At-the-money reference for this option type |
| `option_type` | `call` or `put` |
| `otm_target` | 0.10 / 0.15 / 0.20 / 0.25 |
| `strike`, `otm_actual` | Listed strike used and its true distance from spot (always positive) |
| `bid`, `ask`, `mid`, `spread_pct` | Quote; `spread_pct` = (ask − bid) / mid |
| `open_interest`, `volume` | Liquidity |
| `iv_yahoo` | Yahoo's IV field — for comparison only, unreliable |
| `iv_bid`, `iv_mid` | IV solved from bid / mid |
| `iv_hv_ratio_bid` | **The decision number** (seller's view) |
| `iv_hv_ratio_mid` | Same from mid (fair-value view) |
| `low_price` | `True` if mid < $0.20 — tick size distorts the IV |
| `note` | e.g. `no two-sided quote` |

**Treat as unreliable:** `after_close` rows with `spread_pct` above ~0.5 (post-close quotes are often stale or one-sided), and `low_price = True` rows.

---

## 8. Roadmap

| When | Test | Question answered |
|---|---|---|
| **~20 live days** (late Oct 2026) | IV/HV ratio distribution per strike vs break-evens | Is the backtest's key assumption true in real prices? |
| **2–3 months** | Implied vol vs vol MSTR **then realized** over each option's life | True variance-premium test — did IV overpay for what actually happened? |
| **After a few weeks of put data** | Cash-secured-put version of the P&L backtest | Put break-evens, so puts get a decision rule |
| **6 months+** | Real-price P&L: each logged bid as premium received, settled at the later spot | Would these trades actually have beaten holding? |

**Sample-size reality:** 3 months ≈ 60 trading days but only ~12 independent weekly trades. Ratio tests become informative first; P&L tests need 6+ months, and still cover one market regime.

---

## 9. Caveats and maintenance

- **Trailing HV is not the future.** A ratio below 1 right after a volatility spike partly reflects HV30 still being inflated by the spike. The forward-realized test (§8) fixes this.
- **Black-Scholes assumes smooth moves.** MSTR has fat tails (+20–30% weeks). Some apparent "overpricing" of far OTM options is fair payment for those tails.
- **DTE drifts** between ~4 and ~10 days depending on the weekday; the analysis should account for it.
- **Spot is intraday**, not the official close — fine for averages, noisy for single days.
- **Holidays** aren't modelled; no bids → nothing written.
- **If `n_live` stops growing:** check the Actions tab for failures. For exact timing, trigger the workflow from an external scheduler (e.g. cron-job.org) via GitHub's `workflow_dispatch` API with a fine-grained token limited to this repo and *Actions: read & write* only.
- **US clocks change 1 Nov 2026** — the hourly schedule covers both summer and winter market hours; no change needed.

---

*Not financial advice. Research tool for personal use.*
