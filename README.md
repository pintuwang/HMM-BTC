# BTC 3-State HMM Regime Tracker

Detects Bitcoin market regime (Bull / Chop / Bear) using a Gaussian Hidden Markov Model trained on Yahoo Finance daily data. Designed to complement the MSTR/BTC max pain options strategy.

## How It Works

1. Fetches 2 years of BTC-USD daily OHLCV from Yahoo Finance
2. Engineers 5 features: daily return, realized vol, MA20 distance, 3-day momentum, volume ratio
3. Trains a 3-state Gaussian HMM with 20 random seeds (keeps best)
4. Labels states as Bull / Chop / Bear by mean return
5. Outputs probabilities + options recommendation to `data/hmm_output.json`
6. GitHub Pages renders the dashboard from that JSON

## Setup

### 1. Clone and install locally
```bash
git clone https://github.com/pintuwang/HMM-BTC
cd HMM-BTC
pip install -r requirements.txt
```

### 2. Run manually
```bash
python generate_hmm.py
```
This creates `data/hmm_output.json`.

### 3. View dashboard locally
Open `index.html` in a browser (use a local server for fetch to work):
```bash
python -m http.server 8000
# then open http://localhost:8000
```

### 4. Deploy to GitHub Pages
- Push repo to GitHub
- Go to Settings → Pages → Source: main branch / root
- GitHub Actions will auto-update data twice daily (Mon–Fri)

## Update Schedule (GitHub Actions)
- **08:00 ET** — Pre-market update (overnight BTC move captured)
- **16:30 ET** — Post-close update (fresh options OI + official close)

## States and Options Strategy

| State | Bear Prob | Signal | Action |
|-------|-----------|--------|--------|
| Bull  | < 15%     | GREEN  | Sell puts at max pain |
| Bull  | 15–30%    | YELLOW | Sell puts 1 strike below pain |
| Chop  | < 25%     | YELLOW | Sell puts 2 strikes below pain |
| Chop  | ≥ 25%     | ORANGE | Far OTM only |
| Bear  | Any       | RED    | No new puts / close positions |

## Files
```
├── generate_hmm.py          # Main HMM script
├── index.html               # Dashboard
├── requirements.txt         # Python dependencies
├── data/
│   └── hmm_output.json      # Auto-generated output (do not edit manually)
└── .github/
    └── workflows/
        └── update.yml       # GitHub Actions schedule
```
