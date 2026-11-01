# VolSpike Alert
A machine learning model that predicts whether tomorrow's Nifty 50 price range will spike or not, helping beginner traders to avoid taking large risks on volatile days.

## Demo
Coming in Week 4.

## Problem Statement and Impact
Beginner traders in the Indian Stock Market often take large positions without any awareness if a volatility spike will occur, resulting in a huge loss. This project builds a classifier that predicts whether the next trading day's price range on the Nifty 50 will be at least 1.5 times its 20-day average (a "volatility spike"), using past price data and scheduled-event features such as RBI policy dates and the Union Budget. Success means catching a high share of real spikes (recall) with a reasonable false-alarm rate, and beating a simple "tomorrow looks like today" baseline. It is a risk-awareness tool for beginners, not a trading strategy or financial advice.

## Dataset

- **Source:** Yahoo Finance, downloaded with the `yfinance` Python library (script: `src/download_data.py`)
- **Instrument:** Nifty 50 index, ticker `^NSEI`, daily interval
- **Date range:** 17 September 2007 to 30 October 2026
- **Size:** 4,691 rows (one row per trading day)
- **Downloaded on:** 02/11/2026
- **Raw file:** `data/nifty50_raw.csv`
- **Columns:** Date, Open, High, Low, Close, Volume
- **Target (planned):** a "volatility spike" is a day whose range (High minus Low) is at least 1.5 times its trailing 20-day average. The 1.5x threshold is provisional and will be confirmed after checking how often spikes occur.
- **Additional data (planned):** a hand-built calendar of scheduled events (RBI policy dates, Union Budget days, US Fed meetings).

**Data quality notes**
- The raw CSV has three header rows (Price, Ticker, Date) because of how yfinance exports its output, so it needs special handling when loaded.
- **Volume is 0 for roughly the first 1,300 rows.** 
- Only trading days appear, so weekends and market holidays are absent.
- Yahoo data can be revised later, so results use this fixed download.
- Prices downloaded with yfinance defaults (`auto_adjust=True`); no effect for an index.
## Approach
## Results
## Key Insights
## Challenges Faced
## How to Run Locally
## Limitations and Future Improvements
## Tech Stack and Contact
