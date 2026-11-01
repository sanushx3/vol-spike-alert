import os

import yfinance as yf

os.makedirs("data", exist_ok=True)

ticker = "^NSEI"
start_date = "2007-01-01"  
end_date = "2026-11-01" 

# Download daily historical data
data = yf.download(ticker, start=start_date, end=end_date, interval="1d")

# Save to CSV
output_path = "data/nifty50_raw.csv"
data.to_csv(output_path)

# Print confirmation details
rows = len(data)
first_date = data.index.min().strftime("%Y-%m-%d")
last_date = data.index.max().strftime("%Y-%m-%d")

print(f"Successfully saved {rows} rows to {output_path}")
print(f"Date range: {first_date} to {last_date}")