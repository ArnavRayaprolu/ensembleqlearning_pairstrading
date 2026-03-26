import yfinance as yf
import pandas as pd
import numpy as np

# Define NIFTY100 tickers for Yahoo Finance 
nifty50_tickers = [
    'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS',
    'BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BPCL.NS', 'BHARTIARTL.NS',
    'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS',
    'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS',
    'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS',
    'INFY.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LTIM.NS',
    'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS',
    'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS',
    'SHREECEM.NS', 'SUNPHARMA.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS',
    'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS'
]

# Set the date range for hourly data
start_date = "2024-01-01"
end_date = "2025-01-01"

print("Fetching hourly data (interval=60m). Please wait...")

# Download hourly data in one batch
df = yf.download(
    tickers=nifty50_tickers,
    start=start_date,
    end=end_date,
    interval="60m",
    group_by="ticker",
    auto_adjust=True,
    threads=True,
    progress=True
)

# Extract 'Close' prices into a flat DataFrame
close_prices = {}
for ticker in nifty50_tickers:
    try:
        symbol = ticker.replace(".NS", "")
        close_prices[symbol] = df[ticker]['Close']
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Combine Close prices
combined_df = pd.DataFrame(close_prices)

# Fill missing values forward and backward
combined_df.ffill(inplace=True)
combined_df.bfill(inplace=True)

# Save to CSV
combined_df.to_csv("nifty50_hourly_prices.csv")
print("Saved to 'nifty50_hourly_prices.csv'")
