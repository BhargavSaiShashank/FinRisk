import yfinance as yf
import pandas as pd
import numpy as np
import os

def download_data(tickers, start_date, end_date):
    """
    Downloads historical OHLC data from Yahoo Finance.
    Handles multiple tickers and returns a combined DataFrame.
    """
    data = yf.download(tickers, start=start_date, end=end_date)
    return data

def process_data(raw_data):
    """
    Extracts 'Adj Close' and computes log returns.
    """
    # yfinance MultiIndex columns: Level 0 = Price Type, Level 1 = Ticker
    if 'Adj Close' in raw_data.columns.get_level_values(0):
        adj_close = raw_data['Adj Close']
    elif 'Close' in raw_data.columns.get_level_values(0):
        # Fallback to 'Close' if 'Adj Close' is missing (e.g. for some indices)
        adj_close = raw_data['Close']
    else:
        # Single ticker case
        adj_close = raw_data[['Adj Close']] if 'Adj Close' in raw_data.columns else raw_data[['Close']]
    
    # Fill missing values (forward fill then backward fill)
    adj_close = adj_close.ffill().bfill()
    
    # Log Returns: log(Pt / Pt-1)
    log_returns = np.log(adj_close / adj_close.shift(1)).dropna()
    
    return adj_close, log_returns

def save_processed_data(df, filename):
    folder = os.path.join("data", "processed")
    os.makedirs(folder, exist_ok=True)
    df.to_csv(os.path.join(folder, filename))
    print(f"Saved processed data to {os.path.join(folder, filename)}")

if __name__ == "__main__":
    # Primary assets for multi-asset reinforcement
    indices = ["^STOXX50E", "^GSPC", "^IXIC", "^N225"]
    
    # Explanatory variables
    others = ["^FCHI", "^GDAXI", "^AEX", "FEZ", "EURUSD=X", "EURGBP=X"]
    
    # Macroeconomic variables
    # VIX: ^VIX, US 10Y: ^TNX, Oil: CL=F, Gold: GC=F
    macro = ["^VIX", "^TNX", "CL=F", "GC=F"]
    
    all_tickers = list(set(indices + others + macro))
    
    start = "2008-09-01"
    end = "2025-03-31"
    
    print(f"Downloading data for {len(all_tickers)} tickers...")
    raw_data = download_data(all_tickers, start, end)
    
    adj_close, log_returns = process_data(raw_data)
    
    save_processed_data(adj_close, "adj_close.csv")
    save_processed_data(log_returns, "log_returns.csv")
