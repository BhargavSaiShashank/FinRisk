import yfinance as yf
import pandas as pd
import numpy as np
import os

def download_data(tickers, start_date, end_date):
    """
    Download historical daily data for given tickers.
    """
    data = yf.download(tickers, start=start_date, end=end_date, interval="1d")
    return data

def process_data(data):
    """
    Extract closing prices and compute log returns for the primary asset.
    """
    # yfinance multi-index handling
    if 'Adj Close' in data.columns:
        adj_close = data['Adj Close']
    elif isinstance(data.columns, pd.MultiIndex) and 'Adj Close' in data.columns.get_level_values(0):
        adj_close = data['Adj Close']
    else:
        # Fallback to Close if Adj Close is missing
        adj_close = data['Close']
    
    # Fill missing values
    adj_close = adj_close.ffill().dropna()
    
    # Compute log returns
    log_returns = np.log(adj_close / adj_close.shift(1)).dropna()
    
    return adj_close, log_returns

def save_processed_data(df, filename):
    """
    Save processed dataframe to data/processed directory.
    """
    os.makedirs('data/processed', exist_ok=True)
    path = os.path.join('data/processed', filename)
    df.to_csv(path)
    print(f"Saved processed data to {path}")

if __name__ == "__main__":
    primary_asset = "^STOXX50E"
    # Equity indices: CAC40 (^FCHI), DAX (^GDAXI), AEX (^AEX)
    # ETF: FEZ
    # Currencies: EURUSD (EURUSD=X), EURGBP (EURGBP=X)
    others = ["^FCHI", "^GDAXI", "^AEX", "FEZ", "EURUSD=X", "EURGBP=X"]
    all_tickers = [primary_asset] + others
    
    start = "2008-09-01"
    end = "2025-03-31"
    
    print(f"Downloading data for {all_tickers}...")
    raw_data = download_data(all_tickers, start, end)
    
    adj_close, log_returns = process_data(raw_data)
    
    save_processed_data(adj_close, "adj_close.csv")
    save_processed_data(log_returns, "log_returns.csv")
