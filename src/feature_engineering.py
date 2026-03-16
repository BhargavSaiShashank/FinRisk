import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def compute_technical_indicators(df, price_col='^STOXX50E'):
    """
    Computes SMA, EMA, RSI, and Bollinger Bands using pure pandas.
    """
    series = df[price_col]
    
    # SMA (5, 15)
    df['SMA5'] = series.rolling(window=5).mean()
    df['SMA15'] = series.rolling(window=15).mean()
    
    # EMA (5, 15)
    df['EMA5'] = series.ewm(span=5, adjust=False).mean()
    df['EMA15'] = series.ewm(span=15, adjust=False).mean()
    
    # RSI (14)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    # Handle division by zero
    loss = loss.replace(0, 1e-9)
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20, 2)
    sma20 = series.rolling(window=20).mean()
    std20 = series.rolling(window=20).std()
    df['BB_Mid'] = sma20
    df['BB_Upper'] = sma20 + (std20 * 2)
    df['BB_Lower'] = sma20 - (std20 * 2)
    
    return df

def create_feature_matrix(log_returns, adj_close, primary_asset='^STOXX50E'):
    """
    Enhanced feature matrix including lags (1, 5, 10).
    """
    features = compute_technical_indicators(adj_close.copy())
    
    for col in log_returns.columns:
        # returns lags: 1, 5, 10
        features[f'{col}_lag1'] = log_returns[col].shift(1)
        features[f'{col}_lag5'] = log_returns[col].shift(5)
        features[f'{col}_lag10'] = log_returns[col].shift(10)

    final_df = pd.concat([features, log_returns.rename(columns=lambda x: f"{x}_return" if x == primary_asset else x)], axis=1)
    # Drop rows with NaN due to lagging and indicators (at least 20 for BB/SMA15/Lag10)
    final_df = final_df.dropna()
    
    return final_df

def normalize_features(df, exclude_cols=['^STOXX50E']):
    """
    Normalizes features using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    cols_to_scale = [c for c in df.columns if c not in exclude_cols]
    
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    return df_scaled, scaler
