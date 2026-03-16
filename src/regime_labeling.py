import pandas as pd
import numpy as np

def label_market_regimes(returns, vix=None, window=252):
    """
    Labels historical market regimes (0-3) based on volatility, returns, and VIX.
    Regime 0: Calm
    Regime 1: Rising Vol
    Regime 2: Crash
    Regime 3: Recovery
    """
    df = pd.DataFrame(index=returns.index)
    df['returns'] = returns
    df['vol'] = returns.rolling(window=window).std()
    
    # 1. Volatility Percentiles
    vol_30 = df['vol'].expanding().quantile(0.3)
    vol_70 = df['vol'].expanding().quantile(0.7)
    
    # 2. Return Threshold (Crash)
    ret_mean = returns.expanding().mean()
    ret_std = returns.expanding().std()
    crash_thresh = ret_mean - 2 * ret_std
    
    regimes = np.zeros(len(df))
    
    for i in range(1, len(df)):
        # Default based on volatility
        if df['vol'].iloc[i] <= vol_30.iloc[i]:
            regimes[i] = 0 # Calm
        elif df['vol'].iloc[i] <= vol_70.iloc[i]:
            regimes[i] = 1 # Rising Vol
        else:
            regimes[i] = 1 # High Vol (default to 1 if not crash)
            
        # 3. Crash Detection
        # Extreme negative return or VIX spike
        is_crash = (df['returns'].iloc[i] < crash_thresh.iloc[i])
        if vix is not None:
             is_crash = is_crash or (vix.iloc[i] > 30)
             
        if is_crash:
            regimes[i] = 2 # Crash
            
        # 4. Recovery Detection
        # If previous was crash and current returns are positive
        if regimes[i-1] == 2 and df['returns'].iloc[i] > 0:
            regimes[i] = 3 # Initial recovery
        elif regimes[i-1] == 3 and df['returns'].iloc[i] > -0.005: 
            # Continue recovery if not another crash
            regimes[i] = 3
            
    return pd.Series(regimes, index=returns.index, name='regime')

def plot_regimes(returns, regimes, path='results/plots/regime_timeline.png'):
    import matplotlib.pyplot as plt
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    plt.figure(figsize=(15, 7))
    colors = ['green', 'yellow', 'red', 'blue']
    labels = ['Calm', 'Rising Vol', 'Crash', 'Recovery']
    
    for r in range(4):
        mask = (regimes == r)
        plt.scatter(returns.index[mask], returns[mask], color=colors[r], label=labels[r], alpha=0.5, s=10)
        
    plt.title('Market Regimes Timeline')
    plt.legend()
    plt.savefig(path)
    plt.close()
