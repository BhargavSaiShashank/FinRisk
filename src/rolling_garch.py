import pandas as pd
import numpy as np
from arch import arch_model
from tqdm import tqdm

def rolling_garch_estimation(returns, window=1000, model_type='GARCH'):
    """
    Performs rolling window volatility estimation.
    For each t, fits model on [t-window, t-1] and forecasts for t.
    """
    n = len(returns)
    forecasts = np.full(n, np.nan)
    
    print(f"Running rolling {model_type} estimation (window={window})...")
    
    # Rescale returns to % for stability
    r_scaled = returns * 100
    
    for i in tqdm(range(window, n)):
        train_window = r_scaled.iloc[i-window:i]
        
        try:
            if model_type == 'GJR-GARCH':
                am = arch_model(train_window, vol='Garch', p=1, o=1, q=1, dist='studentst')
            else:
                am = arch_model(train_window, vol='Garch', p=1, q=1, dist='studentst')
            
            res = am.fit(disp='off', show_warning=False)
            # Forecast 1 step ahead
            forecast = res.forecast(horizon=1).variance.iloc[-1, 0]
            forecasts[i] = np.sqrt(forecast) / 100
        except:
            # Fallback to previous forecast or global vol if fit fails
            forecasts[i] = forecasts[i-1] if i > window else np.nan
            
    return pd.Series(forecasts, index=returns.index)

def generate_rolling_vars(returns, window=1000):
    """
    Generates rolling VaR for both GARCH and GJR-GARCH.
    """
    vol_garch = rolling_garch_estimation(returns, window=window, model_type='GARCH')
    vol_gjr = rolling_garch_estimation(returns, window=window, model_type='GJR-GARCH')
    
    # For parametric VaR, we usually use constant z-score or sample dist (nu)
    # Here we use fixed nu=5 for student-t or estimate from last window
    from scipy.stats import t
    z95 = t.ppf(0.05, df=5)
    z99 = t.ppf(0.01, df=5)
    
    vars_df = pd.DataFrame(index=returns.index)
    vars_df['VaR_GARCH_5%'] = -(vol_garch * z95)
    vars_df['VaR_GARCH_1%'] = -(vol_garch * z99)
    vars_df['VaR_GJR_GARCH_5%'] = -(vol_gjr * z95)
    vars_df['VaR_GJR_GARCH_1%'] = -(vol_gjr * z99)
    
    return vars_df, vol_garch, vol_gjr
