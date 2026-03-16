import pandas as pd
import numpy as np
from arch import arch_model
import warnings

def fit_garch_model(returns, model_type='GARCH', p=1, q=1, dist='studentst'):
    """
    Fits a GARCH or GJR-GARCH model.
    rescale=True is often needed for returns in % to help convergence.
    """
    if model_type == 'GJR-GARCH':
        # GJR-GARCH is Garch with o=1 (asymmetric effect)
        am = arch_model(returns, vol='Garch', p=p, o=1, q=q, dist=dist)
    else:
        am = arch_model(returns, vol='Garch', p=p, q=q, dist=dist)
    
    res = am.fit(disp='off')
    return res

def get_volatility_features(returns):
    """
    Fits both models and returns conditional volatilities as a DataFrame.
    """
    # Rescale returns to % for stability
    r_scaled = returns * 100
    
    res_garch = fit_garch_model(r_scaled, model_type='GARCH')
    res_gjr = fit_garch_model(r_scaled, model_type='GJR-GARCH')
    
    vol_df = pd.DataFrame({
        'vol_GARCH': res_garch.conditional_volatility / 100,
        'vol_GJR': res_gjr.conditional_volatility / 100
    }, index=returns.index)
    
    return vol_df, res_garch, res_gjr
