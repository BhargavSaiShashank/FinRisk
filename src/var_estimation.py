import numpy as np
import scipy.stats as stats
import pandas as pd

def compute_parametric_var(mu, sigma, alpha=0.05, dist='studentst', nu=None):
    if dist == 'normal':
        z = stats.norm.ppf(alpha)
    elif dist == 'studentst':
        if nu is None: nu = 5 
        z = stats.t.ppf(alpha, df=nu)
    else:
        raise ValueError("Unsupported distribution")
    
    return -(mu + sigma * z)

def generate_all_vars(returns, res_garch, res_gjr):
    """
    Generates VaR at 1% and 5% for both GARCH and GJR-GARCH.
    """
    mu = 0 # Daily returns mean approximation
    
    vol_garch = res_garch.conditional_volatility / 100
    vol_gjr = res_gjr.conditional_volatility / 100
    
    nu_garch = res_garch.params['nu']
    nu_gjr = res_gjr.params['nu']
    
    vars_df = pd.DataFrame(index=returns.index)
    
    vars_df['VaR_GARCH_5%'] = compute_parametric_var(mu, vol_garch, alpha=0.05, nu=nu_garch)
    vars_df['VaR_GARCH_1%'] = compute_parametric_var(mu, vol_garch, alpha=0.01, nu=nu_garch)
    vars_df['VaR_GJR_GARCH_5%'] = compute_parametric_var(mu, vol_gjr, alpha=0.05, nu=nu_gjr)
    vars_df['VaR_GJR_GARCH_1%'] = compute_parametric_var(mu, vol_gjr, alpha=0.01, nu=nu_gjr)
    
    return vars_df

def calculate_risk_labels(returns, var_estimates):
    """
    Labels: 1 if return <= c, else 0.
    c = max{ r_k+1 | r_k+1 < -VaR_k+1 }
    """
    # Align indices
    common_idx = returns.index.intersection(var_estimates.index)
    r = returns.loc[common_idx]
    v = var_estimates.loc[common_idx]
    
    # Violations: r < -v (loss exceeds VaR)
    violations = r[r < -v]
    
    if len(violations) > 0:
        c = violations.max()
    else:
        c = -v.mean()
        
    labels = (r <= c).astype(int)
    return labels, c
