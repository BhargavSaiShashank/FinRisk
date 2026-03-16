import numpy as np
import pandas as pd
from scipy import stats

def kupiec_pof_test(returns, var_estimates, alpha=0.05):
    """
    Likelihood ratio test for Unconditional Coverage.
    """
    violations = (returns < -var_estimates).astype(int)
    num_violations = violations.sum()
    n = len(returns)
    p = alpha
    
    if num_violations == 0:
        return 1.0, 0.0
        
    p_hat = num_violations / n
    p_hat = max(min(p_hat, 1 - 1e-9), 1e-9)
    p = max(min(p, 1 - 1e-9), 1e-9)
    
    lr = -2 * ( (n - num_violations) * np.log(1 - p) + num_violations * np.log(p) - 
                ( (n - num_violations) * np.log(1 - p_hat) + num_violations * np.log(p_hat) ) )
    
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return p_value, lr

def christoffersen_ind_test(returns, var_estimates):
    """
    Likelihood ratio test for Independence (Conditional Coverage).
    """
    violations = (returns < -var_estimates).astype(int)
    n = len(violations)
    
    n00, n01, n10, n11 = 0, 0, 0, 0
    for i in range(1, n):
        if violations.iloc[i-1] == 0:
            if violations.iloc[i] == 0: n00 += 1
            else: n01 += 1
        else:
            if violations.iloc[i] == 0: n10 += 1
            else: n11 += 1
            
    if (n00 + n01) == 0 or (n10 + n11) == 0:
        return 1.0, 0.0
        
    p01 = n01 / (n00 + n01)
    p11 = n11 / (n10 + n11)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    
    p01 = max(min(p01, 1 - 1e-9), 1e-9)
    p11 = max(min(p11, 1 - 1e-9), 1e-9)
    p = max(min(p, 1 - 1e-9), 1e-9)
    
    lr = -2 * ( (n00 + n10) * np.log(1 - p) + (n01 + n11) * np.log(p) - 
                ( n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11) + n11 * np.log(p11) ) )
                
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return p_value, lr

def acerbi_szekely_test(returns, var_estimates, es_estimates):
    """
    Acerbi-Székely test for ES backtesting.
    Z-stat: 1/N * sum(I_t * r_t / ES_t) + 1
    N: number of violations.
    """
    violations = (returns < -var_estimates)
    n_violations = violations.sum()
    
    if n_violations == 0:
        return 0.0, 1.0 # Stat 0, P-value 1 (safe)
        
    # Standardize exceedances
    # Note: returns are positive for gains, so losses are -r
    # We want mean(-r_t / ES_t) where -r_t > VaR_t
    exceedances = -returns[violations]
    es_v = es_estimates[violations]
    
    # Avoid division by zero
    es_v = es_v.replace(0, 1e-9)
    
    z_stat = (1 / n_violations) * np.sum(exceedances / es_v) - 1
    
    # Simple p-value approximation (one-tailed)
    # Under H0, Z should be around 0.
    # We estimate p-value using a normal distribution for simplicity or bootstrap.
    # Here we report 1 if Z >= 0 else low.
    p_value = 1 - stats.norm.cdf(z_stat) if z_stat > 0 else 0.5 # Dummy p-value logic for now
    
    return z_stat, p_value

def basel_traffic_light(returns, var_estimates):
    """
    Basel traffic light system based on 250-day window.
    Green: 0-4, Yellow: 5-9, Red: 10+
    """
    # Use last 250 days if available
    window = min(len(returns), 250)
    violations = (returns.iloc[-window:] < -var_estimates.iloc[-window:]).sum()
    
    if violations <= 4:
        return "Green", violations
    elif violations <= 9:
        return "Yellow", violations
    else:
        return "Red", violations

def adjust_var_rl(var, predicted_risk, b1=0.30, b2=0.20):
    if predicted_risk == 0:
        return var * (1 - b1)
    else:
        return var * (1 + b2)
