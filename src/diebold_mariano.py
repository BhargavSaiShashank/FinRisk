import numpy as np
from scipy.stats import norm

def dm_test(actual, forecast1, forecast2, h=1, crit='MSE'):
    """
    Diebold-Mariano test for forecast accuracy comparison.
    H0: forecast1 and forecast2 have equal accuracy.
    """
    e1 = actual - forecast1
    e2 = actual - forecast2
    
    if crit == 'MSE':
        d = e1**2 - e2**2
    elif crit == 'MAE':
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("Criterion must be MSE or MAE")
    
    d_mean = np.mean(d)
    n = len(d)
    
    # Variance estimation (Newey-West style for h > 1, but usually h=1 for VaR)
    gamma0 = np.var(d)
    var_d = gamma0 / n
    
    if h > 1:
        for k in range(1, h):
            gamma_k = np.sum((d[k:] - d_mean) * (d[:-k] - d_mean)) / n
            var_d += 2 * gamma_k / n
            
    dm_stat = d_mean / np.sqrt(max(var_d, 1e-12))
    p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value
