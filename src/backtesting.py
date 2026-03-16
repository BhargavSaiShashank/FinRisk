import numpy as np
import pandas as pd
from scipy import stats

def kupiec_test(returns, var_estimates, alpha=0.05):
    """
    Kupiec's POF (Proportion of Failures) test.
    H0: The model provides an accurate VaR estimate.
    """
    violations = (returns < -var_estimates).astype(int)
    num_violations = violations.sum()
    n = len(returns)
    p = alpha
    
    if num_violations == 0:
        return 1.0, 0.0 # Perfect p-value, 0 violations
        
    p_hat = num_violations / n
    
    # Likelihood Ratio
    lr = -2 * ( (n - num_violations) * np.log(1 - p) + num_violations * np.log(p) - 
                ( (n - num_violations) * np.log(1 - p_hat) + num_violations * np.log(p_hat) ) )
    
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return p_value, lr

def christoffersen_test(returns, var_estimates):
    """
    Christoffersen's Independence test.
    Checks if violations are clustered.
    """
    violations = (returns < -var_estimates).astype(int)
    n = len(violations)
    
    # Transition counts
    n00, n01, n10, n11 = 0, 0, 0, 0
    for i in range(1, n):
        if violations.iloc[i-1] == 0:
            if violations.iloc[i] == 0: n00 += 1
            else: n01 += 1
        else:
            if violations.iloc[i] == 0: n10 += 1
            else: n11 += 1
            
    # Transition probabilities
    if (n00 + n01) == 0 or (n10 + n11) == 0:
        return 1.0, 0.0 # Not enough violations for clustering test
        
    p01 = n01 / (n00 + n01)
    p11 = n11 / (n10 + n11)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    
    lr = -2 * ( (n00 + n10) * np.log(1 - p) + (n01 + n11) * np.log(p) - 
                ( n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11) + n11 * np.log(p11) ) )
                
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return p_value, lr

def adjust_var(var, predicted_risk, b1=0.30, b2=0.20):
    """
    Classification Adjusted VaR (Step 9).
    If predicted low: VaR = (1 - b1) * VaR
    If predicted high: VaR = (1 + b2) * VaR
    """
    if predicted_risk == 0:
        return var * (1 - b1)
    else:
        return var * (1 + b2)
