import numpy as np
import pandas as pd

def calibrate_var(returns, var_series, target_rate=0.05):
    """
    Finds a scaling factor for VaR to match the target violation rate.
    Searches factors between 0.8 and 1.5.
    """
    factors = np.linspace(0.8, 1.5, 71) # 0.01 increments
    best_factor = 1.0
    min_diff = float('inf')
    
    for f in factors:
        calibrated_var = var_series * f
        violations = (returns < -calibrated_var).mean()
        diff = abs(violations - target_rate)
        
        if diff < min_diff:
            min_diff = diff
            best_factor = f
            
    print(f"VaR Calibration: Best Factor = {best_factor:.2f}, Final Violation Rate = {(returns < -(var_series * best_factor)).mean():.4f}")
    return best_factor

def apply_calibration(var_series, factor):
    return var_series * factor
