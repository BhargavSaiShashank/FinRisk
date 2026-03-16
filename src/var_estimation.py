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

def compute_parametric_es(mu, sigma, alpha=0.05, dist='studentst', nu=None):
    """
    Computes Expected Shortfall (CVaR).
    """
    if dist == 'normal':
        # ES for normal distribution
        phi_z = stats.norm.pdf(stats.norm.ppf(alpha))
        es = mu + sigma * (-phi_z / alpha)
        return -es
    elif dist == 'studentst':
        if nu is None: nu = 5
        q = stats.t.ppf(alpha, df=nu)
        # ES for Student-t
        # Formula: (f(q) / alpha) * ((nu + q**2) / (nu - 1))
        f_q = stats.t.pdf(q, df=nu)
        es_std = (f_q / alpha) * ((nu + q**2) / (nu - 1))
        # Scale back
        es = mu - sigma * es_std
        return -es
    else:
        raise ValueError("Unsupported distribution")

def generate_all_risks(returns, res_garch, res_gjr):
    """
    Generates VaR and ES at 1% and 5% for both GARCH and GJR-GARCH.
    """
    mu = 0 
    
    vol_garch = res_garch.conditional_volatility / 100
    vol_gjr = res_gjr.conditional_volatility / 100
    
    nu_garch = res_garch.params['nu']
    nu_gjr = res_gjr.params['nu']
    
    risks_df = pd.DataFrame(index=returns.index)
    
    # VaR
    risks_df['VaR_GARCH_5%'] = compute_parametric_var(mu, vol_garch, alpha=0.05, nu=nu_garch)
    risks_df['VaR_GARCH_1%'] = compute_parametric_var(mu, vol_garch, alpha=0.01, nu=nu_garch)
    risks_df['VaR_GJR_GARCH_5%'] = compute_parametric_var(mu, vol_gjr, alpha=0.05, nu=nu_gjr)
    risks_df['VaR_GJR_GARCH_1%'] = compute_parametric_var(mu, vol_gjr, alpha=0.01, nu=nu_gjr)
    
    # ES
    risks_df['ES_GARCH_5%'] = compute_parametric_es(mu, vol_garch, alpha=0.05, nu=nu_garch)
    risks_df['ES_GARCH_1%'] = compute_parametric_es(mu, vol_garch, alpha=0.01, nu=nu_garch)
    risks_df['ES_GJR_GARCH_5%'] = compute_parametric_es(mu, vol_gjr, alpha=0.05, nu=nu_gjr)
    risks_df['ES_GJR_GARCH_1%'] = compute_parametric_es(mu, vol_gjr, alpha=0.01, nu=nu_gjr)
    
    return risks_df

def compute_evt_risk(returns, alpha=0.05, threshold_pct=0.90):
    """
    Computes VaR and ES using Extreme Value Theory (GPD).
    POT (Peaks Over Threshold) approach.
    """
    # Threshold u (e.g. 90th percentile of losses)
    losses = -returns[returns < 0]
    if len(losses) < 50: # Fallback if not enough tail data
        return stats.norm.ppf(alpha) * np.std(returns), stats.norm.pdf(stats.norm.ppf(alpha))/alpha * np.std(returns)
        
    u = np.percentile(losses, threshold_pct * 100)
    extremes = losses[losses > u] - u
    
    if len(extremes) < 2:
        return np.percentile(losses, (1-alpha)*100), np.mean(losses[losses > np.percentile(losses, (1-alpha)*100)])

    # Fit GPD
    shape, loc, scale = stats.genpareto.fit(extremes)
    
    # N total, Nu exceeds threshold
    n = len(losses)
    nu = len(extremes)
    
    # VaR_alpha = u + (scale/shape) * [ ( (n/nu)*(1-alpha) )^-shape - 1 ]
    # Handle shape -> 0 case
    if abs(shape) < 1e-4:
        var_evt = u + scale * np.log((nu/n) / alpha)
    else:
        var_evt = u + (scale/shape) * ( ((nu/n) / alpha)**shape - 1 )
        
    # ES_alpha = (var_evt / (1-shape)) + ( (scale - shape*u) / (1-shape) )
    es_evt = (var_evt + scale - shape*u) / (1 - shape)
    
    return var_evt, es_evt

def adjust_risk_by_regime(base_value, regime_action):
    """
    Multipliers:
    0 (Calm) -> 1.0x
    1 (Rising Vol) -> 1.3x
    2 (Crash) -> 1.8x
    3 (Recovery) -> 1.2x
    """
    multipliers = {
        0: 1.0,
        1: 1.3,
        2: 1.8,
        3: 1.2
    }
    return base_value * multipliers.get(regime_action, 1.0)
