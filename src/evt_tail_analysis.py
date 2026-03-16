import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def fit_gpd_to_tails(returns, threshold_quantile=0.95):
    """
    Fits Generalized Pareto Distribution (GPD) to the tail exceedances.
    For VaR, we look at negative returns (losses).
    """
    losses = -returns
    u = np.quantile(losses, threshold_quantile)
    exceedances = losses[losses > u] - u
    
    # Fit GPD
    # shape (xi), location (always 0 here relative to threshold), scale (sigma)
    shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.kstest(exceedances, 'genpareto', args=(shape, loc, scale))
    
    return {
        'threshold': u,
        'shape': shape,
        'scale': scale,
        'ks_pvalue': ks_pvalue,
        'exceedances': exceedances
    }

def plot_evt_tail(evt_results, path='results/plots/evt_tail_fit.png'):
    exceedances = evt_results['exceedances']
    shape, loc, scale = evt_results['shape'], 0, evt_results['scale']
    
    plt.figure(figsize=(10, 6))
    x = np.linspace(exceedances.min(), exceedances.max(), 100)
    plt.hist(exceedances, bins=20, density=True, alpha=0.5, label='Actual Exceedances')
    plt.plot(x, stats.genpareto.pdf(x, shape, loc, scale), 'r-', lw=2, label='GPD Fit')
    plt.title(f'EVT: GPD Fit to Tail Exceedances (KS p-value: {evt_results["ks_pvalue"]:.4f})')
    plt.legend()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
