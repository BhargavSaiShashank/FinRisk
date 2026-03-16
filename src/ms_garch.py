import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

class MSGARCH:
    """
    Markov-Switching GARCH(1,1) with 3 regimes (Haas et al. 2004).
    States: 1 (Calm), 2 (Volatile), 3 (Crisis)
    """
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.params = None
        self.transition_matrix = None
        self.regime_params = [] # List of (omega, alpha, beta)
        
    def _get_transition_matrix(self, p_params):
        """Constructs a 3x3 transition matrix from 6 free parameters."""
        # p_params: [p11, p12, p21, p22, p31, p32] - logit transformed
        matrix = np.zeros((3, 3))
        for i in range(3):
            logits = p_params[i*2 : (i+1)*2]
            exp_logits = np.exp(logits)
            denom = 1 + np.sum(exp_logits)
            matrix[i, 0] = exp_logits[0] / denom
            matrix[i, 1] = exp_logits[1] / denom
            matrix[i, 2] = 1 / denom
        return matrix

    def _garch_filter(self, returns, omega, alpha, beta):
        """Parallel GARCH filter for a single regime."""
        n = len(returns)
        v = np.zeros(n)
        v[0] = np.var(returns)
        for t in range(1, n):
            v[t] = omega + alpha * (returns[t-1]**2) + beta * v[t-1]
        return v

    def log_likelihood(self, params, returns):
        """Haas MS-GARCH Likelihood using Hamilton Filter."""
        n = len(returns)
        # Unpack params: 3*3 (garch) + 6 (transition) = 15 params
        garch_p = params[:9].reshape((3, 3)) # [[w1, a1, b1], [w2, a2, b2], ...]
        trans_p = params[9:]
        
        P = self._get_transition_matrix(trans_p)
        
        # 1. Compute parallel variances
        v = np.zeros((3, n))
        for j in range(3):
            v[j] = self._garch_filter(returns, garch_p[j, 0], garch_p[j, 1], garch_p[j, 2])
            
        # 2. Hamilton Filter
        filtered_probs = np.zeros((3, n))
        # Initial steady state if possible, else uniform
        curr_probs = np.ones(3) / 3 
        
        log_lik = 0
        for t in range(n):
            # Predicted probability p(s_t | F_{t-1})
            pred_probs = curr_probs @ P
            
            # Likelihood of returns in each state
            lik_j = norm.pdf(returns[t], 0, np.sqrt(np.maximum(v[:, t], 1e-9)))
            
            # Evidence
            evidence = np.sum(pred_probs * lik_j)
            if evidence <= 0: evidence = 1e-15
            log_lik += np.log(evidence)
            
            # Update (Filter)
            curr_probs = (pred_probs * lik_j) / evidence
            filtered_probs[:, t] = curr_probs
            
        return -log_lik

    def fit(self, returns):
        """Estimate parameters via MLE."""
        # Initial Guess: 
        # S1 (Calm), S2 (Vol), S3 (Crisis)
        initial_garch = [
            0.01, 0.05, 0.90, # Calm
            0.10, 0.15, 0.70, # Vol
            0.50, 0.30, 0.40  # Crisis
        ]
        initial_trans = [2, 0, 0, 2, 0, 0] # High diagonal
        initial_params = np.array(initial_garch + initial_trans)
        
        # Bounds to ensure stationarity and positivity
        bounds = []
        for _ in range(3):
            bounds += [(1e-6, 5), (0, 0.99), (0, 0.99)] # w, a, b
        for _ in range(6):
            bounds += [(-10, 10)] # logits
            
        res = minimize(self.log_likelihood, initial_params, args=(returns,), 
                       bounds=bounds, method='L-BFGS-B')
        
        self.params = res.x
        return res

    def get_regime_data(self, returns):
        """Returns regime probabilities and volatility forecasts."""
        n = len(returns)
        garch_p = self.params[:9].reshape((3, 3))
        trans_p = self.params[9:]
        P = self._get_transition_matrix(trans_p)
        
        v = np.zeros((3, n))
        for j in range(3):
            v[j] = self._garch_filter(returns, garch_p[j, 0], garch_p[j, 1], garch_p[j, 2])
            
        filtered_probs = np.zeros((3, n))
        curr_probs = np.ones(3) / 3 
        
        for t in range(n):
            pred_probs = curr_probs @ P
            lik_j = norm.pdf(returns[t], 0, np.sqrt(np.maximum(v[:, t], 1e-9)))
            evidence = np.sum(pred_probs * lik_j)
            if evidence <= 0: evidence = 1e-15
            curr_probs = (pred_probs * lik_j) / evidence
            filtered_probs[:, t] = curr_probs
            
        # Composite Volatility
        vol_composite = np.sqrt(np.sum(filtered_probs * v, axis=0))
        
        df = pd.DataFrame({
            'prob_state1': filtered_probs[0],
            'prob_state2': filtered_probs[1],
            'prob_state3': filtered_probs[2],
            'ms_vol': vol_composite
        }, index=returns.index)
        
        return df
