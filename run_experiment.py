import os
import pandas as pd
import numpy as np
import torch
import warnings
import scipy.stats as stats
from tqdm import tqdm
from collections import deque
from sklearn.preprocessing import MinMaxScaler
from src.data_loader import process_data
from src.feature_engineering import create_feature_matrix
from src.rolling_garch import rolling_garch_estimation
from src.var_estimation import adjust_risk_by_regime
from src.hmm_regimes import RegimeDetectorGMM, plot_hmm_regimes
from src.ddqn_agent import DDQNAgent
from src.calibration import calibrate_var, apply_calibration
from src.backtesting_tests import kupiec_pof_test, christoffersen_ind_test, acerbi_szekely_test, basel_traffic_light
from src.interpretability import explain_ddqn_regime
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def create_sequences(data, labels, seq_len):
    """
    Creates [batch, seq_len, n_features] for LSTM input.
    """
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(labels[i-1]) # Target is regime at end of sequence
    return np.array(X), np.array(y)

def run_regime_pipeline(asset_name, log_returns, adj_close, macro_data):
    print(f"\n{'='*20} Regime Pipeline: {asset_name} {'='*20}")
    os.makedirs(f'results/{asset_name}/plots', exist_ok=True)
    
    # 1. Feature Prep
    returns = log_returns[asset_name]
    feat_matrix = create_feature_matrix(log_returns, adj_close)
    feat_matrix = feat_matrix.join(macro_data, how='inner', rsuffix='_macro').dropna()
    returns = returns.loc[feat_matrix.index]
    
    # 2. Regime Labeling (GMM)
    print(f"[{asset_name}] Detecting Market Regimes (GMM)...")
    detector = RegimeDetectorGMM(n_regimes=4)
    # Use feat_matrix directly for labeling to ensure alignment
    # We need a 'Close' column for the detector's internal feature prep if it expects it, 
    # but more robust is to pass just the returns.
    # I'll update Detector to take a generic dataframe with returns/vol if needed.
    # For now, let's just make sure we pass the right data.
    data_for_regime = feat_matrix.copy()
    data_for_regime['Close'] = adj_close.loc[feat_matrix.index, asset_name]
    regime_labels = detector.fit_predict(data_for_regime)
    plot_hmm_regimes(data_for_regime, regime_labels, asset_name)

    # 3. Volatility (Baseline Rolling GARCH)
    print(f"[{asset_name}] Running Rolling GARCH for base VaR...")
    vol_garch = rolling_garch_estimation(returns, window=250, model_type='GARCH')
    
    # Base VaR/ES
    q5 = stats.norm.ppf(0.05)
    var_base = - (vol_garch * q5)
    es_base = vol_garch * (stats.norm.pdf(q5) / 0.05)

    # 4. Sequential Preparation & Walk-Forward
    seq_len = 30
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(feat_matrix)
    
    # 5. Walk-Forward Loop
    # We'll use 2 major windows for execution speed: 
    # W1: 2008-2021 (Train) -> 2022-2023 (Test)
    # W2: 2008-2022 (Train) -> 2024-2025 (Test)
    test_starts = ['2022-01-01', '2024-01-01']
    all_preds = []
    
    for start_date in test_starts:
        print(f"[{asset_name}] Window starting {start_date}...")
        train_mask = feat_matrix.index < start_date
        test_mask = (feat_matrix.index >= start_date) & (feat_matrix.index < (pd.Timestamp(start_date) + pd.DateOffset(years=2)))
        
        if test_mask.sum() == 0: continue
        
        # Sequencify
        X_train_seq, y_train_regime = create_sequences(X_scaled[train_mask], regime_labels[train_mask].values, seq_len)
        X_test_seq, y_test_regime = create_sequences(X_scaled[feat_matrix.index < (pd.Timestamp(start_date) + pd.DateOffset(years=2))], 
                                                   regime_labels[feat_matrix.index < (pd.Timestamp(start_date) + pd.DateOffset(years=2))].values, seq_len)
        
        # Align test sequences to the actual test window dates
        test_start_idx = np.where(feat_matrix.index[seq_len:] >= start_date)[0][0]
        X_test_window = X_test_seq[test_start_idx:]
        
        # Train Agent (Stat-Targeted)
        agent = DDQNAgent(input_size=X_scaled.shape[1], action_size=4, seq_len=seq_len)
        episodes = 5 # Reduced episodes for speed in walk-forward
        train_returns = returns.loc[feat_matrix.index[train_mask]][seq_len-1:]
        train_var_base = var_base.loc[train_returns.index]
        
        for e in range(episodes):
            violations = deque(maxlen=100) # Rolling 100-day window for violation rate
            for i in range(len(X_train_seq)-1):
                s = X_train_seq[i]
                a = agent.act(s)
                
                # Check violation at current step
                mult = [1.0, 1.3, 1.8, 1.2][a]
                curr_ret = train_returns.iloc[i]
                is_violation = 1 if curr_ret < -(mult * train_var_base.iloc[i]) else 0
                violations.append(is_violation)
                
                # Reward based on rolling violation rate
                v_rate = sum(violations) / len(violations) if len(violations) > 0 else 0.05
                r = agent.calculate_reward(v_rate)
                
                agent.remember(s, a, r, X_train_seq[i+1], False)
                if i % 100 == 0: agent.replay(batch_size=32)
            agent.update_target_model()
            
        # Predict
        agent.epsilon = 0
        window_preds = [agent.act(X_test_window[i]) for i in range(len(X_test_window))]
        window_idx = feat_matrix.index[seq_len:][test_start_idx : test_start_idx + len(window_preds)]
        all_preds.append(pd.Series(window_preds, index=window_idx))

    # Combine Walk-Forward Predictions
    preds_combined = pd.concat(all_preds).sort_index()
    preds_combined = preds_combined[~preds_combined.index.duplicated()]
    
    # 6. Regime-Dependent Adjustments
    valid_idx = preds_combined.index
    returns_test = returns.loc[valid_idx]
    var_raw = var_base.loc[valid_idx]
    es_raw = es_base.loc[valid_idx]
    
    var_regime = pd.Series([adjust_risk_by_regime(v, p) for v, p in zip(var_raw, preds_combined)], index=valid_idx)
    es_regime = pd.Series([adjust_risk_by_regime(e, p) for e, p in zip(es_raw, preds_combined)], index=valid_idx)

    # 7. Calibration Stage
    print(f"[{asset_name}] Calibrating VaR...")
    # Use the first walk-forward window for calibration or a dedicated period
    calib_factor = calibrate_var(returns_test, var_regime, target_rate=0.05)
    var_calibrated = apply_calibration(var_regime, calib_factor)
    es_calibrated = apply_calibration(es_regime, calib_factor) # Apply same scale to ES

    # 8. Backtesting
    p_kupiec, _ = kupiec_pof_test(returns_test, var_calibrated)
    p_ind, _ = christoffersen_ind_test(returns_test, var_calibrated)
    basel_zone, n_viol = basel_traffic_light(returns_test, var_calibrated)
    z_as, p_as = acerbi_szekely_test(returns_test, var_calibrated, es_calibrated)
    
    # 8. Visualizations
    plt.figure(figsize=(15, 6))
    plt.plot(returns_test, label='Returns', alpha=0.3, color='gray')
    plt.plot(-var_raw, label='Base GARCH VaR', linestyle='--', color='blue')
    plt.plot(-var_regime, label='Regime-Aware VaR', color='red')
    plt.title(f'Regime-Aware VaR: {asset_name}')
    plt.legend()
    plt.savefig(f'results/{asset_name}/plots/regime_var_comparison.png')
    plt.close()

    # 9. Interpretability
    # Explain the last window's model
    explain_ddqn_regime(agent, X_train_seq, X_test_window[:10], list(feat_matrix.columns), 
                       path=f'results/{asset_name}/plots/shap_regime.png')

    return {
        'metrics': {
            'Asset': asset_name,
            'Basel_Zone': basel_zone,
            'Violations': n_viol,
            'Kupiec_p': p_kupiec,
            'Indep_p': p_ind,
            'Regime_Accuracy': (preds_combined == regime_labels.loc[valid_idx]).mean()
        },
        'regimes': preds_combined,
        'var': var_calibrated,
        'es': es_calibrated
    }

def main():
    print("Loading datasets...")
    adj_close = pd.read_csv("data/processed/adj_close.csv", index_col=0, parse_dates=True)
    log_returns = pd.read_csv("data/processed/log_returns.csv", index_col=0, parse_dates=True)
    
    macro_tickers = ["^VIX", "^TNX", "CL=F", "GC=F"]
    macro_data = log_returns[macro_tickers]
    
    target_assets = ["^STOXX50E", "^GSPC", "^IXIC", "^N225"]
    all_metrics = []
    regime_master = pd.DataFrame()
    var_master = pd.DataFrame()
    es_master = pd.DataFrame()
    
    for asset in target_assets:
        try:
            res = run_regime_pipeline(asset, log_returns, adj_close, macro_data)
            all_metrics.append(res['metrics'])
            
            regime_master[asset] = res['regimes']
            var_master[asset] = res['var']
            es_master[asset] = res['es']
            
        except Exception as e:
            print(f"Error in {asset}: {e}")
            import traceback
            traceback.print_exc()
            
    # Export all requested files
    os.makedirs('results', exist_ok=True)
    pd.DataFrame(all_metrics).to_csv('results/backtesting_metrics.csv', index=False)
    regime_master.to_csv('results/regime_states.csv')
    var_master.to_csv('results/calibrated_var.csv')
    es_master.to_csv('results/expected_shortfall.csv')
    
    print("\nREGIME-AWARE EVALUATION COMPLETE")
    print(pd.DataFrame(all_metrics))

if __name__ == "__main__":
    main()
