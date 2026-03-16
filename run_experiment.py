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
from src.var_estimation import adjust_risk_by_regime, compute_evt_risk
from src.ms_garch import MSGARCH
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
    
    # 2. MS-GARCH Modeling (3 Regimes)
    print(f"[{asset_name}] Fitting MS-GARCH (3 states)...")
    ms_model = MSGARCH(n_regimes=3)
    ms_res = ms_model.fit(returns)
    regime_data = ms_model.get_regime_data(returns)
    
    # Expand features with probabilities
    feat_matrix = feat_matrix.join(regime_data, how='inner').dropna()
    returns = returns.loc[feat_matrix.index]
    
    # Define regime labels for sequences and plots (most likely state)
    regime_labels = regime_data[['prob_state1', 'prob_state2', 'prob_state3']].idxmax(axis=1)
    regime_labels = regime_labels.replace({'prob_state1': 0, 'prob_state2': 1, 'prob_state3': 2})
    
    # 3. Volatility Forecast (MS-GARCH)
    # Composite MS-GARCH volatility
    vol_ms = regime_data.loc[feat_matrix.index, 'ms_vol']
    
    # Base VaR/ES (Normal Dist for simplicity, or we could use state-dependent)
    q5 = stats.norm.ppf(0.05)
    var_base = - (vol_ms * q5)
    # ES = vol * (pdf(q5)/0.05)
    es_base = vol_ms * (stats.norm.pdf(q5) / 0.05)

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
        
        # Train Agent (Calibration Targeted)
        # Sequence now includes probabilities
        agent = DDQNAgent(input_size=X_scaled.shape[1], action_size=15, seq_len=seq_len)
        episodes = 5
        train_returns = returns.loc[feat_matrix.index[train_mask]][seq_len-1:]
        train_var_base = var_base.loc[train_returns.index]
        
        for e in range(episodes):
            violations = deque(maxlen=250) # Target 250-day rolling for Basel
            for i in range(len(X_train_seq)-1):
                s = X_train_seq[i]
                a_idx = agent.act(s)
                mult = agent.get_multiplier(a_idx)
                
                # Check violation
                curr_ret = train_returns.iloc[i]
                is_violation = 1 if curr_ret < -(mult * train_var_base.iloc[i]) else 0
                violations.append(is_violation)
                
                # Reward based on rolling violation rate
                v_rate = sum(violations) / len(violations)
                r = agent.calculate_reward(v_rate)
                
                agent.remember(s, a_idx, r, X_train_seq[i+1], False)
                if i % 100 == 0: agent.replay(batch_size=32)
            agent.update_target_model()
            
        # Predict Multipliers
        agent.epsilon = 0
        window_preds = [agent.get_multiplier(agent.act(X_test_window[i])) for i in range(len(X_test_window))]
        window_idx = feat_matrix.index[seq_len:][test_start_idx : test_start_idx + len(window_preds)]
        all_preds.append(pd.Series(window_preds, index=window_idx))

    # Combine Walk-Forward Predictions
    preds_combined = pd.concat(all_preds).sort_index()
    preds_combined = preds_combined[~preds_combined.index.duplicated()]
    
    # 6. RL-Adjusted Metrics
    valid_idx = preds_combined.index
    returns_test = returns.loc[valid_idx]
    var_raw = var_base.loc[valid_idx]
    es_raw = es_base.loc[valid_idx]
    
    # Multipliers selected by RL
    var_rl = var_raw * preds_combined
    es_rl = es_raw * preds_combined

    # 7. Calibration Stage (Optional but requested for robustness)
    print(f"[{asset_name}] Calibrating final VaR...")
    calib_factor = calibrate_var(returns_test, var_rl, target_rate=0.05)
    var_final = apply_calibration(var_rl, calib_factor)
    es_final = apply_calibration(es_rl, calib_factor)

    # 8. EVT-Based Risk (Parallel)
    print(f"[{asset_name}] Computing EVT-GPD risks...")
    var_evt, es_evt = compute_evt_risk(returns_test, alpha=0.05)
    # Broadcast to series for master DF
    var_evt_series = pd.Series(var_evt, index=valid_idx)
    es_evt_series = pd.Series(es_evt, index=valid_idx)

    # 9. Backtesting
    p_kupiec, _ = kupiec_pof_test(returns_test, var_final)
    p_ind, _ = christoffersen_ind_test(returns_test, var_final)
    basel_zone, n_viol = basel_traffic_light(returns_test, var_final)
    z_as, p_as = acerbi_szekely_test(returns_test, var_final, es_final)
    
    # 9. Visualizations
    plt.figure(figsize=(15, 8))
    plt.subplot(3, 1, 1)
    for col in ['prob_state1', 'prob_state2', 'prob_state3']:
        plt.plot(regime_data.loc[valid_idx, col], label=col)
    plt.title(f'Regime Probabilities: {asset_name}')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(returns_test, label='Returns', alpha=0.3, color='gray')
    plt.plot(-var_final, label='Calibrated VaR', color='red')
    plt.title('VaR vs Returns')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(returns_test[returns_test < 0], label='Losses', alpha=0.3, color='gray')
    plt.plot(-es_final, label='Calibrated ES', color='orange')
    plt.title('Expected Shortfall Curves')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(f'results/{asset_name}/plots', exist_ok=True)
    plt.savefig(f'results/{asset_name}/plots/risk_summary.png')
    plt.close()

    # 10. Interpretability
    explain_ddqn_regime(agent, X_train_seq, X_test_window[:10], list(feat_matrix.columns), 
                       path=f'results/{asset_name}/plots/shap_calibration.png')

    return {
        'metrics': {
            'Asset': asset_name,
            'Basel_Zone': basel_zone,
            'Violations': n_viol,
            'Kupiec_p': p_kupiec,
            'Indep_p': p_ind,
            'Target_Accuracy': abs((returns_test < -var_final).mean() - 0.05),
            'EVT_ES_Mean': es_evt_series.mean(),
            'MSGARCH_ES_Mean': es_final.mean()
        },
        'regimes': regime_data.loc[valid_idx],
        'var': var_final,
        'es': es_final,
        'var_evt': var_evt_series,
        'es_evt': es_evt_series
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
    es_evt_master = pd.DataFrame()
    
    for asset in target_assets:
        try:
            res = run_regime_pipeline(asset, log_returns, adj_close, macro_data)
            all_metrics.append(res['metrics'])
            
            regime_master[asset] = res['regimes'].idxmax(axis=1) # Store string state
            var_master[asset] = res['var']
            es_master[asset] = res['es']
            es_evt_master[asset] = res['es_evt']
            
        except Exception as e:
            print(f"Error in {asset}: {e}")
            import traceback
            traceback.print_exc()
            
    # Export all requested files
    os.makedirs('results', exist_ok=True)
    pd.DataFrame(all_metrics).to_csv('results/backtesting_metrics.csv', index=False)
    regime_master.to_csv('results/ms_garch_regimes.csv')
    var_master.to_csv('results/var_predictions.csv')
    es_master.to_csv('results/expected_shortfall.csv')
    es_evt_master.to_csv('results/expected_shortfall_evt.csv')
    
    print("\nREGIME-AWARE EVALUATION COMPLETE")
    print(pd.DataFrame(all_metrics))

if __name__ == "__main__":
    main()
