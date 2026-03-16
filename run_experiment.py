import os
import pandas as pd
import numpy as np
import torch
import warnings
import scipy.stats as stats
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from src.data_loader import process_data
from src.feature_engineering import create_feature_matrix
from src.rolling_garch import rolling_garch_estimation
from src.var_estimation import adjust_risk_by_regime # Import the new helper
from src.regime_labeling import label_market_regimes, plot_regimes
from src.ddqn_agent import DDQNAgent
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
    
    # 2. Regime Labeling
    print(f"[{asset_name}] Labeling Market Regimes...")
    vix = macro_data['^VIX'].reindex(returns.index).ffill().bfill() if '^VIX' in macro_data.columns else None
    regime_labels = label_market_regimes(returns, vix=vix)
    plot_regimes(returns, regime_labels, path=f'results/{asset_name}/plots/regime_timeline.png')

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
        
        # Train Agent
        agent = DDQNAgent(input_size=X_scaled.shape[1], action_size=4, seq_len=seq_len)
        episodes = 10
        for e in range(episodes):
            for i in range(len(X_train_seq)-1):
                s, a, r, ns = X_train_seq[i], agent.act(X_train_seq[i]), 0, X_train_seq[i+1]
                r = agent.calculate_reward(a, y_train_regime[i])
                agent.remember(s, a, r, ns, False)
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

    # 7. Backtesting
    p_kupiec, _ = kupiec_pof_test(returns_test, var_regime)
    p_ind, _ = christoffersen_ind_test(returns_test, var_regime)
    basel_zone, n_viol = basel_traffic_light(returns_test, var_regime)
    z_as, p_as = acerbi_szekely_test(returns_test, var_regime, es_regime)
    
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
        'Asset': asset_name,
        'Basel_Zone': basel_zone,
        'Violations': n_viol,
        'Kupiec_p': p_kupiec,
        'Indep_p': p_ind,
        'Regime_Accuracy': (preds_combined == regime_labels.loc[valid_idx]).mean()
    }

def main():
    print("Loading datasets...")
    adj_close = pd.read_csv("data/processed/adj_close.csv", index_col=0, parse_dates=True)
    log_returns = pd.read_csv("data/processed/log_returns.csv", index_col=0, parse_dates=True)
    
    macro_tickers = ["^VIX", "^TNX", "CL=F", "GC=F"]
    macro_data = log_returns[macro_tickers]
    
    target_assets = ["^STOXX50E", "^GSPC", "^IXIC", "^N225"]
    all_results = []
    
    for asset in target_assets:
        try:
            res = run_regime_pipeline(asset, log_returns, adj_close, macro_data)
            all_results.append(res)
        except Exception as e:
            print(f"Error in {asset}: {e}")
            import traceback
            traceback.print_exc()
            
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv('results/regime_aware_comparison.csv', index=False)
    print("\nREGIME-AWARE EVALUATION COMPLETE")
    print(summary_df)

if __name__ == "__main__":
    main()
