import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.data_loader import download_data, process_data, save_processed_data
from src.feature_engineering import create_feature_matrix
from src.rolling_garch import generate_rolling_vars
from src.var_estimation import calculate_risk_labels
from src.boruta_selection import run_boruta_selection, plot_boruta_results
from src.ddqn_agent import DDQNAgent
from src.reward_function import calculate_rho
from src.backtesting_tests import kupiec_pof_test, christoffersen_ind_test, adjust_var_rl
from src.evt_tail_analysis import fit_gpd_to_tails, plot_evt_tail
from src.visualization import plot_var_results, plot_confusion_matrix, plot_reward_curve
from src.tcn_trainer import train_tcn_model, predict_tcn
from src.diebold_mariano import dm_test
from src.tuning import optimize_rl_agent
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

def run_full_pipeline():
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models/saved_models', exist_ok=True)

    # 1. Data Download
    print("Step 1: Downloading Data (2008-2025)...")
    primary = "^STOXX50E"
    others = ["^FCHI", "^GDAXI", "^AEX", "FEZ", "EURUSD=X", "EURGBP=X"]
    raw = download_data([primary] + others, "2008-09-01", "2025-03-31")
    adj_close, log_returns = process_data(raw)

    # 2. Rolling GARCH Estimation (Window=1000)
    print("\nStep 2: Rolling GARCH & VaR Estimation...")
    returns = log_returns[primary]
    vars_df, vol_garch, vol_gjr = generate_rolling_vars(returns, window=1000)
    
    # Target labeling
    labels, c_thresh = calculate_risk_labels(returns, vars_df['VaR_GARCH_5%'])
    
    # 3. Features & Splitting
    print("\nStep 3: Feature Engineering & Sequential Splitting...")
    feat_matrix = create_feature_matrix(log_returns, adj_close)
    full_df = feat_matrix.join(pd.concat([vol_garch.rename('vol_rolling'), labels.rename('risk_label')], axis=1), how='inner').dropna()
    
    train_slice = full_df.index <= '2018-12-31'
    val_slice = (full_df.index > '2018-12-31') & (full_df.index <= '2022-12-31')
    test_slice = full_df.index > '2022-12-31'

    X_raw = full_df.drop(columns=['^STOXX50E_return', 'risk_label'])
    y = full_df['risk_label']

    X_train_raw, y_train = X_raw[train_slice], y[train_slice]
    X_val_raw, y_val = X_raw[val_slice], y[val_slice]
    X_test_raw, y_test = X_raw[test_slice], y[test_slice]

    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), index=X_train_raw.index, columns=X_train_raw.columns)
    X_val = pd.DataFrame(scaler.transform(X_val_raw), index=X_val_raw.index, columns=X_val_raw.columns)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), index=X_test_raw.index, columns=X_test_raw.columns)

    # 4. Feature Selection
    print("\nStep 4: Boruta Feature Selection...")
    selected_cols, _, selector = run_boruta_selection(X_train, y_train)
    plot_boruta_results(X_train, selector, 'results/plots/boruta_ranking.png')
    
    X_train_sel, X_val_sel, X_test_sel = X_train[selected_cols], X_val[selected_cols], X_test[selected_cols]

    # 5. RL Hyperparameter Optimization (Optuna)
    print("\nStep 5: RL Hyperparameter Optimization (Optuna - 30 trials)...")
    best_hparams = optimize_rl_agent(X_train_sel, y_train, X_val_sel, y_val, n_trials=30)
    print(f"Best HParams: {best_hparams}")

    # 6. Final Models Training (DDQN & TCN)
    print("\nStep 6: Training Final RL Agent & TCN Baseline...")
    rho = calculate_rho(y_train)
    agent = DDQNAgent(X_train_sel.shape[1], 2, rho)
    agent.learning_rate = best_hparams['learning_rate']
    agent.gamma = best_hparams['gamma']
    agent.epsilon_decay = best_hparams['epsilon_decay']
    
    rewards = []
    for e in range(15):
        tr = 0
        for i in range(len(X_train_sel)-1):
            s = X_train_sel.iloc[i].values
            a = agent.act(s)
            r = agent.calculate_reward(a, y_train.iloc[i])
            ns = X_train_sel.iloc[i+1].values
            agent.remember(s, a, r, ns, (i == len(X_train_sel)-2))
            tr += r
            if i % 20 == 0: agent.replay(best_hparams['batch_size'])
        agent.update_target_model()
        rewards.append(tr)
        print(f"Episode {e+1}/15 Reward: {tr:.2f} | Epsilon: {agent.epsilon:.3f}")
    
    plot_reward_curve(rewards, 'results/plots/rl_reward_curve.png')
    agent.save('models/saved_models/ddqn_agent.pth')

    tcn_model, device = train_tcn_model(X_train_sel, y_train, X_val_sel, y_val)
    torch.save(tcn_model.state_dict(), 'models/saved_models/tcn_model.pth')

    # 7. Final Evaluation & DM Test
    print("\nStep 7: Final Comparison & Statistical Validation...")
    agent.epsilon = 0
    test_preds_rl = [agent.act(X_test_sel.iloc[i].values) for i in range(len(X_test_sel))]
    
    # TCN needs sequence handling, we'll just align with sequential samples
    # For simplicity, we predict from the start of test set with a small lag
    seq_len = 10
    test_preds_tcn_raw, _ = predict_tcn(tcn_model, X_test_sel, device, seq_len=seq_len)
    
    # Align all series to the test set minus sequence length
    target_idx = X_test_sel.index[seq_len:]
    actual_returns = returns.loc[target_idx]
    
    v_base = vars_df['VaR_GARCH_5%'].loc[target_idx]
    v_gjr = vars_df['VaR_GJR_GARCH_5%'].loc[target_idx]
    
    # RL Adjustment
    preds_rl_aligned = test_preds_rl[seq_len:]
    v_rl = pd.Series([adjust_var_rl(v, p, b1=0.3, b2=0.2) for v, p in zip(v_base, preds_rl_aligned)], index=target_idx)
    
    # DM Test: RL vs GARCH
    dm_stat, dm_p = dm_test(actual_returns.values, -v_base.values, -v_rl.values)
    print(f"Diebold-Mariano RL vs GARCH: Stat={dm_stat:.4f}, p={dm_p:.4f}")

    # Metrics
    summary = []
    for name, v_ser in [("GARCH", v_base), ("GJR-GARCH", v_gjr), ("RL-Adjusted", v_rl)]:
        p_kupiec, _ = kupiec_pof_test(actual_returns, v_ser)
        p_christ, _ = christoffersen_ind_test(actual_returns, v_ser)
        summary.append({
            'Model': name,
            'Violations': (actual_returns < -v_ser).sum(),
            'Kupiec p-val': p_kupiec,
            'Christoffersen p-val': p_christ
        })
    
    results_df = pd.DataFrame(summary)
    results_df.to_csv('results/final_metrics.csv', index=False)
    print("\nFinal Results Summary:")
    print(results_df)

    # 8. Visualizations
    plot_var_results(actual_returns, v_base, v_rl, 'Rolling VaR Comparison (RL vs GARCH)', 'results/plots/final_var_rolling.png')
    plot_confusion_matrix(y[target_idx], preds_rl_aligned, 'RL Risk Classification', 'results/plots/rl_confusion.png')
    
    evt_res = fit_gpd_to_tails(actual_returns)
    plot_evt_tail(evt_res, 'results/plots/final_evt_plot.png')

    print("\nAdvanced Replication Complete.")

if __name__ == "__main__":
    run_full_pipeline()
