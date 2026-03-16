import os
import pandas as pd
import numpy as np
import torch
import warnings
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from src.data_loader import process_data
from src.feature_engineering import create_feature_matrix
from src.rolling_garch import rolling_garch_estimation, generate_rolling_vars
from src.var_estimation import calculate_risk_labels, generate_all_risks
from src.ddqn_agent import DDQNAgent
from src.backtesting_tests import kupiec_pof_test, christoffersen_ind_test, acerbi_szekely_test, basel_traffic_light, adjust_var_rl
from src.interpretability import explain_ddqn
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

def run_asset_pipeline(asset_name, log_returns, adj_close, macro_data):
    """
    Runs the full robust framework for a single asset.
    Includes walk-forward validation and final reporting.
    """
    print(f"\n{'='*20} Pipeline: {asset_name} {'='*20}")
    os.makedirs(f'results/{asset_name}/plots', exist_ok=True)
    
    # 1. Feature Prep
    returns = log_returns[asset_name]
    feat_matrix = create_feature_matrix(log_returns, adj_close)
    
    # Add macro data to the feat matrix if not already there
    # Ensure no overlap issues
    feat_matrix = feat_matrix.join(macro_data, how='inner', rsuffix='_macro').dropna()
    returns = returns.loc[feat_matrix.index]
    
    # 2. Volatility Analysis (Rolling)
    # We use a 250-day rolling window
    print(f"[{asset_name}] Running Rolling GARCH Estimation...")
    vol_garch = rolling_garch_estimation(returns, window=250, model_type='GARCH') 
    vol_gjr = rolling_garch_estimation(returns, window=250, model_type='GJR-GARCH')
    
    # 3. VaR & ES Estimation
    print(f"[{asset_name}] Generating VaR and ES estimates...")
    risks_df = pd.DataFrame(index=returns.index)
    
    q5 = stats.norm.ppf(0.05)
    risks_df['VaR_GARCH_5%'] = - (vol_garch * stats.norm.ppf(0.05))
    risks_df['ES_GARCH_5%'] = vol_garch * (stats.norm.pdf(stats.norm.ppf(0.05)) / 0.05)

    # 4. Walk-Forward Validation
    # We'll simulate 2 chunks: 2008-2020 (train) -> 2021-2025 (test)
    # In a full framework, this would be more granular.
    print(f"[{asset_name}] Starting Walk-Forward Evaluation...")
    
    labels, _ = calculate_risk_labels(returns, risks_df['VaR_GARCH_5%'])
    full_df = feat_matrix.join(labels.rename('target'), how='inner').dropna()
    
    split_date = '2022-01-01'
    X = full_df.drop(columns=['target'])
    y = full_df['target']
    
    X_train_raw = X[X.index < split_date]
    y_train = y[y.index < split_date]
    X_test_raw = X[X.index >= split_date]
    y_test = y[y.index >= split_date]
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # 5. RL Agent Training (Asymmetric Reward)
    print(f"[{asset_name}] Training Robust RL Agent...")
    agent = DDQNAgent(X_train.shape[1], 2)
    
    batch_size = 64
    episodes = 20 # Increased for stability
    for e in range(episodes):
        total_reward = 0
        for i in range(len(X_train)-1):
            s = X_train[i]
            a = agent.act(s)
            r = agent.calculate_reward(a, y_train.iloc[i])
            ns = X_train[i+1]
            agent.remember(s, a, r, ns, False)
            total_reward += r
            if i % 50 == 0: agent.replay(batch_size)
        agent.update_target_model()
        if (e+1) % 5 == 0:
            print(f"Episode {e+1}/{episodes} | Total Reward: {total_reward:.2f} | Eps: {agent.epsilon:.3f}")

    # 6. Final Evaluation
    print(f"[{asset_name}] Executing Final Evaluation & Stress Tests...")
    agent.epsilon = 0 # No exploration for testing
    preds = [agent.act(X_test[i]) for i in range(len(X_test))]
    preds_ser = pd.Series(preds, index=y_test.index)
    
    v_base = risks_df['VaR_GARCH_5%'].loc[y_test.index]
    es_base = risks_df['ES_GARCH_5%'].loc[y_test.index]
    
    # Apply RL adjustment to VaR and ES
    v_rl = pd.Series([adjust_var_rl(v, p) for v, p in zip(v_base, preds)], index=y_test.index)
    # Simple ES adjustment: scaling proportional to VaR
    es_rl = es_base * (v_rl / v_base) 

    # Backtesting
    p_kupiec, _ = kupiec_pof_test(returns.loc[y_test.index], v_rl)
    p_ind, _ = christoffersen_ind_test(returns.loc[y_test.index], v_rl)
    basel_zone, n_viol = basel_traffic_light(returns.loc[y_test.index], v_rl)
    
    # Acerbi-Szekely for ES
    z_as, p_as = acerbi_szekely_test(returns.loc[y_test.index], v_rl, es_rl)
    
    results = {
        'Asset': asset_name,
        'Violations': n_viol,
        'Basel_Zone': basel_zone,
        'Kupiec_p': p_kupiec,
        'Indep_p': p_ind,
        'AcerbiSzekely_Z': z_as,
        'Expected_Shortfall_Mean': es_rl.mean()
    }
    
    # 7. Model Interpretation
    print(f"[{asset_name}] Generating SHAP Interpretations...")
    explain_ddqn(agent, X_train, X_test[:50], list(X.columns), 
                path=f'results/{asset_name}/plots/shap_summary.png')

    # 8. Stress Testing
    print(f"[{asset_name}] Running Stress Test Scenarios...")
    stress_results = run_stress_tests(returns, v_rl, es_rl)
    results['Stress_Test_Failures'] = stress_results['failures']

    # Visualizations
    plt.figure(figsize=(12, 6))
    plt.plot(returns.loc[y_test.index], label='Returns', alpha=0.3, color='gray')
    plt.plot(-v_base, label='GARCH VaR (5%)', linestyle='--', color='blue')
    plt.plot(-v_rl, label='RL-Adjusted VaR (5%)', color='red')
    plt.fill_between(y_test.index, -es_rl, -v_rl, color='red', alpha=0.1, label='RL ES Zone')
    plt.title(f'Risk Estimates: {asset_name} (Test Set)')
    plt.legend()
    plt.savefig(f'results/{asset_name}/plots/var_es_comparison.png')
    plt.close()

    return results

def run_stress_tests(returns, var_series, es_series):
    """
    Simulates shocks and evaluates model response.
    """
    # 1. 2008 Crisis Scenario
    # 2. COVID Crash Scenario
    # 3. Sudden 10% Market Drop
    shocks = {
        'Financial_Crisis_2008': returns.loc['2008-09-01':'2008-12-31'],
        'COVID_Crash_2020': returns.loc['2020-03-01':'2020-05-31']
    }
    
    failures = 0
    for name, data in shocks.items():
        if len(data) == 0: continue
        v_stress = var_series.reindex(data.index).ffill().bfill()
        violations = (data < -v_stress).sum()
        if violations > len(data) * 0.1: # If > 10% violations, it's a 'failure' under stress
            failures += 1
            
    # Synthetic shock
    synthetic_drop = -0.10
    if len(var_series) > 0:
        last_var = var_series.iloc[-1]
        if synthetic_drop < -last_var:
            failures += 1
            
    return {'failures': failures}

def main():
    # 1. Load Processed Data
    print("Loading datasets...")
    adj_close = pd.read_csv("data/processed/adj_close.csv", index_col=0, parse_dates=True)
    log_returns = pd.read_csv("data/processed/log_returns.csv", index_col=0, parse_dates=True)
    
    # 2. Extract Macro Data
    macro_tickers = ["^VIX", "^TNX", "CL=F", "GC=F"]
    macro_data = log_returns[macro_tickers]
    
    all_results = []
    # Asset List
    target_assets = ["^STOXX50E", "^GSPC", "^IXIC", "^N225"]
    
    for asset in target_assets:
        try:
            res = run_asset_pipeline(asset, log_returns, adj_close, macro_data)
            all_results.append(res)
        except Exception as e:
            print(f"Error processing {asset}: {e}")
            
    # 3. Final Summary
    summary_df = pd.DataFrame(all_results)
    os.makedirs('results', exist_ok=True)
    summary_df.to_csv('results/robust_framework_comparison.csv', index=False)
    print("\n" + "="*50)
    print("ROBUST FRAMEWORK EVALUATION COMPLETE")
    print("="*50)
    print(summary_df)

if __name__ == "__main__":
    main()
