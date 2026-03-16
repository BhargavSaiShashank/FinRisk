import pandas as pd
import os

def export_regime_reports():
    """
    Synthesizes the results from the regime-aware pipeline into structured reports.
    """
    if not os.path.exists('results/regime_aware_comparison.csv'):
        print("Comparison file not found.")
        return

    df = pd.read_csv('results/regime_aware_comparison.csv')
    
    # 1. regime_metrics.csv
    regime_metrics = df[['Asset', 'Regime_Accuracy']]
    regime_metrics.to_csv('results/regime_metrics.csv', index=False)
    
    # 2. var_backtests.csv
    # Aggregating Kupiec and Independence tests
    var_backtests = df[['Asset', 'Violations', 'Basel_Zone', 'Kupiec_p', 'Indep_p']]
    var_backtests.to_csv('results/var_backtests.csv', index=False)
    
    # 3. expected_shortfall.csv
    # (Assuming we have ES metrics in the pipeline, if not we'll placeholder or just export what we have)
    # The current comparison CSV doesn't have ES mean yet, I'll check if I can extract more.
    
    print("Exported: regime_metrics.csv, var_backtests.csv")

if __name__ == "__main__":
    export_regime_reports()
