# VaR Estimation via RL and GARCH Models (2025 Replication)

This repository implements the methodology described in the research paper: **"Bridging Econometrics and AI: VaR Estimation via Reinforcement Learning and GARCH Models (2025)"**.

## Methodology Overview

1.  **Volatility Modeling**: GARCH(1,1) and GJR-GARCH(1,1) are used to estimate conditional volatility of the Euro Stoxx 50 Index.
2.  **VaR Computation**: Parametric Value-at-Risk is computed at 95% and 99% confidence levels.
3.  **Risk Classification**: A binary classification problem is framed to identify "High Risk" states where returns exceed the calculated VaR threshold.
4.  **Reinforcement Learning**: A Double Deep Q-Network (DDQN) agent is trained to classify market states, optimizing a custom reward function based on the minority class ratio.
5.  **Classification-Adjusted VaR**: The final VaR is adjusted based on RL predictions to improve backtesting performance.

## Project Structure

- `src/`: Core logic (Loaders, GARCH, RL Agent).
- `notebooks/`: Sequential walk-through of the research methodology.
- `results/`: Backtesting results and plots.

## Installation

> [!IMPORTANT]
> Recommended Python version: **3.10 or 3.11**. 
> Some dependencies (like `arch` and `numba`) may have compatibility issues with Python 3.14+.

```bash
python -m pip install -r requirements.txt
```

## Reproducing Results

You can reproduce the entire experiment by running:

```bash
python run_experiment.py
```

Alternatively, you can explore the steps sequentially via the notebooks:
1.  Run `notebooks/01_data_analysis.ipynb` for data acquisition and EDA.
2.  Run `notebooks/02_garch_modeling.ipynb` for volatility estimation.
3.  Run `notebooks/03_rl_training.ipynb` for RL model training and evaluation.
