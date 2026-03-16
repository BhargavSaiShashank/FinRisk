import optuna
import numpy as np
import pandas as pd
from src.ddqn_agent import DDQNAgent
from src.reward_function import calculate_rho
from sklearn.metrics import f1_score

def optimize_rl_agent(X_train, y_train, X_val, y_val, n_trials=30):
    rho = calculate_rho(y_train)
    
    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.90, 0.99)
        eps_decay = trial.suggest_float("epsilon_decay", 0.990, 0.999)
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        
        agent = DDQNAgent(X_train.shape[1], 2, rho)
        agent.learning_rate = lr
        agent.gamma = gamma
        agent.epsilon_decay = eps_decay
        
        # Short training for optimization
        for e in range(3):
            for i in range(0, len(X_train)-1, 2): # sparse step to speed up
                s = X_train.iloc[i].values
                a = agent.act(s)
                r = agent.calculate_reward(a, y_train.iloc[i])
                ns = X_train.iloc[i+1].values
                agent.remember(s, a, r, ns, (i >= len(X_train)-3))
                if i % 20 == 0: agent.replay(batch_size)
        
        # Validation
        agent.epsilon = 0
        preds = [agent.act(X_val.iloc[i].values) for i in range(len(X_val))]
        f1 = f1_score(y_val, preds)
        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params
