import shap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def explain_ddqn(agent, X_background, X_explain, feature_names, path='results/plots/shap_rl.png'):
    """
    Explains the DDQN agent's Q-value outputs using SHAP.
    Focuses on the 'High Risk' action (action 1).
    """
    device = agent.device
    agent.model.eval()

    def model_wrapper(x):
        x_tensor = torch.FloatTensor(x).to(device)
        with torch.no_grad():
            q_values = agent.model(x_tensor)
        # Return Q-value for action 1 (High Risk)
        return q_values[:, 1].cpu().numpy()

    # Use a small background dataset for KernelExplainer speed
    explainer = shap.KernelExplainer(model_wrapper, shap.sample(X_background, 50))
    shap_values = explainer.shap_values(X_explain)

    # Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False)
    plt.title('SHAP Explainability: RL High Risk Signal')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    
    return shap_values

def explain_baseline(model, X_explain, feature_names, path='results/plots/shap_baseline.png'):
    """
    Explains a scikit-learn model using SHAP.
    """
    explainer = shap.Explainer(model, X_explain)
    shap_values = explainer(X_explain)

    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, show=False)
    plt.title('SHAP Feature Importance: Baseline Model')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    
    return shap_values
