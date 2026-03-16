import shap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def explain_ddqn_regime(agent, X_seq_background, X_seq_explain, feature_names, path='results/plots/shap_regime.png'):
    """
    Explains the Regime-Aware DDQN agent's Q-value outputs.
    Focuses on the predicted regime (max Q-value).
    """
    device = agent.device
    agent.model.eval()

    def model_wrapper(x_flat):
        # x_flat shape: [batch, seq_len * input_size]
        # Reshape back to [batch, seq_len, input_size]
        batch_size = x_flat.shape[0]
        x_seq = x_flat.reshape(batch_size, agent.seq_len, agent.input_size)
        x_tensor = torch.FloatTensor(x_seq).to(device)
        with torch.no_grad():
            q_values = agent.model(x_tensor)
        # Return the max Q-value or specific regime Q-values
        return q_values.cpu().numpy()

    # Flatten the sequences for SHAP KernelExplainer
    # SHAP expects 2D input
    X_flat_background = X_seq_background.reshape(X_seq_background.shape[0], -1)
    X_flat_explain = X_seq_explain.reshape(X_seq_explain.shape[0], -1)
    
    # We create names for sequence features: t-n_feature
    seq_feature_names = []
    for i in range(agent.seq_len):
        lag = agent.seq_len - 1 - i
        for name in feature_names:
            seq_feature_names.append(f"t-{lag}_{name}")

    print("Running SHAP KernelExplainer (this may take a while)...")
    explainer = shap.KernelExplainer(model_wrapper, shap.sample(X_flat_background, 10))
    # Explaining all actions
    shap_values = explainer.shap_values(X_flat_explain)

    # Plot for representative actions or just sum of impacts
    for r in range(min(agent.action_size, 4)): # Limit to first 4 for multi-plot, or we can adjust
        if r >= len(shap_values): break
        
        plt.figure(figsize=(12, 8))
        # Handle different SHAP output formats (version dependent)
        s_val = shap_values[r] if isinstance(shap_values, list) else shap_values[:, :, r]
        
        shap.summary_plot(s_val, X_flat_explain, feature_names=seq_feature_names, show=False, max_display=15)
        
        if agent.action_size == 15:
            mult = 0.8 + (r * 0.05)
            plt.title(f'SHAP Explainability: Multiplier {mult:.2f}')
        else:
            plt.title(f'SHAP Explainability: Action {r}')
            
        r_path = path.replace('.png', f'_action_{r}.png')
        os.makedirs(os.path.dirname(r_path), exist_ok=True)
        plt.savefig(r_path, bbox_inches='tight')
        plt.close()
    
    return shap_values
