import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd
import numpy as np
import os

def plot_boruta_ranking(feat_ranks, path='results/plots/boruta_ranking.png'):
    plt.figure(figsize=(10, 12))
    sns.barplot(x='rank', y='feature', data=feat_ranks.sort_values('rank').head(30), palette='magma')
    plt.title('Boruta Feature Ranking (Top 30)')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.ylabel('True Risk Label')
    plt.xlabel('Predicted Risk Label')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()

def plot_roc_curve(y_true, y_probs, title, path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()

def plot_var_results(returns, var_garch, var_rl, title, path):
    plt.figure(figsize=(15, 7))
    plt.plot(returns, label='Actual Returns', color='gray', alpha=0.3, lw=1)
    plt.plot(-var_garch, label='GARCH VaR (95%)', color='blue', lw=1.5, alpha=0.8)
    plt.plot(-var_rl, label='RL-Adjusted VaR (95%)', color='red', lw=1.5, alpha=0.8)
    
    # Highlight violations
    violations = returns[returns < -var_rl]
    plt.scatter(violations.index, violations, color='black', marker='x', s=10, label='RL VaR Violations')
    
    plt.title(title)
    plt.legend()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()

def plot_reward_curve(rewards, path='results/plots/rl_rewards.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, color='green')
    plt.title('DDQN Training Reward Convergence')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
