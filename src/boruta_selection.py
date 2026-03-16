import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_boruta_selection(X, y, n_estimators='auto', random_state=42):
    """
    Runs Boruta feature selection using a RandomForestClassifier.
    """
    # X and y must be numpy arrays for BorutaPy
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=random_state)
    
    # Boruta object
    feat_selector = BorutaPy(rf, n_estimators=n_estimators, verbose=0, random_state=random_state)
    
    # Convert to numpy
    X_val = X.values
    y_val = y.values
    
    # Find all relevant features
    feat_selector.fit(X_val, y_val)
    
    # Results
    selected_features = X.columns[feat_selector.support_].tolist()
    tentative_features = X.columns[feat_selector.support_weak_].tolist()
    
    return selected_features, tentative_features, feat_selector

def plot_boruta_results(X, feat_selector, path='results/plots/boruta_importance.png'):
    """
    Plots the rank of features from Boruta selection.
    """
    ranks = feat_selector.ranking_
    feat_ranks = pd.DataFrame({'feature': X.columns, 'rank': ranks}).sort_values('rank')
    
    plt.figure(figsize=(10, 12))
    sns.barplot(x='rank', y='feature', data=feat_ranks.head(30), palette='viridis')
    plt.title('Boruta Feature Ranking (Top 30)')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
    
    return feat_ranks
