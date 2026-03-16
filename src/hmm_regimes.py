import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

class RegimeDetectorGMM:
    """
    Detects market regimes using a Gaussian Mixture Model (GMM).
    Proxy for Hidden Markov Model (HMM) for stable unsupervised clustering.
    """
    def __init__(self, n_regimes=4):
        self.n_regimes = n_regimes
        self.model = GaussianMixture(n_components=n_regimes, covariance_type='full', 
                                    random_state=42, n_init=10)
        self.regime_map = {}

    def prepare_features(self, df):
        """Extracts features for GMM clustering."""
        features = pd.DataFrame(index=df.index)
        features['returns'] = df['Close'].pct_change()
        features['vol'] = features['returns'].rolling(20).std()
        features['drawdown'] = (df['Close'] / df['Close'].rolling(window=252, min_periods=1).max()) - 1
        
        # Add VIX if available
        if 'VIX' in df.columns:
            features['vix'] = df['VIX']
        
        return features.dropna()

    def fit_predict(self, df):
        """Fits the GMM and predicts regime labels."""
        features = self.prepare_features(df)
        X = features.values
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit and Predict
        labels = self.model.fit_predict(X_scaled)
        
        # Map labels to meaningful regimes (0: Calm, 1: Vol, 2: Crash, 3: Recovery)
        # Based on mean volatility of the cluster
        means = []
        for i in range(self.n_regimes):
            means.append(features['vol'][labels == i].mean())
        
        # Sort indices by volatility to order: Calm < Vol < Crash
        # Recovery is a bit trickier, usually moderate vol but positive returns.
        # Simple heuristic for now: Sort by mean vol.
        sorted_indices = np.argsort(means)
        self.regime_map = {old: new for new, old in enumerate(sorted_indices)}
        
        final_labels = pd.Series(np.array([self.regime_map[label] for label in labels]), index=features.index)
        
        return final_labels.reindex(df.index).ffill().bfill().astype(int)

def plot_hmm_regimes(df, labels, asset_name):
    """Visualizes the detected regimes on the price chart."""
    plt.figure(figsize=(15, 7))
    colors = ['#2ecc71', '#f1c40f', '#e74c3c', '#3498db'] # Green, Yellow, Red, Blue
    regime_names = ['Calm', 'Rising Vol', 'Crash', 'Recovery']
    
    for i in range(4):
        mask = (labels == i)
        plt.scatter(df.index[mask], df['Close'][mask], c=colors[i], label=regime_names[i], s=5)
    
    plt.title(f"Dynamic Market Regimes (GMM) - {asset_name}")
    plt.legend()
    plt.savefig(f"results/plots/{asset_name}_regimes.png")
    plt.close()
