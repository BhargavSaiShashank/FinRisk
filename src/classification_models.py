import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import ADASYN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_baseline_models(X_train, y_train, X_test, y_test):
    """
    Trains baseline classifiers: Logistic Regression, SVM, and MLP.
    Uses ADASYN to handle class imbalance.
    """
    # Handle Imbalance
    ada = ADASYN(random_state=42)
    X_resampled, y_resampled = ada.fit_resample(X_train, y_train)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(probability=True),
        'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'G-Mean': np.sqrt(recall_score(y_test, y_pred) * recall_score(y_test, y_pred, pos_label=0))
        }
        
    return results, models

def get_g_mean(y_true, y_pred):
    rec_pos = recall_score(y_true, y_pred, pos_label=1)
    rec_neg = recall_score(y_true, y_pred, pos_label=0)
    return np.sqrt(rec_pos * rec_neg)
