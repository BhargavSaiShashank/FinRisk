import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.tcn_model import TCN

def train_tcn_model(X_train, y_train, X_val, y_val, seq_len=10, epochs=20, batch_size=32):
    """
    Trains a TCN for binary risk classification.
    Inputs are reshaped into sequences.
    """
    def create_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X.iloc[i:(i + seq_len)].values)
            ys.append(y.iloc[i + seq_len])
        return torch.FloatTensor(np.array(Xs)), torch.FloatTensor(np.array(ys))

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_len)
    
    train_loader = DataLoader(TensorDataset(X_train_seq, y_train_seq), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_seq, y_val_seq), batch_size=batch_size)
    
    model = TCN(X_train.shape[1], [64, 64, 64, 64]) # 4 levels for [1,2,4,8] dilations
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print("Training TCN...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f}")
        
    return model, device

def predict_tcn(model, X, device, seq_len=10):
    model.eval()
    Xs = []
    # Note: we need the previous seq_len items for the first test prediction
    # For simplicity in this project, we assume X already includes the buffer if needed
    # or just pad/mask. Here we just process sequential chunks.
    with torch.no_grad():
        features = []
        for i in range(len(X) - seq_len):
            features.append(X.iloc[i:(i + seq_len)].values)
        
        X_tensor = torch.FloatTensor(np.array(features)).to(device)
        probs = model(X_tensor).squeeze().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        
    return preds, probs
