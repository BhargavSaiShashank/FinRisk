import torch
import torch.nn as nn

class TemporalEncoder(nn.Module):
    """
    LSTM-based encoder to extract latent features from market sequences.
    Input shape: (batch, seq_len, input_size)
    Output shape: (batch, latent_dim)
    """
    def __init__(self, input_size, hidden_dim=64, latent_dim=32, num_layers=2):
        super(TemporalEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=num_layers, 
                            batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        _, (h_n, _) = self.lstm(x)
        # Use last layer's hidden state
        # h_n shape: [num_layers, batch, hidden_dim]
        last_hidden = h_n[-1]
        latent = torch.relu(self.fc(last_hidden))
        return latent

class RegimeAwareQNetwork(nn.Module):
    """
    Combined network: Temporal Encoder + Q-value head.
    Input: Sequence of features.
    """
    def __init__(self, input_size, action_size=4, hidden_dim=64, latent_dim=32):
        super(RegimeAwareQNetwork, self).__init__()
        self.encoder = TemporalEncoder(input_size, hidden_dim, latent_dim)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        return self.head(latent)
