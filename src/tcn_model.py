import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=self.padding, dilation=dilation)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               padding=self.padding, dilation=dilation)
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [batch, channels, seq_len]
        out = self.relu(self.conv1(x))
        out = out[:, :, :-self.padding] # causal padding
        out = self.dropout1(out)
        
        out = self.relu(self.conv2(out))
        out = out[:, :, :-self.padding] # causal padding
        out = self.dropout2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [ResidualBlock(in_channels, out_channels, kernel_size, 
                                     dilation=dilation_size, dropout=dropout)]
        
        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        x = x.transpose(1, 2) # [batch, input_size, seq_len]
        y = self.tcn(x)
        # pooled output from last time step
        o = self.linear(y[:, :, -1])
        return self.sigmoid(o)
