import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, dropout_p=0.2):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.dropout1 = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_p)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

class DDQNAgent:
    def __init__(self, state_size, action_size, minority_ratio=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(state_size, action_size).to(self.device)
        self.target_model = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_t = torch.FloatTensor(state).to(self.device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
            
        self.model.eval() # Eval mode for dropout
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor(np.array([s for s, a, r, ns, d in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([a for s, a, r, ns, d in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([r for s, a, r, ns, d in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([ns for s, a, r, ns, d in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([d for s, a, r, ns, d in minibatch])).to(self.device)

        self.model.train() # Train mode for gradient step
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = torch.argmax(self.model(next_states), dim=1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def calculate_reward(self, action, label):
        """
        Asymmetric risk-aware reward definition:
        - Correct Risk detection (TP): +5
        - Missed Crash (FN): -10
        - False Alarm (FP): -1
        - Correct Low Risk (TN): +1
        """
        if action == 1 and label == 1: return 5.0    # TP
        if action == 1 and label == 0: return -1.0   # FP
        if action == 0 and label == 0: return 1.0    # TN
        if action == 0 and label == 1: return -10.0  # FN
        return 0.0

    def save(self, path):
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'epsilon': self.epsilon,
            'input_dim': self.state_size
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.update_target_model()
        self.epsilon = checkpoint['epsilon']
