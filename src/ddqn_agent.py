import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from src.temporal_encoder import RegimeAwareQNetwork

class DDQNAgent:
    """
    Regime-Probability Aware DDQN Agent for VaR Calibration.
    State: [prob_s1, prob_s2, prob_s3, vol, vix, oil, yields]
    Action: Discrete level mapped to [0.8, 1.5] multiplier.
    """
    def __init__(self, input_size, action_size=15, seq_len=30):
        self.input_size = input_size
        self.action_size = action_size # 15 levels: 0.8, 0.85, ..., 1.5
        self.seq_len = seq_len
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RegimeAwareQNetwork(input_size, action_size).to(self.device)
        self.target_model = RegimeAwareQNetwork(input_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def get_multiplier(self, action_idx):
        """Maps action index 0-14 to multiplier 0.8-1.5."""
        return 0.8 + (action_idx * 0.05)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([s for s, a, r, ns, d in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([a for s, a, r, ns, d in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([r for s, a, r, ns, d in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([ns for s, a, r, ns, d in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([d for s, a, r, ns, d in minibatch])).to(self.device)

        self.model.train()
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = torch.argmax(self.model(next_states), dim=1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def calculate_reward(self, current_violation_rate, target_rate=0.05):
        """
        Target-driven reward for VaR Calibration.
        Targets 5%, with sharp penalties for drifting (2%/8% thresholds).
        """
        error = abs(current_violation_rate - target_rate)
        reward = -error
        
        # Penalties
        if current_violation_rate < 0.02:
            reward -= 0.1 # Over-conservative
        elif current_violation_rate > 0.08:
            reward -= 0.2 # Too risky
            
        return reward

    def save(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target_model()
