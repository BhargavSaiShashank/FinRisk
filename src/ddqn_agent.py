import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from src.temporal_encoder import RegimeAwareQNetwork

class DDQNAgent:
    """
    Regime-Aware DDQN Agent.
    State: Sequence of features [seq_len, input_size]
    Action: 0=Calm, 1=Rising Vol, 2=Crash, 3=Recovery
    """
    def __init__(self, input_size, action_size=4, seq_len=30):
        self.input_size = input_size
        self.action_size = action_size
        self.seq_len = seq_len
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
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
        # state expected: [seq_len, input_size]
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device) # Add batch dim
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        # states shape: [batch, seq_len, input_size]
        states = torch.FloatTensor(np.array([s for s, a, r, ns, d in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([a for s, a, r, ns, d in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([r for s, a, r, ns, d in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([ns for s, a, r, ns, d in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([d for s, a, r, ns, d in minibatch])).to(self.device)

        self.model.train()
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            # DDQN: use model to pick action, target_model to evaluate
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

    def calculate_reward(self, action, true_label):
        """
        Regime Reward Logic:
        - Correct Regime (Match): +5
        - Early Crash Detect (Action=2, Label=2): +10
        - Missed Crash (Action!=2, Label=2): -15
        - False Crash Alarm (Action=2, Label!=2): -2
        - Others: default
        """
        if action == true_label:
            reward = 5.0
            if action == 2: reward += 5.0 # Extra bonus for getting crash right
            return reward
            
        if true_label == 2 and action != 2:
            return -15.0 # Heavy penalty for missing crash
            
        if action == 2 and true_label != 2:
            return -2.0 # False alarm
            
        return -1.0 # Generic penalty for mismatch

    def save(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target_model()
