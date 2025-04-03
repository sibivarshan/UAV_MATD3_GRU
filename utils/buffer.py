import numpy as np
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for idx in idxs:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)

class MultiAgentReplayBuffer:
    def __init__(self, max_size, num_agents):
        self.max_size = max_size
        self.num_agents = num_agents
        self.buffers = [ReplayBuffer(max_size) for _ in range(num_agents)]
    
    def add(self, states, actions, rewards, next_states, dones):
        for i in range(self.num_agents):
            self.buffers[i].add(
                states[i], 
                actions[i], 
                rewards[i], 
                next_states[i], 
                dones[i]
            )
    
    def sample(self, batch_size):
        samples = [buffer.sample(batch_size) for buffer in self.buffers]
        
        # Stack samples from all agents
        states = torch.stack([s[0] for s in samples])
        actions = torch.stack([s[1] for s in samples])
        rewards = torch.stack([s[2] for s in samples])
        next_states = torch.stack([s[3] for s in samples])
        dones = torch.stack([s[4] for s in samples])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffers[0])