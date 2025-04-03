import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class GRUCritic(nn.Module):
    def __init__(self, state_dim=Config.STATE_DIM, action_dim=Config.ACTION_DIM, hidden_dim=Config.HIDDEN_DIM):
        super(GRUCritic, self).__init__()
        # Q1 architecture
        self.gru1 = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.gru2 = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        
    def forward(self, x, u, hidden1=None, hidden2=None):
        # x shape: (batch, seq_len, state_dim)
        # u shape: (batch, action_dim)
        
        # Q1
        out1, hidden1 = self.gru1(x, hidden1)
        out1 = out1[:, -1, :]  # Take last output
        out1 = torch.cat([out1, u], 1)
        out1 = F.relu(self.fc1(out1))
        out1 = self.fc2(out1)
        
        # Q2
        out2, hidden2 = self.gru2(x, hidden2)
        out2 = out2[:, -1, :]  # Take last output
        out2 = torch.cat([out2, u], 1)
        out2 = F.relu(self.fc3(out2))
        out2 = self.fc4(out2)
        
        return out1, out2, hidden1, hidden2
    
    def Q1(self, x, u, hidden=None):
        out, hidden = self.gru1(x, hidden)
        out = out[:, -1, :]
        out = torch.cat([out, u], 1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out, hidden