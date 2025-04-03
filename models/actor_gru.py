import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class GRUActor(nn.Module):
    def __init__(self, state_dim=Config.STATE_DIM, action_dim=Config.ACTION_DIM, hidden_dim=Config.HIDDEN_DIM):
        super(GRUActor, self).__init__()
        self.gru = nn.GRU(state_dim, hidden_dim, num_layers=Config.GRU_LAYERS, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, state_dim)
        out, hidden = self.gru(x, hidden)
        
        # Take the last output
        out = out[:, -1, :]
        
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = torch.tanh(self.fc3(out))  # Actions are in [-1, 1]
        
        return out, hidden