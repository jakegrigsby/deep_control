
import torch
from torch import nn
import torch.nn.functional as F

from . import utils

class BaselineActor(nn.Module):
    def __init__(self, obs_size, action_size, max_action):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 300)
        self.bn2 = nn.BatchNorm1d(300)
        self.out = nn.Linear(300, action_size)
        self.max_act = max_action
    
    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        act = self.max_act * torch.tanh(self.out(x))
        return act

class BaselineCritic(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(obs_size + action_size, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 300)
        self.bn2 = nn.BatchNorm1d(300)
        self.out = nn.Linear(300, 1)
    
    def forward(self, state, action):
        state_act = torch.cat((state, action), dim=1)
        x = F.relu(self.bn1(self.fc1(state_act)))
        x = F.relu(self.bn2(self.fc2(x)))
        val = self.out(x)
        return val

class BaselineNQF(torch.nn.Module):
    def __init__(self, obs_size, action_size, max_action):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 300)
        self.bn2 = nn.BatchNorm1d(300)

        self.mu = nn.Linear(300, action_size)
        self.v = nn.Linear(300, 1)
        self.l = nn.Linear(300, action_size**2)

        self.tril_mask = torch.autograd.Variable(torch.tril(torch.ones(action_size, action_size), diagonal=-1).unsqueeze(0).to(utils.device))
        self.diag_mask = torch.autograd.Variable(torch.diag(torch.diag(torch.ones(action_size, action_size))).unsqueeze(0).to(utils.device))

        self.action_size = action_size
        self.max_act = max_action

    def forward(self, state):
        x = self.bn1(F.relu(self.fc1(state)))
        x = self.bn2(F.relu(self.fc2(x)))

        mu = self.max_act * torch.tanh(self.mu(x))
        v = self.v(x)
        l = self.l(x).view(-1, self.action_size, self.action_size)
        return mu, l, v


