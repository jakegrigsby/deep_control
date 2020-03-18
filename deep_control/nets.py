
import torch
from torch import nn
import torch.nn.functional as F

class BaselineActor(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 400)
        self.bn2 = nn.BatchNorm1d(400)
        self.fc3 = nn.Linear(400, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, action_size)
    
    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        act = torch.tanh(self.out(x))
        return act

class BaselineCritic(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400 + action_size, 300)
        self.bn2 = nn.BatchNorm1d(300)
        self.fc3 = nn.Linear(300, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, 1)
    
    def forward(self, state, action):
        x = F.relu(self.bn1(self.fc1(state)))
        x_act = torch.cat((x, action), dim=1)
        x = F.relu(self.bn2(self.fc2(x_act)))
        x = F.relu(self.bn3(self.fc3(x)))
        val = self.out(x)
        return val

class BaselineNQF(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100)

        self.mu = nn.Linear(100, action_size)
        self.v = nn.Linear(100, 1)
        self.l = nn.Linear(100, action_size**2)

        self.tril_mask = torch.autograd.Variable(torch.tril(torch.ones(action_size, action_size), diagonal=-1).unsqueeze(0))
        self.diag_mask = torch.autograd.Variable(torch.diag(torch.diag(torch.ones(action_size, action_size))).unsqueeze(0))

        self.action_size = action_size

    def forward(self, state):
        x = self.bn1(F.relu(self.fc1(state)))
        x = self.bn2(F.relu(self.fc2(x)))

        mu = torch.tanh(self.mu(x))
        v = self.v(x)
        l = self.l(x).view(-1, self.action_size, self.action_size)
        return mu, l, v


