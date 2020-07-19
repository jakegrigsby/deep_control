import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal

from . import utils


class BaselineActor(nn.Module):
    def __init__(self, obs_size, action_size, max_action):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.out = nn.Linear(300, action_size)
        self.max_act = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        act = self.max_act * torch.tanh(self.out(x))
        return act


class BaselineCritic(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(obs_size + action_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.out = nn.Linear(300, 1)

    def forward(self, state, action):
        state_act = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(state_act))
        x = F.relu(self.fc2(x))
        val = self.out(x)
        return val


class StochasticActor(nn.Module):
    def __init__(self, obs_size, action_size, max_action):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.mu = nn.Linear(300, action_size)
        self.log_std = nn.Linear(300, action_size)
        self.max_act = max_action

    def forward(self, state, stochastic=False):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        if not stochastic:
            act = self.max_act * torch.tanh(mu)
            logp_a = None
        else:
            std = torch.exp(torch.clamp(log_std, -20, 2))
            dist = Normal(mu, std)
            unsquashed_act = dist.rsample()
            logp_a = dist.log_prob(unsquashed_act).sum(axis=-1)
            logp_a -= (
                2 * (np.log(2) - unsquashed_act - F.softplus(-2 * unsquashed_act))
            ).sum(axis=1)
            logp_a = logp_a.unsqueeze(1)
            act = self.max_act * torch.tanh(unsquashed_act)
        return act, logp_a
