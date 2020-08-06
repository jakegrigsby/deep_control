import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal

from . import utils


class BaselinePixelActor(nn.Module):
    def __init__(self, obs_shape, action_size, max_action):
        super().__init__()
        assert len(obs_shape) == 3
        channels = obs_shape[0]
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)

        output_height, output_width = utils.compute_conv_output(
            utils.compute_conv_output(
                utils.compute_conv_output(
                    obs_shape[1:], kernel_size=(3, 3), stride=(2, 2)
                ),
                kernel_size=(3, 3),
                stride=(2, 2),
            ),
            kernel_size=(3, 3),
            stride=(1, 1),
        )
        self.fc1 = nn.Linear(output_height * output_width * 32, 200)
        self.fc2 = nn.Linear(200, 200)
        self.out = nn.Linear(200, action_size)
        self.max_act = max_action

    def forward(self, state):
        state = state / 255.0
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        act = self.max_act * torch.tanh(self.out(x))
        return act


class BaselinePixelCritic(nn.Module):
    def __init__(self, obs_shape, action_size):
        super().__init__()
        assert len(obs_shape) == 3
        channels = obs_shape[0]
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)

        output_height, output_width = utils.compute_conv_output(
            utils.compute_conv_output(
                utils.compute_conv_output(
                    obs_shape[1:], kernel_size=(3, 3), stride=(2, 2)
                ),
                kernel_size=(3, 3),
                stride=(2, 2),
            ),
            kernel_size=(3, 3),
            stride=(1, 1),
        )
        self.fc1 = nn.Linear(action_size + (output_height * output_width * 32), 200)
        self.fc2 = nn.Linear(200, 200)
        self.out = nn.Linear(200, 1)

    def forward(self, state, action):
        state = state / 255.0
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # flatten, concat action
        x = x.view(x.size(0), -1)
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        val = self.out(x)
        return val


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
