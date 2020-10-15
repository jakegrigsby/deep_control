import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn

from . import utils


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class BigPixelEncoder(nn.Module):
    def __init__(self, obs_shape, out_dim=50):
        super().__init__()
        channels = obs_shape[0]
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1)

        output_height, output_width = utils.compute_conv_output(
            obs_shape[1:], kernel_size=(3, 3), stride=(2, 2)
        )
        for _ in range(3):
            output_height, output_width = utils.compute_conv_output(
                (output_height, output_width), kernel_size=(3, 3), stride=(1, 1)
            )

        self.fc = nn.Linear(output_height * output_width * 32, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.apply(weight_init)

    def forward(self, obs):
        obs /= 255.0
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.ln(x)
        state = torch.tanh(x)
        return state


class SmallPixelEncoder(nn.Module):
    def __init__(self, obs_shape, out_dim=50):
        super().__init__()
        channels = obs_shape[0]
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        output_height, output_width = utils.compute_conv_output(
            obs_shape[1:], kernel_size=(8, 8), stride=(4, 4)
        )

        output_height, output_width = utils.compute_conv_output(
            (output_height, output_width), kernel_size=(4, 4), stride=(2, 2)
        )

        output_height, output_width = utils.compute_conv_output(
            (output_height, output_width), kernel_size=(3, 3), stride=(1, 1)
        )

        self.fc = nn.Linear(output_height * output_width * 64, out_dim)
        self.apply(weight_init)

    def forward(self, obs):
        obs /= 255.0
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        state = self.fc(x)
        return state


class StochasticBigActor(nn.Module):
    def __init__(
        self, state_space_size, act_space_size, log_std_low=-10, log_std_high=2,
    ):
        super().__init__()
        self.fc1 = nn.Linear(state_space_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2 * act_space_size)
        self.log_std_low = log_std_low
        self.log_std_high = log_std_high
        self.apply(weight_init)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        # split output into mean log_std of action distribution
        mu, log_std = out.chunk(2, dim=1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_low + 0.5 * (self.log_std_high - self.log_std_low) * (
            log_std + 1
        )
        std = log_std.exp()
        dist = SquashedNormal(mu, std)
        return dist


class BigCritic(nn.Module):
    def __init__(self, state_space_size, act_space_size):
        super().__init__()
        self.fc1 = nn.Linear(state_space_size + act_space_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)

        self.apply(weight_init)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat((state, action), dim=1)))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class BaselineActor(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.out = nn.Linear(300, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        act = torch.tanh(self.out(x))
        return act


class BaselineCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.out = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        val = self.out(x)
        return val


"""
Credit for actor distribution code: https://github.com/denisyarats/pytorch_sac_ae/blob/master/sac_ae.py
"""


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class StochasticActor(nn.Module):
    def __init__(self, obs_size, action_size, log_std_low, log_std_high):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 2 * action_size)
        self.log_std_low = log_std_low
        self.log_std_high = log_std_high

    def forward(self, state, stochastic=False):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu, log_std = x.chunk(2, dim=1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_low + 0.5 * (self.log_std_high - self.log_std_low) * (
            log_std + 1
        )
        std = log_std.exp()
        dist = SquashedNormal(mu, std)
        return dist


class BaselineDiscreteActor(nn.Module):
    def __init__(self, obs_shape, action_size):
        super().__init__()
        self.fc1 = nn.Linear(obs_shape, 400)
        self.fc2 = nn.Linear(400, 300)
        self.act_p = nn.Linear(300, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        act_p = F.softmax(self.act_p(x), dim=1)
        dist = pyd.categorical.Categorical(act_p)
        return dist


class BaselineDiscreteCritic(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        self.fc1 = nn.Linear(obs_shape, 400)
        self.fc2 = nn.Linear(400, 300)
        self.out = nn.Linear(300, action_shape)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        vals = self.out(x)
        return vals
