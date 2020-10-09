import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn

from . import utils


class BaselineEncoder(nn.Module):
    def __init__(self, inp_size):
        super().__init__()
        self.fc1 = nn.Linear(inp_size, 400)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        self.rep = h
        return h


class BaselineActor(nn.Module):
    def __init__(self, obs_size, action_size, max_action):
        super().__init__()
        self.encoder = BaselineEncoder(obs_size)
        self.fc = nn.Linear(400, 300)
        self.out = nn.Linear(300, action_size)
        self.max_act = max_action

    def forward(self, state):
        x = self.encoder(state)
        x = F.relu(self.fc(x))
        self.rep = x
        act = self.max_act * torch.tanh(self.out(x))
        return act


class BaselineCritic(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.encoder = BaselineEncoder(obs_size)
        self.fc = nn.Linear(400 + action_size, 300)
        self.out = nn.Linear(300, 1)

    def forward(self, state, action):
        x = self.encoder(state)
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc(x))
        self.rep = x
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
    def __init__(self, obs_size, action_size, max_action, log_std_low, log_std_high):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 2 * action_size)
        self.max_act = max_action
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


class BaselinePixelEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        channels = obs_shape[0]
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)

        output_height, output_width = utils.compute_conv_output(
            utils.compute_conv_output(obs_shape[1:], kernel_size=(3, 3), stride=(2, 2)),
            kernel_size=(3, 3),
            stride=(1, 1),
        )

        self.fc = nn.Linear(output_height * output_width * 32, 400)
        self.ln = nn.LayerNorm(400)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = self.ln(x)
        self.rep = x
        return x


class BaselinePixelActor(BaselineActor):
    def __init__(self, obs_shape, action_size, max_action):
        super().__init__(obs_shape[0], action_size, max_action)
        self.encoder = BaselinePixelEncoder(obs_shape)

    def forward(self, state):
        state = state / 255.0
        return super().forward(state)


class BaselinePixelCritic(BaselineCritic):
    def __init__(self, obs_shape, action_size):
        super().__init__(obs_shape[0], action_size)
        self.encoder = BaselinePixelEncoder(obs_shape)

    def forward(self, state, action):
        state = state / 255.0
        return super().forward(state, action)


class StochasticPixelActor(StochasticActor):
    def __init__(self, obs_shape, action_size, max_action):
        super().__init__(obs_shape[0], action_size, max_action)
        self.encoder = BaselinePixelEncoder(obs_shape)

    def forward(self, state, stochastic=False):
        state = state / 255.0
        return super().forward(state, stochastic=stochastic)


class BaselineDiscreteCritic(nn.Module):
    """
    aka Dueling DQN (https://arxiv.org/abs/1511.06581)
    """

    def __init__(self, obs_shape, action_size):
        super().__init__()
        self.encoder = BaselineEncoder(obs_shape)
        self.fc = nn.Linear(400, 300)
        self.v_out = nn.Linear(300, 1)
        self.a_out = nn.Linear(300, action_size)

    def forward(self, state):
        x = self.encoder(state)
        x = F.relu(self.fc(x))
        self.rep = x
        val = self.v_out(x)
        advantage = self.a_out(x)
        return val + (advantage - advantage.mean(1, keepdim=True))


class BaselinePixelDiscreteCritic(BaselineDiscreteCritic):
    def __init__(self, obs_shape, action_size):
        super().__init__(obs_shape[0], action_size)
        self.encoder = BaselinePixelEncoder(obs_shape)

    def forward(self, state):
        state = state / 255.0
        return super().forward(state)


class BaselineDiscreteActor(nn.Module):
    def __init__(self, obs_shape, action_size):
        super().__init__()
        self.encoder = BaselineEncoder(obs_shape)
        self.fc = nn.Linear(400, 300)
        self.act_p = nn.Linear(300, action_size)

    def _sample_from(self, act_p):
        act_dist = pyd.categorical.Categorical(act_p)
        act = act_dist.sample().view(-1, 1)
        logp_a = torch.log(act_p + 1e-8)
        return act, logp_a

    def forward(self, state, stochastic=False):
        x = self.encoder(state)
        x = F.relu(self.fc(x))
        self.rep = x
        act_p = F.softmax(self.act_p(x), dim=1)
        if not stochastic:
            act = torch.argmax(act_p, dim=1)
            logp_a = torch.log(act_p.gather(1, act.view(-1, 1)))
        else:
            act, logp_a = self._sample_from(act_p)
        act = act.float()
        return act, logp_a


class BaselinePixelDiscreteActor(BaselineDiscreteActor):
    def __init__(self, obs_shape, action_size):
        super().__init__(obs_shape[0], action_size)
        self.encoder = BaselinePixelEncoder(obs_shape)

    def forward(self, state, stochastic=False):
        state = state / 255.0
        return super().forward(state, stochastic=stochastic)
