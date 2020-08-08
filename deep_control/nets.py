import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from . import utils


class BaselineEncoder(nn.Module):
    def __init__(self, inp_size):
        super().__init__()
        self.fc1 = nn.Linear(inp_size, 400)

    def forward(self, x):
        return F.relu(self.fc1(x))


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
        val = self.out(x)
        return val


class StochasticActor(nn.Module):
    def __init__(self, obs_size, action_size, max_action):
        super().__init__()
        self.encoder = BaselineEncoder(obs_size)
        self.fc = nn.Linear(400, 300)
        self.mu = nn.Linear(300, action_size)
        self.log_std = nn.Linear(300, action_size)
        self.max_act = max_action

    def _sample_from(self, mu, log_std):
        std = torch.exp(torch.clamp(log_std, -20, 2))
        dist = Normal(mu, std)
        unsquashed_act = dist.rsample()
        logp_a = dist.log_prob(unsquashed_act).sum(axis=-1)
        logp_a -= (
            2 * (np.log(2) - unsquashed_act - F.softplus(-2 * unsquashed_act))
        ).sum(axis=1)
        logp_a = logp_a.unsqueeze(1)
        return unsquashed_act, logp_a

    def forward(self, state, stochastic=False):
        x = self.encoder(state)
        x = F.relu(self.fc(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        if not stochastic:
            act = self.max_act * torch.tanh(mu)
            logp_a = None
        else:
            unsquashed_act, logp_a = self._sample_from(mu, log_std)
            act = self.max_act * torch.tanh(unsquashed_act)
        return act, logp_a


class BaselinePixelEncoder(nn.Module):
    def __init__(self, obs_shape):
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

        self.fc = nn.Linear(output_height * output_width * 32, 400)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
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
        act_dist = Categorical(act_p)
        act = act_dist.sample().view(-1, 1)
        logp_a = torch.log(act_p + 1e-7)
        return act, logp_a

    def forward(self, state, stochastic=False):
        x = self.encoder(state)
        x = F.relu(self.fc(x))
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
