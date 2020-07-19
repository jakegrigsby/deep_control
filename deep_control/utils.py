import os
import random
from collections import namedtuple

import gym
import numpy as np
import torch

from . import run

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_hparams_dict(hparams_dict):
    return {key: val for key, val in hparams_dict.items() if val}


def warmup_buffer(buffer, env, warmup_steps, max_episode_steps):
    # use warmp up steps to add random transitions to the buffer
    state = env.reset()
    done = False
    steps_this_ep = 0
    for _ in range(warmup_steps):
        if done:
            state = env.reset()
            steps_this_ep = 0
            done = False
        rand_action = env.action_space.sample()
        next_state, reward, done, info = env.step(rand_action)
        buffer.push(state, rand_action, reward, next_state, done)
        state = next_state
        steps_this_ep += 1
        if steps_this_ep >= max_episode_steps:
            done = True


class FlattenRoboticsDictWrapper(gym.Wrapper):
    def __init_(self, env):
        super().__init__(env)

    def step(self, action):
        obs_dict, rew, done, info = self.env.step(action)
        return self._flatten_obs(obs_dict), rew, done, info

    def reset(self):
        return self._flatten_obs(self.env.reset())

    def _flatten_obs(self, obs_dict):
        return np.concatenate((obs_dict["observation"], obs_dict["desired_goal"]))


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


def torch_and_pad(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)


def evaluate_agent(
    agent, env, eval_episodes, max_episode_steps, render=False, verbosity=0
):
    agent.eval()
    returns = run.run(agent, env, eval_episodes, max_episode_steps, render, verbosity=0)
    mean_return = returns.mean()
    return mean_return


def mean(lst):
    return float(sum(lst)) / len(lst)


def make_process_dirs(run_name, base_path="dc_saves"):
    base_dir = os.path.join(base_path, run_name)
    i = 0
    while os.path.exists(base_dir + f"_{i}"):
        i += 1
    base_dir += f"_{i}"
    os.makedirs(base_dir)
    return base_dir


def exploration_noise(action, random_process, max_action):
    return np.clip(action + random_process.sample(), -max_action, max_action)


"""
Credit for update functions:
https://github.com/ikostrikov/pytorch-ddpg-naf
"""


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


""" This is all from: https://github.com/matthiasplappert/keras-rl/blob/master/rl/random.py """


class AnnealedGaussianProcess:
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.0
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(
        self,
        theta,
        mu=0.0,
        sigma=1.0,
        dt=1e-2,
        x0=None,
        size=1,
        sigma_min=None,
        n_steps_annealing=1000,
    ):
        super(OrnsteinUhlenbeckProcess, self).__init__(
            mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing
        )
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        )
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


class GaussianExplorationNoise:
    def __init__(self, size, start_scale=0.1, final_scale=1.0, steps_annealed=1000):
        assert start_scale >= final_scale
        self.size = size
        self.start_scale = start_scale
        self.final_scale = final_scale
        self.steps_annealed = steps_annealed
        self._current_scale = start_scale
        self._scale_slope = (start_scale - final_scale) / steps_annealed

    def sample(self):
        noise = self._current_scale * torch.randn(*self.size)
        self._current_scale = max(
            self._current_scale - self._scale_slope, self.final_scale
        )
        return noise.numpy()

    def reset_states(self):
        pass
