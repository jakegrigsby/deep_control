from collections import namedtuple
import random
import os

import numpy as np
import torchvision
import torch
import gym

import run

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlattenRoboticsDictWrapper(gym.Wrapper):
    def __init_(self, env):
        super().__init__(env)

    def step(self, action):
        obs_dict, rew, done, info = self.env.step(action)
        return self._flatten_obs(obs_dict), rew, done, info
    
    def reset(self):
        return self._flatten_obs(self.env.reset())

    def _flatten_obs(self, obs_dict):
        return np.concatenate((obs_dict['observation'], obs_dict['desired_goal']))


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def torch_and_pad(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return torch.from_numpy(
        x.astype(np.float32)
        ).unsqueeze(0)


def evaluate_agent(agent, env, args):
    agent.eval()
    returns = run.run(agent, env, args.eval_episodes, args.max_episode_steps, verbosity=0)
    mean_return = returns.mean()
    return mean_return

def collect_rollout(agent, random_process, eps, env, args):
    # collect new experience
    state = env.reset()
    random_process.reset_states()
    done = False 
    rollout = []
    for step in range(args.max_episode_steps):
        if done: break
        action = agent.forward(state)
        noisy_action = exploration_noise(action, random_process, eps)
        next_state, reward, done, info = env.step(noisy_action)
        if args.render: env.render()
        rollout.append((state, noisy_action, reward, next_state, done, info))
        state = next_state
    return rollout


class ReplayBuffer:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        while len(self.memory) < self.capacity:
            self.memory.append(None)
        self.position = 0
        self.max_filled = 0

    def push(self, state, action, reward, next_state, done):
        # pad batch axis
        action = torch_and_pad(action)
        reward = torch_and_pad(reward)
        state = torch_and_pad(state)
        next_state = torch_and_pad(next_state)
        done = torch.tensor([int(done)])
        self.memory[self.position] = Transition(state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        self.max_filled = min(self.max_filled + 1, self.capacity)

    def sample(self, batch_size):
        if self.max_filled < batch_size: return None
        return random.sample(self.memory[:self.max_filled], batch_size)

    def __len__(self):
        return len(self.memory)


def mean(lst):
    return float(sum(lst)) / len(lst)


def make_process_dirs(run_name, base_path='saves'):
    base_dir = os.path.join(base_path, run_name)
    i = 0
    while os.path.exists(base_dir + f"_{i}"):
        i += 1
    base_dir += f"_{i}"
    os.makedirs(base_dir)
    return base_dir

def exploration_noise(action, random_process, eps):
    return action + eps*random_process.sample().astype(np.float32)

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
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma

class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)
 
