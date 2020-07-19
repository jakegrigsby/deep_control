import os

import numpy as np
import torch

from . import nets, utils


class DDPGAgent:
    def __init__(self, obs_space_size, action_space_size, max_action):
        self.actor = nets.BaselineActor(obs_space_size, action_space_size, max_action)
        self.critic = nets.BaselineCritic(obs_space_size, action_space_size)

    def to(self, device):
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def train(self):
        self.actor.train()
        self.critic.train()

    def save(self, path):
        actor_path = os.path.join(path, "actor.pt")
        critic_path = os.path.join(path, "critic.pt")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load(self, path):
        actor_path = os.path.join(path, "actor.pt")
        critic_path = os.path.join(path, "critic.pt")
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))

    def forward(self, state):
        # first need to add batch dimension and convert to torch tensors
        state = self.process_state(state)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)
        return np.squeeze(action.cpu().numpy(), 0)

    def process_state(self, state):
        return torch.from_numpy(np.expand_dims(state, 0).astype(np.float32)).to(
            utils.device
        )


class TD3Agent:
    def __init__(self, obs_space_size, act_space_size, max_action):
        self.actor = nets.BaselineActor(obs_space_size, act_space_size, max_action)
        self.critic1 = nets.BaselineCritic(obs_space_size, act_space_size)
        self.critic2 = nets.BaselineCritic(obs_space_size, act_space_size)

    def to(self, device):
        self.actor = self.actor.to(device)
        self.critic1 = self.critic1.to(device)
        self.critic2 = self.critic2.to(device)

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def save(self, path):
        actor_path = os.path.join(path, "actor.pt")
        critic1_path = os.path.join(path, "critic1.pt")
        critic2_path = os.path.join(path, "critic2.pt")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic1.state_dict(), critic1_path)
        torch.save(self.critic2.state_dict(), critic2_path)

    def load(self, path):
        actor_path = os.path.join(path, "actor.pt")
        critic1_path = os.path.join(path, "critic1.pt")
        critic2_path = os.path.join(path, "critic2.pt")
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic1.load_state_dict(torch.load(critic1_path))
        self.critic2.load_state_dict(torch.load(critic2_path))

    def forward(self, state):
        # first need to add batch dimension and convert to torch tensors
        state = self.process_state(state)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)
        return np.squeeze(action.cpu().numpy(), 0)

    def process_state(self, state):
        return torch.from_numpy(np.expand_dims(state, 0).astype(np.float32)).to(
            utils.device
        )


class SACAgent(TD3Agent):
    def __init__(self, obs_space_size, act_space_size, max_action):
        self.actor = nets.StochasticActor(obs_space_size, act_space_size, max_action)
        self.critic1 = nets.BaselineCritic(obs_space_size, act_space_size)
        self.critic2 = nets.BaselineCritic(obs_space_size, act_space_size)
        self.max_act = max_action

    def forward(self, state):
        state = self.process_state(state)
        self.actor.eval()
        with torch.no_grad():
            act, _ = self.actor.forward(state, stochastic=False)
        return np.squeeze(act.cpu().numpy(), 0)

    def stochastic_forward(self, state, process_states=False, track_gradients=True):
        if process_states:
            state = self.process_state(state)
        if track_gradients:
            act, logp_a = self.actor.forward(state, stochastic=True)
        else:
            with torch.no_grad():
                act, logp_a = self.actor.forward(state, stochastic=True)
        if process_states:
            act = np.squeeze(act.cpu().numpy(), 0)
        return act, logp_a
