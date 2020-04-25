import os

import numpy as np
import torch

from . import nets
from . import utils

class NAFAgent:
    def __init__(self, obs_space_size, act_space_size, max_action):
        self.network = nets.BaselineNQF(obs_space_size, act_space_size, max_action)

    def to(self, device):
        self.network = self.network.to(device)
    
    def parallelize(self):
        if not isinstance(self.network, torch.nn.DataParallel):
            self.actor = torch.nn.DataParallel(self.network)
    
    def eval(self):
        self.network.eval()
    
    def train(self):
        self.network.train()
    
    def save(self, path):
        save_path = os.path.join(path, 'naf_net.pt')
        torch.save(self.network.state_dict(), save_path)
    
    def load(self, path):
        save_path = os.path.join(path, 'naf_net.pt')
        self.network.load_state_dict(torch.load(save_path, map_location=utils.device))

    def forward(self, state):
        state = self.process_state(state)
        self.network.eval()
        with torch.no_grad():
            mu, _, _ = self.network(state)
        return np.squeeze(mu.cpu().numpy(), 0)
    
    def process_state(self, state):
        return torch.from_numpy(np.expand_dims(state, 0).astype(np.float32))


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
        actor_path = os.path.join(path, 'actor.pt')
        critic_path = os.path.join(path, 'critic.pt')
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
    
    def load(self, path):
        actor_path = os.path.join(path, 'actor.pt')
        critic_path = os.path.join(path, 'critic.pt')
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
        return torch.from_numpy(np.expand_dims(state, 0).astype(np.float32))

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
        actor_path = os.path.join(path, 'actor.pt')
        critic1_path = os.path.join(path, 'critic1.pt')
        critic2_path = os.path.join(path, 'critic2.pt')
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic1.state_dict(), critic1_path)
        torch.save(self.critic2.state_dict(), critic2_path)
    
    def load(self, path):
        actor_path = os.path.join(path, 'actor.pt')
        critic1_path = os.path.join(path, 'critic1.pt')
        critic2_path = os.path.join(path, 'critic2.pt')
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
        return torch.from_numpy(np.expand_dims(state, 0).astype(np.float32))

