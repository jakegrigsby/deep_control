import os

import numpy as np
import torch

from . import nets
from . import utils

class NAFAgent:
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

#############################################

class PendulumDDPGAgent(DDPGAgent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(3, 1)
        self.critic = nets.BaselineCritic(3, 1)

class PendulumTD3Agent(TD3Agent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(3, 1)
        self.critic1 = nets.BaselineCritic(3, 1)
        self.critic2 = nets.BaselineCritic(3, 1)

class PendulumNAFAgent(NAFAgent):
    def __init__(self):
        self.network = nets.BaselineNQF(3, 1)
        
#############################################

class MountaincarNAFAgent(NAFAgent):
    def __init__(self):
        self.network = nets.BaselineNQF(2, 1)

class MountaincarDDPGAgent(DDPGAgent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(2, 1)
        self.critic = nets.BaselineCritic(2, 1)

class MountaincarTD3Agent(TD3Agent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(2, 1)
        self.critic1 = nets.BaselineCritic(2, 1)
        self.critic2 = nets.BaselineCritic(2, 1)

#############################################

class AntNAFAgent(NAFAgent):
    def __init__(self):
        self.network = nets.BaselineNQF(111, 8)

class AntDDPGAgent(DDPGAgent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(111, 8)
        self.critic = nets.BaselineCritic(111, 8)

class AntTD3Agent(TD3Agent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(111, 8)
        self.critic1 = nets.BaselineCritic(111, 8)
        self.critic2 = nets.BaselineCritic(111, 8)

#############################################

class WalkerNAFAgent(NAFAgent):
    def __init__(self):
        self.network = nets.BaselineNQF(17, 6)

class WalkerDDPGAgent(DDPGAgent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(17, 6)
        self.critic = nets.BaselineCritic(17, 6)

class WalkerTD3Agent(TD3Agent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(17, 6)
        self.critic1 = nets.BaselineCritic(17, 6)
        self.critic2 = nets.BaselineCritic(17, 6)

#############################################

class SwimmerNAFAgent(NAFAgent):
    def __init__(self):
        self.network = nets.BaselineNQF(8, 2)

class SwimmerDDPGAgent(DDPGAgent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(8, 2)
        self.critic = nets.BaselineCritic(8, 2)

class SwimmerTD3Agent(TD3Agent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(8, 2)
        self.critic1 = nets.BaselineCritic(8, 2)
        self.critic2 = nets.BaselineCritic(8, 2)

#############################################

class CheetahNAFAgent(NAFAgent):
    def __init__(self):
        self.network = nets.BaselineNQF(17, 6)

class CheetahDDPGAgent(DDPGAgent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(17, 6)
        self.critic = nets.BaselineCritic(17, 6)

class CheetahTD3Agent(TD3Agent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(17, 6)
        self.critic1 = nets.BaselineCritic(17, 6)
        self.critic2 = nets.BaselineCritic(17, 6)

#############################################

class ReacherNAFAgent(NAFAgent):
    def __init__(self):
        self.network = nets.BaselineNQF(11, 2)

class ReacherDDPGAgent(DDPGAgent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(11, 2)
        self.critic = nets.BaselineCritic(11, 2)

class ReacherTD3Agent(TD3Agent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(11, 2)
        self.critic1 = nets.BaselineCritic(11, 2)
        self.critic2 = nets.BaselineCritic(11, 2)

#############################################

class HopperNAFAgent(NAFAgent):
    def __init__(self):
        self.network = nets.BaselineNQF(11, 3)

class HopperDDPGAgent(DDPGAgent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(11, 3)
        self.critic = nets.BaselineCritic(11, 3)

class HopperDDPGAgent(TD3Agent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(11, 3)
        self.critic1 = nets.BaselineCritic(11, 3)
        self.critic2 = nets.BaselineCritic(11, 3)

#############################################

class HumanoidNAFAgent(NAFAgent):
    def __init__(self):
        self.network = nets.BaselineNQF(376, 17)

HumanoidStandupNAFAgent = HumanoidNAFAgent

class HumanoidDDPGAgent(DDPGAgent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(376, 17)
        self.critic = nets.BaselineCritic(376, 17)

HumanoidStandupDDPGAgent = HumanoidDDPGAgent

class HumanoidTD3Agent(TD3Agent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(376, 17)
        self.critic1 = nets.BaselineCritic(376, 17)
        self.critic2 = nets.BaselineCritic(376, 17)

HumanoidStandupTD3Agent = HumanoidTD3Agent
##############################################

class DictBasedNAFAgent(NAFAgent):
    def process_state(self, state):
        state_goal = np.concatenate((state['observation'], state['desired_goal']))
        return super().process_state(state_goal)

class DictBasedDDPGAgent(DDPGAgent):
    def process_state(self, state):
        state_goal = np.concatenate((state['observation'], state['desired_goal']))
        return super().process_state(state_goal)

class DictBasedTD3Agent(TD3Agent):
    def process_state(self, state):
        state_goal = np.concatenate((state['observation'], state['desired_goal']))
        return super().process_state(state_goal)

##############################################

class FetchPushNAFAgent(DictBasedNAFAgent):
    def __init__(self):
        self.network = nets.BaselineNQF(28, 4)

class FetchReachNAFAgent(DictBasedNAFAgent):
    def __init__(self):
        self.network = nets.BaselineNQF(13, 4)

FetchSlideNAFAgent = FetchPushNAFAgent
FetchPickAndPlaceNAFAgent = FetchPushNAFAgent


class FetchPushDDPGAgent(DictBasedDDPGAgent):
    def __init__(self):
        self.actor = nets.BaselineActor(28, 4)
        self.critic = nets.BaselineCritic(28, 4)

class FetchReachDDPGAgent(DictBasedDDPGAgent):
    def __init__(self):
        self.actor = nets.BaselineActor(13, 4)
        self.critic = nets.BaselineCritic(13, 4)

FetchSlideDDPGAgent = FetchPushDDPGAgent
FetchPickAndPlaceDDPGAgent = FetchPushDDPGAgent


class FetchPushTD3Agent(DictBasedTD3Agent):
    def __init__(self):
        self.actor = nets.BaselineActor(28, 4)
        self.critic1 = nets.BaselineCritic(28, 4)
        self.critic2 = nets.BaselineCritic(28, 4)

class FetchReachTD3Agent(DictBasedTD3Agent):
    def __init__(self):
        self.actor = nets.BaselineActor(13, 4)
        self.critic1 = nets.BaselineCritic(13, 4)
        self.critic2 = nets.BaselineCritic(13, 4)

FetchSlideTD3Agent = FetchPushTD3Agent
FetchPickAndPlaceTD3Agent = FetchPushTD3Agent

##############################################


class HandReachNAFAgent(DictBasedNAFAgent):
    def __init__(self):
        self.network = nets.BaselineNQF(63+15, 20)

class HandManipulateBlockNAFAgent(DictBasedNAFAgent):
    def __init__(self):
        self.network = nets.BaselineNQF(61+7, 20)

HandManipulateBlockRotateZNAFAgent = HandManipulateBlockNAFAgent
HandManipulateBlockRotateParallelNAFAgent = HandManipulateBlockNAFAgent
HandManipulateBlockRotateXYZNAFAgnet = HandManipulateBlockNAFAgent
HandManipulateBlockFullNAFAgent = HandManipulateBlockNAFAgent

HandManipulateEggNAFAgent = HandManipulateBlockNAFAgent
HandManipulateEggRotateNAFAgent = HandManipulateBlockNAFAgent
HandManipulateEggFullNAFAgent = HandManipulateBlockNAFAgent

HandManipulatePenNAFAgent = HandManipulateBlockNAFAgent
HandManipulatePenRotateNAFAgent = HandManipulateBlockNAFAgent
HandManipulatePenFullNAFAgent = HandManipulateBlockNAFAgent


class HandReachDDPGAgent(DictBasedDDPGAgent):
    def __init__(self):
        self.actor = nets.BaselineActor(63+15, 20)
        self.critic = nets.BaselineCritic(63+15, 20)

class HandManipulateBlockDDPGAgent(DictBasedDDPGAgent):
    def __init__(self):
        self.actor = nets.BaselineActor(61+7, 20)
        self.critic = nets.BaselineCritic(61+7, 20)

HandManipulateBlockRotateZDDPGAgent = HandManipulateBlockDDPGAgent
HandManipulateBlockRotateParallelDDPGAgent = HandManipulateBlockDDPGAgent
HandManipulateBlockRotateXYZDDPGAgnet = HandManipulateBlockDDPGAgent
HandManipulateBlockFullDDPGAgent = HandManipulateBlockDDPGAgent

HandManipulateEggDDPGAgent = HandManipulateBlockDDPGAgent
HandManipulateEggRotateDDPGAgent = HandManipulateBlockDDPGAgent
HandManipulateEggFullDDPGAgent = HandManipulateBlockDDPGAgent

HandManipulatePenDDPGAgent = HandManipulateBlockDDPGAgent
HandManipulatePenRotateDDPGAgent = HandManipulateBlockDDPGAgent
HandManipulatePenFullDDPGAgent = HandManipulateBlockDDPGAgent


class HandReachTD3Agent(DictBasedTD3Agent):
    def __init__(self):
        self.actor = nets.BaselineActor(63+15, 20)
        self.critic1 = nets.BaselineCritic(63+15, 20)
        self.critic2 = nets.BaselineCritic(63+15, 20)

class HandManipulateBlockTD3Agent(DictBasedTD3Agent):
    def __init__(self):
        self.actor = nets.BaselineActor(61+7, 20)
        self.critic1 = nets.BaselineCritic(61+7, 20)
        self.critic2 = nets.BaselineCritic(61+7, 20)

HandManipulateBlockRotateZTD3Agent = HandManipulateBlockTD3Agent
HandManipulateBlockRotateParallelTD3Agent = HandManipulateBlockTD3Agent
HandManipulateBlockRotateXYZTD3Agnet = HandManipulateBlockTD3Agent
HandManipulateBlockFullTD3Agent = HandManipulateBlockTD3Agent

HandManipulateEggTD3Agent = HandManipulateBlockTD3Agent
HandManipulateEggRotateTD3Agent = HandManipulateBlockTD3Agent
HandManipulateEggFullTD3Agent = HandManipulateBlockTD3Agent

HandManipulatePenTD3Agent = HandManipulateBlockTD3Agent
HandManipulatePenRotateTD3Agent = HandManipulateBlockTD3Agent
HandManipulatePenFullTD3Agent = HandManipulateBlockTD3Agent


