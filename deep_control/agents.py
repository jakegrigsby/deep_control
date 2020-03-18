import os

import numpy as np
import torch

import nets
import utils

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
        with torch.no_grad():
            mu, _, _ = self.network(state)
        return np.squeeze(mu.cpu().numpy(), 0)
    
    def process_state(self, state):
        return torch.from_numpy(np.expand_dims(state, 0).astype(np.float32))


class DDPGAgent():
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


#############################################

class PendulumDDPGAgent(DDPGAgent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(3, 1)
        self.critic = nets.BaselineCritic(3, 1)

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

#############################################

class AntNAFAgent(NAFAgent):
    def __init__(self):
        self.network = nets.BaselineNQF(111, 8)

class AntDDPGAgent(DDPGAgent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(111, 8)
        self.critic = nets.BaselineCritic(111, 8)

#############################################

class WalkerNAFAgent(NAFAgent):
    def __init__(self):
        self.network = nets.BaselineNQF(17, 6)

class WalkerDDPGAgent(DDPGAgent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(17, 6)
        self.critic = nets.BaselineCritic(17, 6)

#############################################

class SwimmerNAFAgent(NAFAgent):
    def __init__(self):
        self.network = nets.BaselineNQF(8, 2)

class SwimmerDDPGAgent(DDPGAgent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(8, 2)
        self.critic = nets.BaselineCritic(8, 2)

#############################################

class CheetahNAFAgent(NAFAgent):
    def __init__(self):
        self.network = nets.BaselineNQF(17, 6)

class CheetahDDPGAgent(DDPGAgent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(17, 6)
        self.critic = nets.BaselineCritic(17, 6)

#############################################

class ReacherNAFAgent(NAFAgent):
    def __init__(self):
        self.network = nets.BaselineNQF(11, 2)

class ReacherDDPGAgent(DDPGAgent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(11, 2)
        self.critic = nets.BaselineCritic(11, 2)

#############################################

class HopperNAFAgent(NAFAgent):
    def __init__(self):
        self.network = nets.BaselineNQF(11, 3)

class HopperDDPGAgent(DDPGAgent):
    def __init__(self):
        super().__init__()
        self.actor = nets.BaselineActor(11, 3)
        self.critic = nets.BaselineCritic(11, 3)

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
##############################################

class DictBasedNAFAgent(NAFAgent):
    def process_state(self, state):
        state_goal = np.concatenate((state['observation'], state['desired_goal']))
        return super().process_state(state_goal)

class DictBasedDDPGAgent(DDPGAgent):
    def process_state(self, state):
        state_goal = np.concatenate((state['observation'], state['desired_goal']))
        return super().process_state(state_goal)

##############################################
# all v1s

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

##############################################

# all v0s

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
HandManipulatePenFullNAFAgnet = HandManipulateBlockNAFAgent


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
HandManipulatePenFullDDPGAgnet = HandManipulateBlockDDPGAgent



