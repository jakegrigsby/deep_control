import torch

from . import ddpg, sac, td3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
