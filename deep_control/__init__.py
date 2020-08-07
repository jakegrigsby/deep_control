from . import ddpg, sac, td3
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
