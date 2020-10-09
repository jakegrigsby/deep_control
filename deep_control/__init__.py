import torch

from . import ddpg, sac, td3, sac_aug, mbpo, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
