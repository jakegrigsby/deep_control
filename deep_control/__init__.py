import torch

from . import ddpg, mbpo, models, sac, sac_aug, td3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
