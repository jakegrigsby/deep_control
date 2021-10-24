import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from . import (
    ddpg,
    sac,
    sac_aug,
    td3,
    grac,
    redq,
    tsr_caql,
    discor,
    sunrise,
    sbc,
    awac,
    aac,
)
