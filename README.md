# MLC @ UVA DDPG Baseline

![MLC Logo](misc/mlc_logo.png)

**A Pytorch implementation of Deep Determinisitc Policy Gradient for simple continuous control tasks.**

More info:
[Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf)


### Pretrained Agents
Watching a pretrained agent on `Pendulum-v0`:
```bash
python run.py --env Pendulum-v0 --agent saves/pretrained_pendulum --episodes 10
```

or on `MountainCarContinuous-v0`:
```bash
python run.py --env MountainCarContinuous-v0 --agent saves/pretrained_mountaincar --episodes 10
```

![Pendulum GIF](misc/pendulum.gif)
![Mountaincar GIF](misc/mountaincar.gif)

### Train Agents
```bash
python ddpg.py
```

There are a ton of CL flags. See the bottom of `ddpg.py` for a full list, but here are the important ones:
* `--env` is the gym environment id. Options are MountainCarContinuous-v0 and Pendulum-v0
* `--num_episodes` is how many episodes of experience to collect during training. Defaults to 500.
* `--batch_size` is how many sample transitions are passed through the networks at once during training. Defaults to 128. This may need to be reduced when running on CPUs.
* `--render` is either `1` or `0`. `1` lets you watch the agent as it learns. This slows the process down.
