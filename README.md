# Deep Control
## Simple PyTorch Implementations of Deep RL Algorithms for Continuous Control Research

This repository contains re-implementations of Deep RL algorithms for continuous action spaces. Some highlights:

1) Code is readable, and written to be easy to modify for future research. Many popular Deep RL frameworks are highly modular, which can make it confusing to identify the changes in a new method. Aside from universal components like the replay buffer, network architectures, etc., each implementation in this repo is contained in a single file.
2) Train and test on different environments (for generalization research).
3) Built-in Tensorboard logging, parameter saving.
4) Support for offline (batch) RL.
5) Quick setup for benchmarks like Gym MuJoco, Atari, PyBullet, and DeepMind Control Suite.

### What's included?

#### Deep Deterministic Policy Gradient (DDPG)
Paper: [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971), Lillicrap et al., 2015.

Description: a baseline model-free, offline, actor-critic method that forms the template for many of the other algorithms here.

Code: `deep_control.ddpg` (*with extra comments for an intro to deep actor-critics*)
Examples: `examples/basic_control/ddpg_gym.py`

#### Twin Delayed DDPG (TD3)
Paper: [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477), Fujimoto et al., 2018.

Description: Builds off of DDPG and makes several changes to improve the critic's learning and performance (Clipped Double Q Learning, Target Smoothing, Actor Delay). Also includes the TD regularization term from "[TD-Regularized Actor-Critic Methods](https://arxiv.org/abs/1812.08288)."

Code: `deep_control.td3`
Examples: `examples/basic_control/td3_gym.py`

Other References: [author's implementation](https://github.com/sfujim/TD3)

#### Soft Actor Critic (SAC)
Paper: [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290), Haarnoja et al., 2018.

Description: Samples actions from a stochastic actor rather than relying on added exploration noise during training. Uses a TD3-like double critic system. We *do* implement the learnable entropy coefficient approach described in the [follow-up paper](https://arxiv.org/abs/1812.05905). This version also supports the self-regularized crticic updates from GRAC (see below).

Code: `deep_control.sac`
Examples: `examples/dmc/sac.py`, `examples/sacd_demo.py`

Other References: [Yarats and Kostrikov's implementation](https://github.com/denisyarats/pytorch_sac), [author's implementation](https://github.com/haarnoja/sac).

#### Pixel SAC with Data Augmentation (SAC+AUG)
Paper: [Measuring Visual Generalization in Continuous Control from Pixels](https://arxiv.org/abs/2010.06740), Grigsby and Qi, 2020

Description: This is a pixel-specific version of SAC with a few tricks/hyperparemter settings to improve performance. We include many different data augmentation techniques, including those used in [RAD](https://arxiv.org/abs/2004.14990), [DrQ](https://arxiv.org/abs/2004.13649) and [Network Randomization](https://arxiv.org/abs/1910.05396). The DrQ augmentation is turned on by default, and has a huge impact on performance.

*Please Note: If you are interested in control from images, these features are implemented much more thoroughly in another repo: [jakegrigsby/super_sac](https://github.com/jakegrigsby/super_sac)*

Code: `deep_control.sac_aug`
Examples: `examples/dmcr/sac_aug.py`

Other References: [SAC+AE code](https://github.com/denisyarats/pytorch_sac_ae), [RAD Procgen code](https://github.com/pokaxpoka/rad_procgen), [DrQ](https://github.com/denisyarats/drq)

#### Self-Guided and Self-Regularized Actor-Critic (GRAC)
Paper: [GRAC: Self-Regularized Actor-Critic](https://arxiv.org/abs/2009.08973), Shao et al., 2020.

Description: GRAC is a combination of a stochastic policy with TD3-like stability improvements and CEM-based action selection like you'd see in Qt-Opt or CAQL.

Code: `deep_control.grac`
Examples: `examples/dmc/grac.py`

Other References: [author's implementation](https://github.com/stanford-iprl-lab/GRAC)

#### Randomized Ensemble Double Q-Learning (REDQ)
Paper: [Randomized Ensemble Double Q-Learning: Learning Fast Without a Model](https://openreview.net/forum?id=AY8zfZm0tDd)

Description: Extends the double Q trick to random subsets of a larger critic ensemble. Reduced Q function bias allows for a much higher replay ratio. REDQ is sample efficient but slow (compared to other model-free methods). We implement the SAC version.

Code: `deep_control.redq`
Examples: `examples/dmc/redq.py`

#### Distributional Correction (DisCor)
Paper: [DisCor: Corrective Feedback in Reinforcement Learning via Distribution Correction](https://arxiv.org/abs/2003.07305), Kumar et al., 2020.

Description: Reduce the effect of inaccurate target values propagating through the Q-function by learning to estimate the target networks' inaccuracies and adjusting the TD error accordingly. Implemented on top of standard SAC.

Code: `deep_control.discor`
Examples: `examples/dmc/discor.py`

#### Simple Unified Framework for Ensemble Learning (SUNRISE)
Paper: [SUNRISE: A Simple Unified Framework for Ensemble Learning in Deep Reinforcement Learning](https://arxiv.org/abs/2007.04938), Lee et al., 2020.

Description: Extends SAC using an ensemble of actors and critics. Adds UCB-based exploration, ensembled inference, and a simpler weighted bellman backup. This version does not use the replay buffer masks from the original.

Code: `deep_control.sunrise`
Examples: `examples/dmc/sunrise.py`

#### Stochastic Behavioral Cloning (SBC)

Description: A simple approach to offline RL that trains the actor network to emulate the action choices of the demonstration dataset. Uses the stochastic actor from SAC and some basic ensembling to make this a reasonable baseline.

Code: `deep_control.sbc`
Examples: `examples/d4rl/sbc.py`

#### Advantage Weighted Actor Critic (AWAC) and Critic Regularized Regression (CRR)
Paper: [Accelerating Online Reinforcement Learning with Offline Datasets](https://arxiv.org/abs/2006.09359), Nair et al., 2020. & [Critic Regularized Regression](https://arxiv.org/abs/2006.15134), Wang et al., 2020.

Description: TD3 with a stochastic policy and a modified actor update that makes better use of offline experience before finetuning in the online environment. The current implementation is a mix between AWAC and CRR. We allow for online finetuning and use standard critic networks as in AWAC, but add the binary advantage function, and max/mean advantage estimates from CRR. The `actor_per` experience prioritization trick is discussed in [A Closer Look at Advantage-Filtered Behavioral Cloning
in High-Noise Datasets](https://arxiv.org/abs/2110.04698), Grigsby and Qi, 2021.

Code: `deep_control.awac`
Examples: `examples/d4rl/awac.py`

#### Automatic Actor Critic (AAC)
Paper: [Towards Automatic Actor-Critic Solutions to Continuous Control](https://arxiv.org/abs/2106.08918), Grigsby et al., 2021

Description: AAC uses a genetic algorithm to automatically tune the hyperparameters of SAC. A population of SAC agents is trained in parallel with a shared relay buffer and several design decisions that reduce hyperparameter sensitivity while (mostly) preserving sample efficiency. Please refer to the paper for more details. **This is the official author implementation.**

Code: `deep_control.aac`


### Installation
```bash
git clone https://github.com/jakegrigsby/deep_control.git
cd deep_control
pip install -e .
```

### Examples
see the `examples` folder for a look at how to train agents in environments like the DeepMind Control Suite and OpenAI Gym.

