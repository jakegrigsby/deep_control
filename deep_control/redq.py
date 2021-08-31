import argparse
import copy
import math
import os
from itertools import chain
import random

import numpy as np
import tensorboardX
import torch
import torch.nn.functional as F
import tqdm

from . import envs, nets, replay, run, utils, device


class REDQAgent:
    def __init__(
        self,
        obs_space_size,
        act_space_size,
        log_std_low,
        log_std_high,
        critic_ensemble_size=10,
        actor_network_cls=nets.StochasticActor,
        critic_network_cls=nets.BigCritic,
        hidden_size=1024,
    ):
        self.actor = actor_network_cls(
            obs_space_size,
            act_space_size,
            log_std_low,
            log_std_high,
            dist_impl="pyd",
            hidden_size=hidden_size,
        )
        self.critics = [
            critic_network_cls(obs_space_size, act_space_size, hidden_size=hidden_size)
            for _ in range(critic_ensemble_size)
        ]

    def to(self, device):
        self.actor = self.actor.to(device)
        for i, critic in enumerate(self.critics):
            self.critics[i] = critic.to(device)

    def eval(self):
        self.actor.eval()
        for critic in self.critics:
            critic.eval()

    def train(self):
        self.actor.train()
        for critic in self.critics:
            critic.train()

    def save(self, path):
        actor_path = os.path.join(path, "actor.pt")
        torch.save(self.actor.state_dict(), actor_path)
        for i, critic in enumerate(self.critics):
            critic_path = os.path.join(path, f"critic{i}.pt")
            torch.save(critic.state_dict(), critic_path)

    def load(self, path):
        actor_path = os.path.join(path, "actor.pt")
        self.actor.load_state_dict(torch.load(actor_path))
        for i, critic in enumerate(self.critics):
            critic_path = os.path.join(path, f"critic{i}.pt")
            critic.load_state_dict(torch.load(critic_path))

    def forward(self, state, from_cpu=True):
        if from_cpu:
            state = self.process_state(state)
        self.actor.eval()
        with torch.no_grad():
            act_dist = self.actor.forward(state)
            act = act_dist.mean
        self.actor.train()
        if from_cpu:
            act = self.process_act(act)
        return act

    def sample_action(self, state, from_cpu=True):
        if from_cpu:
            state = self.process_state(state)
        self.actor.eval()
        with torch.no_grad():
            act_dist = self.actor.forward(state)
            act = act_dist.sample()
        self.actor.train()
        if from_cpu:
            act = self.process_act(act)
        return act

    def process_state(self, state):
        return torch.from_numpy(np.expand_dims(state, 0).astype(np.float32)).to(
            utils.device
        )

    def process_act(self, act):
        return np.squeeze(act.clamp(-1.0, 1.0).cpu().numpy(), 0)


def redq(
    agent,
    buffer,
    train_env,
    test_env,
    num_steps=1_000_000,
    transitions_per_step=1,
    max_episode_steps=100_000,
    batch_size=512,
    tau=0.005,
    actor_lr=3e-4,
    critic_lr=3e-4,
    alpha_lr=1e-4,
    gamma=0.99,
    eval_interval=5000,
    eval_episodes=10,
    warmup_steps=1000,
    actor_clip=None,
    critic_clip=None,
    actor_l2=0.0,
    critic_l2=0.0,
    target_delay=2,
    save_interval=100_000,
    name="redq_run",
    render=False,
    save_to_disk=True,
    log_to_disk=True,
    verbosity=0,
    actor_updates_per_step=1,
    critic_updates_per_step=20,
    random_ensemble_size=2,
    init_alpha=0.1,
    infinite_bootstrap=True,
    **kwargs,
):
    """
    "Randomized Ensembled Dobule Q-Learning: Learning Fast Without a Model", Chen et al., 2020

    REDQ is an extension of the clipped double Q learning trick. To create the
    target value, we sample M critic networks from an ensemble of size N. This
    reduces the overestimation bias of the critics, and also allows us to use
    much higher replay ratios (actor_updates_per_step or critic_updates_per_step
    >> transitions_per_step). This makes REDQ very sample efficient, but really
    hurts wall clock time relative to SAC/TD3. REDQ's sample efficiency makes
    MBPO a more fair comparison, in which case it would be considered fast.
    REDQ can be applied to just about any actor-critic algorithm; we implement
    it on SAC here.
    """
    assert len(agent.critics) >= random_ensemble_size

    if save_to_disk or log_to_disk:
        save_dir = utils.make_process_dirs(name)
    if log_to_disk:
        # create tb writer, save hparams
        writer = tensorboardX.SummaryWriter(save_dir)
        writer.add_hparams(locals(), {})

    ###########
    ## SETUP ##
    ###########
    agent.to(device)
    agent.train()
    # initialize target networks
    target_agent = copy.deepcopy(agent)
    target_agent.to(device)
    for target_critic, agent_critic in zip(target_agent.critics, agent.critics):
        utils.hard_update(target_critic, agent_critic)
    target_agent.train()

    # set up optimizers
    critic_optimizer = torch.optim.Adam(
        chain(*(critic.parameters() for critic in agent.critics)),
        lr=critic_lr,
        weight_decay=critic_l2,
        betas=(0.9, 0.999),
    )
    actor_optimizer = torch.optim.Adam(
        agent.actor.parameters(),
        lr=actor_lr,
        weight_decay=actor_l2,
        betas=(0.9, 0.999),
    )
    log_alpha = torch.Tensor([math.log(init_alpha)]).to(device)
    log_alpha.requires_grad = True
    log_alpha_optimizer = torch.optim.Adam([log_alpha], lr=alpha_lr, betas=(0.5, 0.999))
    target_entropy = -train_env.action_space.shape[0]

    ###################
    ## TRAINING LOOP ##
    ###################
    # warmup the replay buffer with random actions
    run.warmup_buffer(buffer, train_env, warmup_steps, max_episode_steps)
    done = True
    steps_iter = range(num_steps)
    if verbosity:
        steps_iter = tqdm.tqdm(steps_iter)
    for step in steps_iter:
        # collect experience
        for _ in range(transitions_per_step):
            if done:
                state = train_env.reset()
                steps_this_ep = 0
                done = False
            action = agent.sample_action(state)
            next_state, reward, done, info = train_env.step(action)
            if infinite_bootstrap and steps_this_ep + 1 == max_episode_steps:
                # allow infinite bootstrapping
                done = False
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            steps_this_ep += 1
            if steps_this_ep >= max_episode_steps:
                done = True

        # critic update
        for _ in range(critic_updates_per_step):
            learn_critics(
                buffer=buffer,
                target_agent=target_agent,
                agent=agent,
                critic_optimizer=critic_optimizer,
                log_alpha=log_alpha,
                batch_size=batch_size,
                gamma=gamma,
                critic_clip=critic_clip,
                random_ensemble_size=random_ensemble_size,
            )

        # actor update
        for _ in range(actor_updates_per_step):
            learn_actor(
                buffer=buffer,
                agent=agent,
                actor_optimizer=actor_optimizer,
                log_alpha_optimizer=log_alpha_optimizer,
                target_entropy=target_entropy,
                batch_size=batch_size,
                log_alpha=log_alpha,
                gamma=gamma,
                actor_clip=actor_clip,
            )

        # move target model towards training model
        if step % target_delay == 0:
            for (target_critic, agent_critic) in zip(
                target_agent.critics, agent.critics
            ):
                utils.soft_update(target_critic, agent_critic, tau)

        if (step % eval_interval == 0) or (step == num_steps - 1):
            mean_return = run.evaluate_agent(
                agent, test_env, eval_episodes, max_episode_steps, render
            )
            if log_to_disk:
                writer.add_scalar("return", mean_return, step * transitions_per_step)

        if step % save_interval == 0 and save_to_disk:
            agent.save(save_dir)

    if save_to_disk:
        agent.save(save_dir)
    return agent


def learn_critics(
    buffer,
    target_agent,
    agent,
    critic_optimizer,
    batch_size,
    log_alpha,
    gamma,
    critic_clip,
    random_ensemble_size,
):
    per = isinstance(buffer, replay.PrioritizedReplayBuffer)
    if per:
        batch, imp_weights, priority_idxs = buffer.sample(batch_size)
        imp_weights = imp_weights.to(device)
    else:
        batch = buffer.sample(batch_size)

    # prepare transitions for models
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
    state_batch = state_batch.to(device)
    next_state_batch = next_state_batch.to(device)
    action_batch = action_batch.to(device)
    reward_batch = reward_batch.to(device)
    done_batch = done_batch.to(device)

    agent.train()

    ###################
    ## CRITIC UPDATE ##
    ###################
    alpha = torch.exp(log_alpha)
    with torch.no_grad():
        action_dist_s1 = agent.actor(next_state_batch)
        action_s1 = action_dist_s1.rsample()
        logp_a1 = action_dist_s1.log_prob(action_s1).sum(-1, keepdim=True)

        target_critic_ensemble = random.sample(
            target_agent.critics, random_ensemble_size
        )
        target_critic_ensemble_preds = (
            critic(next_state_batch, action_s1) for critic in target_critic_ensemble
        )
        target_action_value_s1 = torch.min(*target_critic_ensemble_preds)
        td_target = reward_batch + gamma * (1.0 - done_batch) * (
            target_action_value_s1 - (alpha * logp_a1)
        )

    # update critics
    critic_loss = 0.0
    for i, critic in enumerate(agent.critics):
        agent_critic_pred = critic(state_batch, action_batch)
        td_error = td_target - agent_critic_pred
        critic_loss += 0.5 * (td_error ** 2)
    if per:
        critic_loss *= imp_weights
    critic_loss = critic_loss.mean()
    critic_optimizer.zero_grad()
    critic_loss.backward()
    if critic_clip:
        torch.nn.utils.clip_grad_norm_(
            chain(*(critic.parameters() for critic in agent.critics)),
            critic_clip,
        )
    critic_optimizer.step()

    if per:
        # just using td error of the last critic here, although an average is probably better
        new_priorities = (abs(td_error) + 1e-5).cpu().detach().squeeze(1).numpy()
        buffer.update_priorities(priority_idxs, new_priorities)


def learn_actor(
    buffer,
    agent,
    actor_optimizer,
    log_alpha_optimizer,
    target_entropy,
    batch_size,
    log_alpha,
    gamma,
    actor_clip,
):
    per = isinstance(buffer, replay.PrioritizedReplayBuffer)
    if per:
        batch, *_ = buffer.sample(batch_size)
        imp_weights = imp_weights.to(device)
    else:
        batch = buffer.sample(batch_size)

    # prepare transitions for models
    state_batch, *_ = batch
    state_batch = state_batch.to(device)

    agent.train()
    alpha = torch.exp(log_alpha)

    ##################
    ## ACTOR UPDATE ##
    ##################
    dist = agent.actor(state_batch)
    agent_actions = dist.rsample()
    logp_a = dist.log_prob(agent_actions).sum(-1, keepdim=True)
    stacked_preds = torch.stack(
        [critic(state_batch, agent_actions) for critic in agent.critics], dim=0
    )
    mean_critic_pred = torch.mean(stacked_preds, dim=0)
    actor_loss = -(mean_critic_pred - (alpha.detach() * logp_a)).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    if actor_clip:
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), actor_clip)
    actor_optimizer.step()

    ##################
    ## ALPHA UPDATE ##
    ##################
    alpha_loss = (-alpha * (logp_a + target_entropy).detach()).mean()
    log_alpha_optimizer.zero_grad()
    alpha_loss.backward()
    log_alpha_optimizer.step()


def add_args(parser):
    parser.add_argument(
        "--num_steps", type=int, default=10 ** 6, help="Number of steps in training"
    )
    parser.add_argument(
        "--transitions_per_step",
        type=int,
        default=1,
        help="env transitions per training step. Defaults to 1, but will need to \
        be set higher for repaly ratios < 1",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=100000,
        help="maximum steps per episode",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="training batch size"
    )
    parser.add_argument(
        "--tau", type=float, default=0.005, help="for model parameter % update"
    )
    parser.add_argument(
        "--actor_lr", type=float, default=3e-4, help="actor learning rate"
    )
    parser.add_argument(
        "--critic_lr", type=float, default=3e-4, help="critic learning rate"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="gamma, the discount factor"
    )
    parser.add_argument(
        "--init_alpha",
        type=float,
        default=0.1,
        help="initial entropy regularization coefficeint.",
    )
    parser.add_argument(
        "--alpha_lr",
        type=float,
        default=1e-4,
        help="alpha (entropy regularization coefficeint) learning rate",
    )
    parser.add_argument(
        "--buffer_size", type=int, default=1_000_000, help="replay buffer size"
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=5000,
        help="how often to test the agent without exploration (in episodes)",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=10,
        help="how many episodes to run for when testing",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=1000, help="warmup length, in steps"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="flag to enable env rendering during training",
    )
    parser.add_argument(
        "--actor_clip",
        type=float,
        default=None,
        help="gradient clipping for actor updates",
    )
    parser.add_argument(
        "--critic_clip",
        type=float,
        default=None,
        help="gradient clipping for critic updates",
    )
    parser.add_argument(
        "--name", type=str, default="redq_run", help="dir name for saves"
    )
    parser.add_argument(
        "--actor_l2",
        type=float,
        default=0.0,
        help="L2 regularization coeff for actor network",
    )
    parser.add_argument(
        "--critic_l2",
        type=float,
        default=0.0,
        help="L2 regularization coeff for critic network",
    )
    parser.add_argument(
        "--target_delay",
        type=int,
        default=2,
        help="How many training steps to go between target network updates",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100_000,
        help="How many steps to go between saving the agent params to disk",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="verbosity > 0 displays a progress bar during training",
    )
    parser.add_argument(
        "--critic_updates_per_step",
        type=int,
        default=20,
        help="how many critic gradient updates to make per training step. The REDQ paper calls this variable G.",
    )
    parser.add_argument(
        "--actor_updates_per_step",
        type=int,
        default=1,
        help="how many actor gradient updates to make per training step",
    )
    parser.add_argument(
        "--prioritized_replay",
        action="store_true",
        help="flag that enables use of prioritized experience replay",
    )
    parser.add_argument(
        "--skip_save_to_disk",
        action="store_true",
        help="flag to skip saving agent params to disk during training",
    )
    parser.add_argument(
        "--skip_log_to_disk",
        action="store_true",
        help="flag to skip saving agent performance logs to disk during training",
    )
    parser.add_argument(
        "--log_std_low",
        type=float,
        default=-10,
        help="Lower bound for log std of action distribution.",
    )
    parser.add_argument(
        "--log_std_high",
        type=float,
        default=2,
        help="Upper bound for log std of action distribution.",
    )
    parser.add_argument(
        "--random_ensemble_size",
        type=int,
        default=2,
        help="How many random critic networks to use per TD target computation. The REDQ paper calls this variable M",
    )
    parser.add_argument(
        "--critic_ensemble_size",
        type=int,
        default=10,
        help="How many critic networks to sample from on each TD target computation. This it the total size of the critic ensemble. The REDQ paper calls this variable N",
    )
