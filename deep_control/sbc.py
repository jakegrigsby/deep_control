import argparse
import os
from itertools import chain
import random

import numpy as np
import tensorboardX
import torch
import tqdm

from . import envs, nets, replay, run, utils, device


class SBCAgent:
    def __init__(
        self,
        obs_space_size,
        act_space_size,
        log_std_low,
        log_std_high,
        ensemble_size=5,
        actor_net_cls=nets.StochasticActor,
        hidden_size=1024,
        beta_dist=False,
    ):
        self.actors = [
            actor_net_cls(
                obs_space_size,
                act_space_size,
                log_std_low,
                log_std_high,
                dist_impl="beta" if beta_dist else "pyd",
                hidden_size=hidden_size,
            )
            for _ in range(ensemble_size)
        ]

    def to(self, device):
        for i, actor in enumerate(self.actors):
            self.actors[i] = actor.to(device)

    def eval(self):
        for actor in self.actors:
            actor.eval()

    def train(self):
        for actor in self.actors:
            actor.train()

    def save(self, path):
        for i, actor in enumerate(self.actors):
            actor_path = os.path.join(path, f"actor{i}.pt")
            torch.save(actor.state_dict(), actor_path)

    def load(self, path):
        for i, actor in enumerate(self.actors):
            actor_path = os.path.join(path, f"actor{i}.pt")
            actor.load_state_dict(torch.load(actor_path))

    def forward(self, state, from_cpu=True):
        # evaluation forward:
        # take the average of the mean of each
        # actor's distribution.
        if from_cpu:
            state = self.process_state(state)
        self.eval()
        with torch.no_grad():
            act = torch.stack(
                [actor.forward(state).mean for actor in self.actors], dim=0
            ).mean(0)
        self.train()
        if from_cpu:
            act = self.process_act(act)
        return act

    def process_state(self, state):
        return torch.from_numpy(state).unsqueeze(0).float().to(device)

    def process_act(self, act):
        return act.clamp(-1.0, 1.0).cpu().squeeze(0).numpy()


def sbc(
    agent,
    buffer,
    test_env,
    num_steps_offline=1_000_000,
    batch_size=256,
    log_prob_clip=None,
    max_episode_steps=100_000,
    actor_lr=1e-4,
    eval_interval=5000,
    eval_episodes=10,
    actor_clip=None,
    actor_l2=0.0,
    save_interval=100_000,
    name="sbc_run",
    render=False,
    save_to_disk=True,
    log_to_disk=True,
    verbosity=0,
    **kwargs,
):
    """
    Stochastic Behavioral Cloning

    A simple approach to offline RL that learns to emulate the
    behavior dataset in a supervised way. Uses the stochastic actor
    from SAC, and adds some basic ensembling to improve performance
    and make this a reasonable baseline.

    For examples of how to set up and run offline RL in deep_control,
    see examples/d4rl/sbc.py
    """
    if save_to_disk or log_to_disk:
        save_dir = utils.make_process_dirs(name)
    if log_to_disk:
        writer = tensorboardX.SummaryWriter(save_dir)
        writer.add_hparams(locals(), {})

    ###########
    ## SETUP ##
    ###########
    agent.to(device)
    agent.train()

    actor_optimizer = torch.optim.Adam(
        chain(*(actor.parameters() for actor in agent.actors)),
        lr=actor_lr,
        weight_decay=actor_l2,
        betas=(0.9, 0.999),
    )

    ###################
    ## TRAINING LOOP ##
    ###################

    steps_iter = range(num_steps_offline)
    if verbosity:
        steps_iter = tqdm.tqdm(steps_iter)
    for step in steps_iter:
        learn_sbc(
            buffer=buffer,
            agent=agent,
            batch_size=batch_size,
            actor_optimizer=actor_optimizer,
            actor_clip=actor_clip,
            log_prob_clip=log_prob_clip,
        )

        if (step % eval_interval == 0) or (step == num_steps_offline - 1):
            mean_return = run.evaluate_agent(
                agent, test_env, eval_episodes, max_episode_steps, render
            )
            if log_to_disk:
                writer.add_scalar("return", mean_return, step)

        if step % save_interval == 0 and save_to_disk:
            agent.save(save_dir)

    if save_to_disk:
        agent.save(save_dir)

    return agent


def learn_sbc(
    buffer,
    agent,
    batch_size,
    actor_optimizer,
    actor_clip,
    log_prob_clip,
):
    agent.train()

    #############################
    ## SUPERVISED ACTOR UPDATE ##
    #############################

    actor_loss = 0.0
    for actor in agent.actors:
        # sample a fresh batch of data to keep the ensemble unique
        batch = buffer.sample(batch_size)
        state_batch, action_batch, *_ = batch
        state_batch = state_batch.to(device)
        action_batch = action_batch.to(device)

        # maximize the probability that the agent takes the demonstration's action in this state
        dist = actor(state_batch)
        logp_demo_act = dist.log_prob(action_batch).sum(-1, keepdim=True)
        if log_prob_clip:
            logp_demo_act = logp_demo_act.clamp(-log_prob_clip, log_prob_clip)
        actor_loss += -logp_demo_act.mean()

    # actor gradient step
    actor_optimizer.zero_grad()
    actor_loss.backward()
    if actor_clip:
        torch.nn.utils.clip_grad_norm_(
            chain(*(actor.parameters() for actor in agent.actors)), actor_clip
        )
    actor_optimizer.step()


def add_args(parser):
    parser.add_argument(
        "--num_steps_offline",
        type=int,
        default=10 ** 6,
        help="Number of offline training steps",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=100000,
        help="maximum steps per episode",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="training batch size"
    )
    parser.add_argument(
        "--actor_lr", type=float, default=3e-4, help="actor learning rate"
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
        "--name", type=str, default="redq_run", help="dir name for saves"
    )
    parser.add_argument(
        "--actor_l2",
        type=float,
        default=0.0,
        help="L2 regularization coeff for actor network",
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
        "--ensemble_size",
        type=int,
        default=5,
        help="actor ensemble size",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=1024,
        help="actor network hidden dim",
    )
