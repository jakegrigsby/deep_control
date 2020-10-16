import argparse
import copy
import math
import os
import time
from itertools import chain

import gym
import numpy as np
import tensorboardX
import torch
import torch.nn.functional as F
import tqdm

from . import envs, nets, replay, run, utils, critic_searchers, sac

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRACAgent(sac.SACAgent):
    def __init__(self, obs_space_size, act_space_size, *args):
        super().__init__(obs_space_size, act_space_size, None, None)
        self.actor = nets.GracBaselineActor(obs_space_size, act_space_size)
        self.cem = critic_searchers.CEM(act_space_size, max_action=1.)

def grac(
    agent,
    buffer,
    train_env,
    test_env,
    num_steps=1_000_000,
    transitions_per_step=1,
    max_critic_updates_per_step=10,
    critic_target_improvement=0.75,
    gamma=0.99,
    batch_size=512,
    actor_lr=3e-4,
    critic_lr=3e-4,
    eval_interval=5000,
    eval_episodes=10,
    warmup_steps=1000,
    actor_clip=None,
    critic_clip=None,
    name="grac_run",
    max_episode_steps=100_000,
    render=False,
    save_interval=100_000,
    verbosity=0,
    critic_l2=0.0,
    actor_l2=0.0,
    log_to_disk=True,
    save_to_disk=True,
    **kwargs,
):
    if save_to_disk or log_to_disk:
        save_dir = utils.make_process_dirs(name)
    if log_to_disk:
        # create tb writer, save hparams
        writer = tensorboardX.SummaryWriter(save_dir)
        writer.add_hparams(locals(), {})

    agent.to(device)
    agent.cem.batch_size = batch_size
    agent.train()

    # set up optimizers
    critic_optimizer = torch.optim.Adam(
        chain(agent.critic1.parameters(), agent.critic2.parameters(),),
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

    # warmup the replay buffer with random actions
    run.warmup_buffer(buffer, train_env, warmup_steps, max_episode_steps)

    steps_iter = range(num_steps)
    if verbosity:
        steps_iter = tqdm.tqdm(steps_iter)

    done = True
    for step in steps_iter:
        # collect experience
        for _ in range(transitions_per_step):
            if done:
                state = train_env.reset()
                steps_this_ep = 0
                done = False
            action = agent.sample_action(state)
            next_state, reward, done, info = train_env.step(action)
            # allow infinite bootstrapping
            if steps_this_ep + 1 == max_episode_steps:
                done = False
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            steps_this_ep += 1
            if steps_this_ep >= max_episode_steps:
                done = True
        learn(
            buffer=buffer,
            agent=agent,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            critic_target_improvement=critic_target_improvement,
            max_critic_updates_per_step=max_critic_updates_per_step,
            batch_size=batch_size,
            gamma=gamma,
            critic_clip=critic_clip,
            actor_clip=actor_clip,
        )

        if step % eval_interval == 0 or step == num_steps - 1:
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


def learn(
    buffer,
    agent,
    actor_optimizer,
    critic_optimizer,
    critic_target_improvement,
    max_critic_updates_per_step,
    batch_size,
    gamma,
    critic_clip,
    actor_clip,
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

    with torch.no_grad():
        action_dist_s1 = agent.actor(next_state_batch)
        action_s1 = action_dist_s1.sample().clamp(-1., 1.)
        cem_action_s1 = agent.cem.search(next_state_batch, action_s1, agent.critic2)

        clip_double_q_a1 = torch.min(
            agent.critic1(next_state_batch, action_s1),
            agent.critic2(next_state_batch, action_s1),
        )

        clip_double_q_cema1 = torch.min(
            agent.critic1(next_state_batch, cem_action_s1),
            agent.critic2(next_state_batch, cem_action_s1),
        )

        better_action_mask = (clip_double_q_cema1 >= clip_double_q_a1).squeeze(1)
        best_action_s1 = action_s1.clone()
        best_action_s1[better_action_mask] = cem_action_s1[better_action_mask]
        y_1 = agent.critic1(next_state_batch, best_action_s1)
        y_2 = agent.critic2(next_state_batch, best_action_s1)

        max_min_s1_value = torch.max(clip_double_q_a1, clip_double_q_cema1)
        td_target = reward_batch + gamma * (1.0 - done_batch) * max_min_s1_value

    # update critics
    critic_loss_initial = None
    for critic_update in range(max_critic_updates_per_step):
        a_critic1_pred = agent.critic1(state_batch, action_batch)
        a_critic2_pred = agent.critic2(state_batch, action_batch)
        td_error1 = td_target - a_critic1_pred
        td_error2 = td_target - a_critic2_pred

        a1_critic1_pred = agent.critic1(next_state_batch, best_action_s1)
        a1_critic2_pred = agent.critic2(next_state_batch, best_action_s1)
        a1_constraint1 = y_1 - a1_critic1_pred
        a1_constraint2 = y_2 - a1_critic2_pred

        elementwise_critic_loss = (
            (td_error1 ** 2)
            + (td_error2 ** 2)
            + (a1_constraint1 ** 2)
            + (a1_constraint2 ** 2)
        )
        if per:
            elementwise_loss *= imp_weights
        critic_loss = .5 * elementwise_critic_loss.mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        if critic_clip:
            torch.nn.utils.clip_grad_norm_(
                chain(agent.critic1.parameters(), agent.critic2.parameters()),
                critic_clip,
            )
        critic_optimizer.step()
        if critic_update == 0:
            critic_loss_initial = critic_loss
        elif critic_loss <= critic_target_improvement * critic_loss_initial:
            break

    # actor update
    dist = agent.actor(state_batch)
    agent_actions = dist.rsample()
    agent_action_value = agent.critic1(state_batch, agent_actions)

    cem_actions = agent.cem.search(state_batch, agent_actions, agent.critic1)
    logp_cema = dist.log_prob(cem_actions).sum(-1, keepdim=True)
    cem_action_value = agent.critic1(state_batch, cem_actions)

    agent_value_gap = torch.max(
            cem_action_value - agent_action_value, 
            torch.zeros_like(agent_action_value),
        ).detach()
    
    actor_loss = -(agent_action_value + (1.0 / agent_actions.shape[1]) * agent_value_gap * logp_cema).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    if actor_clip:
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), actor_clip)
    actor_optimizer.step()

    if per:
        new_priorities = (abs(td_error1) + 1e-5).cpu().detach().squeeze(1).numpy()
        buffer.update_priorities(priority_idxs, new_priorities)


def add_args(parser):
    parser.add_argument(
        "--num_steps", type=int, default=10 ** 6, help="number of steps in training"
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
        "--actor_lr", type=float, default=1e-4, help="actor learning rate"
    )
    parser.add_argument(
        "--critic_lr", type=float, default=1e-4, help="critic learning rate"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="gamma, the discount factor"
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
        "--name", type=str, default="grac_run", help="dir name for saves"
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
        "--max_critic_updates_per_step",
        type=int,
        default=10,
        help="Max critic updates to make per step. The GRAC paper calls this K",
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
        "--critic_target_improvement",
        type=float,
        default=0.75,
        help="Stop critic updates when loss drops by this factor. The GRAC paper calls this alpha",
    )
