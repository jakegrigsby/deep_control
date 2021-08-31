import argparse
import copy
import math
import os
from itertools import chain

import numpy as np
import tensorboardX
import torch
import torch.nn.functional as F
import tqdm

from . import envs, nets, replay, run, utils, critic_searchers, sac

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRACAgent(sac.SACAgent):
    def __init__(
        self,
        obs_space_size,
        act_space_size,
        log_std_low,
        log_std_high,
        actor_net_cls=nets.StochasticActor,
        critic_net_cls=nets.BigCritic,
        hidden_size=1024,
    ):
        super().__init__(
            obs_space_size=obs_space_size,
            act_space_size=act_space_size,
            log_std_low=log_std_low,
            log_std_high=log_std_high,
            actor_net_cls=actor_net_cls,
            hidden_size=hidden_size,
        )
        self.cem = critic_searchers.CEM(act_space_size, max_action=1.0)


def grac(
    agent,
    buffer,
    train_env,
    test_env,
    num_steps=1_000_000,
    transitions_per_step=1,
    max_critic_updates_per_step=20,
    critic_target_improvement_init=0.7,
    critic_target_improvement_final=0.9,
    gamma=0.99,
    batch_size=512,
    actor_lr=1e-4,
    critic_lr=1e-4,
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
    infinite_bootstrap=True,
    **kwargs,
):
    """
    Train `agent` on `train_env` using GRAC, and evaluate on `test_env`.

    GRAC: Self-Guided and Self-Regularized Actor-Critic (https://sites.google.com/view/gracdrl)

    GRAC is a combination of a stochastic policy with
    TD3-like stability improvements and CEM-based action selection
    like you'd find in Qt-Opt or CAQL.

    This is a pretty faithful reimplementation of the authors' version
    (https://github.com/stanford-iprl-lab/GRAC/blob/master/GRAC.py), with
    a couple differences:

    1) The default agent architecture and batch size are chosen for a fair
        comparison with popular SAC settings (meaning they are larger). The
        agent's action distribution is also implemented in a way that is more
        like SAC (outputting log_std of a tanh squashed normal distribution)
    2) The agent never collects experience with actions selected by CEM.
    """
    if save_to_disk or log_to_disk:
        save_dir = utils.make_process_dirs(name)
    if log_to_disk:
        # create tb writer, save hparams
        writer = tensorboardX.SummaryWriter(save_dir)
        writer.add_hparams(locals(), {})

    # no target networks!
    agent.to(device)
    agent.cem.batch_size = batch_size
    agent.train()

    # the critic target improvement ratio is annealed during training
    critic_target_imp_slope = (
        critic_target_improvement_final - critic_target_improvement_init
    ) / num_steps
    current_target_imp = lambda step: min(
        critic_target_improvement_init + critic_target_imp_slope * step,
        critic_target_improvement_final,
    )

    # set up optimizers
    critic_optimizer = torch.optim.Adam(
        chain(
            agent.critic1.parameters(),
            agent.critic2.parameters(),
        ),
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
            if infinite_bootstrap:
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
            critic_target_improvement=current_target_imp(step),
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

    ###################
    ## CRITIC UPDATE ##
    ###################

    with torch.no_grad():
        # sample an action as normal
        action_dist_s1 = agent.actor(next_state_batch)
        action_s1 = action_dist_s1.sample().clamp(-1.0, 1.0)
        # use CEM to find a higher value action
        cem_action_s1 = agent.cem.search(next_state_batch, action_s1, agent.critic2)

        # clipped double q learning using both the agent and CEM actions
        clip_double_q_a1 = torch.min(
            agent.critic1(next_state_batch, action_s1),
            agent.critic2(next_state_batch, action_s1),
        )

        clip_double_q_cema1 = torch.min(
            agent.critic1(next_state_batch, cem_action_s1),
            agent.critic2(next_state_batch, cem_action_s1),
        )

        # best_action_s1 = argmax_a(clip_double_q_a1, clip_double_q_cema1)
        better_action_mask = (clip_double_q_cema1 >= clip_double_q_a1).squeeze(1)
        best_action_s1 = action_s1.clone()
        best_action_s1[better_action_mask] = cem_action_s1[better_action_mask]

        # critic opinions of best actions that were found
        y_1 = agent.critic1(next_state_batch, best_action_s1)
        y_2 = agent.critic2(next_state_batch, best_action_s1)

        # "max min double q learning"
        max_min_s1_value = torch.max(clip_double_q_a1, clip_double_q_cema1)
        td_target = reward_batch + gamma * (1.0 - done_batch) * max_min_s1_value

    # update critics
    critic_loss_initial = None
    for critic_update in range(max_critic_updates_per_step):
        # standard bellman error
        a_critic1_pred = agent.critic1(state_batch, action_batch)
        a_critic2_pred = agent.critic2(state_batch, action_batch)
        td_error1 = td_target - a_critic1_pred
        td_error2 = td_target - a_critic2_pred

        # constraints that discourage large changes in Q(s_{t+1}, a_{t+1}),
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
        critic_loss = 0.5 * elementwise_critic_loss.mean()
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

    ##################
    ## ACTOR UPDATE ##
    ##################

    # get agent's actions in this state
    dist = agent.actor(state_batch)
    agent_actions = dist.rsample().clamp(-1.0, 1.0)
    agent_action_value = agent.critic1(state_batch, agent_actions)

    # find higher-value actions with CEM
    cem_actions = agent.cem.search(state_batch, agent_actions, agent.critic1)
    logp_cema = dist.log_prob(cem_actions).sum(-1, keepdim=True)
    cem_action_value = agent.critic1(state_batch, cem_actions)

    # how much better are the CEM actions than the agent's?
    # clipped for rare cases where CEM actually finds a worse action...
    cem_advantage = F.relu(cem_action_value - agent_action_value).detach()
    # cem_adv_coeff = 1 / |A| ; best guess here is that this is meant
    # to balance the \sum_{i}_{log\pi(cem_action)_{i}}, which can get large
    # early in training when CEM tends to find very unlikely actions
    cem_adv_coeff = 1.0 / agent_actions.shape[1]

    actor_loss = -(
        agent_action_value + cem_adv_coeff * cem_advantage * logp_cema
    ).mean()
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
        "--critic_target_improvement_init",
        type=float,
        default=0.7,
        help="Stop critic updates when loss drops by this factor. The GRAC paper calls this alpha",
    )
    parser.add_argument(
        "--critic_target_improvement_final",
        type=float,
        default=0.9,
        help="Stop critic updates when loss drops by this factor. The GRAC paper calls this alpha",
    )
