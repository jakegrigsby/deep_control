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

from . import envs, nets, replay, run, utils, critic_searchers, grac

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TSR_CAQLAgent = grac.GRACAgent


def tsr_caql(
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
    name="tsr_caql_run",
    max_episode_steps=100_000,
    render=False,
    save_interval=100_000,
    verbosity=0,
    critic_l2=0.0,
    actor_l2=0.0,
    log_to_disk=True,
    save_to_disk=True,
    debug_logs=False,
    infinite_bootstrap=True,
    **kwargs,
):
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
            if infinite_bootstrap and (steps_this_ep + 1 == max_episode_steps):
                # allow infinite bootstrapping
                done = False
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            steps_this_ep += 1
            if steps_this_ep >= max_episode_steps:
                done = True

        learning_info = learn(
            buffer=buffer,
            agent=agent,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            target_entropy=-train_env.action_space.shape[0],
            critic_target_improvement=current_target_imp(step),
            max_critic_updates_per_step=max_critic_updates_per_step,
            batch_size=batch_size,
            gamma=gamma,
            critic_clip=critic_clip,
            actor_clip=actor_clip,
        )

        if debug_logs:
            for key, val in learning_info.items():
                writer.add_scalar(key, val.item(), step * transitions_per_step)

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
    target_entropy,
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

    def min_and_argmin(x, y, x_args, y_args):
        min_ = torch.min(x, y)
        use_x_mask = (x <= y).squeeze(1)
        argmin = y_args.clone()
        argmin[use_x_mask] = x_args[use_x_mask]
        return min_, argmin

    def max_and_argmax(x, y, x_args, y_args):
        max_ = torch.max(x, y)
        use_x_mask = (x >= y).squeeze(1)
        argmax = y_args.clone()
        argmax[use_x_mask] = x_args[use_x_mask]
        return max_, argmax

    ###################
    ## CRITIC UPDATE ##
    ###################
    with torch.no_grad():
        # sample an action as normal
        action_dist_s1 = agent.actor(next_state_batch)
        action_s1 = action_dist_s1.sample()
        action_value_s1_q1 = agent.critic1(next_state_batch, action_s1)
        action_value_s1_q2 = agent.critic2(next_state_batch, action_s1)

        # use CEM to find a higher value action
        cem_actions_s1_q1 = agent.cem.search(next_state_batch, action_s1, agent.critic1)
        cem_action_value_s1_q1 = agent.critic1(next_state_batch, cem_actions_s1_q1)
        cem_actions_s1_q2 = agent.cem.search(next_state_batch, action_s1, agent.critic2)
        cem_action_value_s1_q2 = agent.critic2(next_state_batch, cem_actions_s1_q2)
        best_q1, best_actions_q1 = max_and_argmax(
            action_value_s1_q1, cem_action_value_s1_q1, action_s1, cem_actions_s1_q1
        )
        best_q2, best_actions_q2 = max_and_argmax(
            action_value_s1_q2, cem_action_value_s1_q2, action_s1, cem_actions_s1_q2
        )
        clipped_double_q_s1, final_actions_s1 = min_and_argmin(
            best_q1, best_q2, best_actions_q1, best_actions_q2
        )
        td_target = reward_batch + gamma * (1.0 - done_batch) * clipped_double_q_s1
        y1 = agent.critic1(next_state_batch, final_actions_s1)
        y2 = agent.critic2(next_state_batch, final_actions_s1)

    learning_info = {
        "td_target": td_target.mean(),
        "clip_double_q_s1_mean": clipped_double_q_s1.mean(),
    }

    # update critics
    critic_loss_initial = None
    for critic_update in range(max_critic_updates_per_step):
        # standard bellman error
        a_critic1_pred = agent.critic1(state_batch, action_batch)
        a_critic2_pred = agent.critic2(state_batch, action_batch)
        td_error1 = td_target - a_critic1_pred
        td_error2 = td_target - a_critic2_pred

        # constraints that discourage large changes in Q(s_{t+1}, a_{t+1}),
        a1_critic1_pred = agent.critic1(next_state_batch, final_actions_s1)
        a1_critic2_pred = agent.critic2(next_state_batch, final_actions_s1)
        a1_constraint1 = y1 - a1_critic1_pred
        a1_constraint2 = y2 - a1_critic2_pred

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
    agent_actions = dist.rsample()
    logp_a = dist.log_prob(agent_actions).sum(-1, keepdim=True)
    with torch.no_grad():
        agent_action_value = torch.min(
            agent.critic1(state_batch, agent_actions),
            agent.critic2(state_batch, agent_actions),
        )
        # find higher-value actions with CEM
        cem_actions_q1 = agent.cem.search(state_batch, agent_actions, agent.critic1)
        cem_action_value_q1 = agent.critic1(state_batch, cem_actions_q1)
        cem_actions_q2 = agent.cem.search(state_batch, agent_actions, agent.critic2)
        cem_action_value_q2 = agent.critic2(state_batch, cem_actions_q2)
        cem_action_value, cem_actions = min_and_argmin(
            cem_action_value_q1, cem_action_value_q2, cem_actions_q1, cem_actions_q2
        )
    logp_cema = dist.log_prob(cem_actions).sum(-1, keepdim=True)

    # how much better are the CEM actions than the agent's?
    # clipped for rare cases where CEM actually finds a worse action...
    cem_advantage = F.relu(cem_action_value - agent_action_value).detach()
    actor_loss = -(cem_advantage * logp_cema).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    if actor_clip:
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), actor_clip)
    actor_optimizer.step()
    learning_info.update(
        {
            "cem_adv": cem_advantage.mean(),
            "actor_loss": actor_loss,
            "logp_a": logp_a.mean(),
            "logp_cema": logp_cema.mean(),
            "agent_action_value": agent_action_value.mean(),
            "cem_action_value": cem_action_value.mean(),
        }
    )

    if per:
        new_priorities = (abs(td_error1) + 1e-5).cpu().detach().squeeze(1).numpy()
        buffer.update_priorities(priority_idxs, new_priorities)

    return learning_info


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
        "--name", type=str, default="tsr_caql_run", help="dir name for saves"
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
    parser.add_argument(
        "--debug_logs",
        action="store_true",
    )
