import argparse
import copy
import math
import os
from itertools import chain

import numpy as np
import tensorboardX
import torch
import torch.nn.functional as F
import torch.distributions as pyd
import tqdm

from . import envs, nets, replay, run, utils, device, sac
from deep_control.adv_estimator import AdvantageEstimator, AdvEstimatorFilter


class AWACAgent(sac.SACAgent):
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
            obs_space_size,
            act_space_size,
            log_std_low,
            log_std_high,
            actor_net_cls,
            critic_net_cls,
            hidden_size=hidden_size,
        )
        self.actor.dist_impl = "pyd"


def awac(
    agent,
    buffer,
    train_env,
    test_env,
    num_steps_offline=25_000,
    num_steps_online=500_000,
    gradient_updates_per_step=1,
    transitions_per_online_step=1,
    max_episode_steps=100_000,
    actor_per=True,
    batch_size=1024,
    tau=0.005,
    beta=1.0,
    crr_function="binary",
    adv_method="mean",
    adv_method_n=4,
    actor_lr=1e-4,
    critic_lr=1e-4,
    gamma=0.99,
    eval_interval=5000,
    eval_episodes=10,
    warmup_steps=1000,
    actor_clip=None,
    critic_clip=None,
    actor_l2=0.0,
    critic_l2=0.0,
    target_delay=2,
    actor_delay=1,
    save_interval=100_000,
    name="awac_run",
    render=False,
    save_to_disk=True,
    log_to_disk=True,
    verbosity=0,
    infinite_bootstrap=True,
    **kwargs,
):

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
    utils.hard_update(target_agent.critic1, agent.critic1)
    utils.hard_update(target_agent.critic2, agent.critic2)
    target_agent.train()
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
    # set up adv filter
    adv_estimator = AdvantageEstimator(
        agent.actor, [agent.critic1, agent.critic2], method=adv_method, n=adv_method_n
    )
    adv_filter = AdvEstimatorFilter(adv_estimator, crr_function, beta=beta)

    ###################
    ## TRAINING LOOP ##
    ###################

    total_steps = num_steps_offline + num_steps_online
    steps_iter = range(total_steps)
    if verbosity:
        steps_iter = tqdm.tqdm(steps_iter)

    done = True
    for step in steps_iter:

        if step > num_steps_offline:
            # collect online experience
            for _ in range(transitions_per_online_step):
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

        for _ in range(gradient_updates_per_step):
            learn_awac(
                buffer=buffer,
                target_agent=target_agent,
                agent=agent,
                adv_filter=adv_filter,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                batch_size=batch_size,
                gamma=gamma,
                critic_clip=critic_clip,
                actor_clip=actor_clip,
                update_policy=step % actor_delay == 0,
                actor_per=actor_per,
            )

            # move target model towards training model
            if step % target_delay == 0:
                utils.soft_update(target_agent.critic1, agent.critic1, tau)
                utils.soft_update(target_agent.critic2, agent.critic2, tau)

        if (step % eval_interval == 0) or (step == total_steps - 1):
            mean_return = run.evaluate_agent(
                agent, test_env, eval_episodes, max_episode_steps, render
            )
            if log_to_disk:
                writer.add_scalar(
                    "return", mean_return, step * transitions_per_online_step
                )

        if step % save_interval == 0 and save_to_disk:
            agent.save(save_dir)

    if save_to_disk:
        agent.save(save_dir)

    return agent


def learn_awac(
    buffer,
    target_agent,
    agent,
    adv_filter,
    actor_optimizer,
    critic_optimizer,
    batch_size,
    gamma,
    critic_clip,
    actor_clip,
    update_policy=True,
    actor_per=True,
):
    if actor_per:
        assert isinstance(buffer, replay.PrioritizedReplayBuffer)
        # sample with priorities for actor update
        actor_batch, *_ = buffer.sample(batch_size)
        # critic samples uniformly to find new high-adv experience
        critic_batch, priority_idxs = buffer.sample_uniform(batch_size)
    else:
        batch = buffer.sample(batch_size)
        actor_batch = batch
        critic_batch = batch

    agent.train()
    ###################
    ## CRITIC UPDATE ##
    ###################
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = critic_batch
    state_batch = state_batch.to(device)
    next_state_batch = next_state_batch.to(device)
    action_batch = action_batch.to(device)
    reward_batch = reward_batch.to(device)
    done_batch = done_batch.to(device)

    with torch.no_grad():
        action_dist_s1 = agent.actor(next_state_batch)
        action_s1 = action_dist_s1.rsample()
        logp_a1 = action_dist_s1.log_prob(action_s1).sum(-1, keepdim=True)
        target_action_value_s1 = torch.min(
            target_agent.critic1(next_state_batch, action_s1),
            target_agent.critic2(next_state_batch, action_s1),
        )
        td_target = reward_batch + gamma * (1.0 - done_batch) * target_action_value_s1

    # update critics
    agent_critic1_pred = agent.critic1(state_batch, action_batch)
    agent_critic2_pred = agent.critic2(state_batch, action_batch)
    td_error1 = td_target - agent_critic1_pred
    td_error2 = td_target - agent_critic2_pred
    critic_loss = 0.5 * (td_error1 ** 2 + td_error2 ** 2)
    critic_loss = critic_loss.mean()
    critic_optimizer.zero_grad()
    critic_loss.backward()
    if critic_clip:
        torch.nn.utils.clip_grad_norm_(
            chain(agent.critic1.parameters(), agent.critic2.parameters()), critic_clip
        )
    critic_optimizer.step()

    if actor_per:
        with torch.no_grad():
            adv = adv_filter.adv_estimator(state_batch, action_batch)
        new_priorities = (F.relu(adv) + 1e-5).cpu().detach().squeeze(1).numpy()
        buffer.update_priorities(priority_idxs, new_priorities)

    if update_policy:
        ##################
        ## ACTOR UPDATE ##
        ##################
        state_batch, *_ = actor_batch
        state_batch = state_batch.to(device)

        dist = agent.actor(state_batch)
        actions = dist.sample()
        logp_a = dist.log_prob(actions).sum(-1, keepdim=True)
        with torch.no_grad():
            filtered_adv = adv_filter(state_batch, actions)
        actor_loss = -(logp_a * filtered_adv).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        if actor_clip:
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), actor_clip)
        actor_optimizer.step()


def add_args(parser):
    parser.add_argument(
        "--num_steps_offline",
        type=int,
        default=500_000,
        help="Number of steps of offline learning",
    )
    parser.add_argument(
        "--num_steps_online",
        type=int,
        default=50_000,
        help="Number of steps of online learning",
    )
    parser.add_argument(
        "--transitions_per_online_step",
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
        "--batch_size", type=int, default=1024, help="training batch size"
    )
    parser.add_argument(
        "--tau", type=float, default=0.005, help="for model parameter % update"
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
        "--name", type=str, default="awac_run", help="dir name for saves"
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
        help="How many steps to go between target network updates",
    )
    parser.add_argument(
        "--actor_delay",
        type=int,
        default=1,
        help="How many steps to go between actor updates",
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
        "--gradient_updates_per_step",
        type=int,
        default=1,
        help="how many gradient updates to make per training step",
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
        "--beta",
        type=float,
        default=1.0,
        help="Lambda variable from AWAC actor update and Beta from CRR",
    )
    parser.add_argument(
        "--crr_function",
        type=str,
        default="binary",
        choices=["binary", "exp", "exp_norm"],
        help="Approach for adjusting advantage weights",
    )
    parser.add_argument(
        "--adv_method",
        type=str,
        default="mean",
        help="Approach for estimating the advantage function. Choices include {'max', 'mean'}.",
    )
    parser.add_argument(
        "--adv_method_n",
        type=int,
        default=4,
        help="How many actions to sample from the policy when estimating the advantage. CRR uses 4.",
    )
