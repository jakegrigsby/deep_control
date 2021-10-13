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


def clipped_double_q(critics, s, a):
    val = torch.stack([q(s, a) for q in critics], dim=0).min(0).values
    return val


class SunriseAgent:
    def __init__(
        self,
        obs_space_size,
        act_space_size,
        log_std_low,
        log_std_high,
        ensemble_size=5,
        ucb_bonus=5.0,
        hidden_size=512,
        actor_net_cls=nets.StochasticActor,
        critic_net_cls=nets.BigCritic,
    ):
        self.actors = [
            actor_net_cls(
                obs_space_size,
                act_space_size,
                log_std_low,
                log_std_high,
                hidden_size=hidden_size,
                dist_impl="pyd",
            )
            for _ in range(ensemble_size)
        ]
        self.critics = [
            [
                critic_net_cls(obs_space_size, act_space_size, hidden_size=hidden_size)
                for _ in range(2)
            ]
            for _ in range(ensemble_size)
        ]
        # SUNRISE Eq 6 lambda variable
        self.ucb_bonus = ucb_bonus

    @property
    def actor_params(self):
        return chain(*(actor.parameters() for actor in self.actors))

    @property
    def critic_params(self):
        critic_params = []
        for critic_pair in self.critics:
            for critic in critic_pair:
                critic_params.append(critic.parameters())
        return chain(*critic_params)

    def to(self, device):
        for i, (actor, critics) in enumerate(zip(self.actors, self.critics)):
            for j, critic in enumerate(critics):
                self.critics[i][j] = critic.to(device)
            self.actors[i] = actor.to(device)

    def eval(self):
        for actor, critics in zip(self.actors, self.critics):
            for critic in critics:
                critic.eval()
            actor.eval()

    def train(self):
        for actor, critics in zip(self.actors, self.critics):
            for critic in critics:
                critic.train()
            actor.train()

    def save(self, path):
        for i, (actor, critics) in enumerate(zip(self.actors, self.critics)):
            actor_path = os.path.join(path, f"actor{i}.pt")
            torch.save(actor.state_dict(), actor_path)
            for j, critic in enumerate(critics):
                critic_path = os.path.join(path, f"critic{i}_{j}.pt")
                torch.save(critic.state_dict(), critic_path)

    def load(self, path):
        for i, (actor, critics) in enumerate(zip(self.actors, self.critics)):
            actor_path = os.path.join(path, f"actor{i}.pt")
            actor.load_state_dict(torch.load(actor_path))
            for j, critic in enumerate(critics):
                critic_path = os.path.join(path, f"critic{i}_{j}.pt")
                critic.load_state_dict(torch.load(critic_path))

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

    def sample_action(self, state, from_cpu=True):
        # training (exploration) forward:
        if from_cpu:
            state = self.process_state(state)
        self.eval()
        # a = argmax_a Q_mean(s, a) + \lambda * Q_std(s, a)
        with torch.no_grad():
            # generate a candidate action from each actor
            act_candidates = torch.stack(
                [actor.forward(state).sample().squeeze(0) for actor in self.actors],
                dim=0,
            )
            # evaluate each action on the min of each pair of critics, for NxN q vals
            q_inputs = (state.repeat(len(act_candidates), 1), act_candidates)
            q_vals = torch.stack(
                [clipped_double_q(critics, *q_inputs) for critics in self.critics],
                dim=0,
            )
            # use mean and std over the critic axis to compute ucb term
            ucb_val = q_vals.mean(0) + self.ucb_bonus * q_vals.std(0)
            # argmax over action axis
            argmax_ucb_val = torch.argmax(ucb_val)
            act = act_candidates[argmax_ucb_val].unsqueeze(0)
        self.train()
        if from_cpu:
            act = self.process_act(act)
        return act

    def process_state(self, state):
        return torch.from_numpy(np.expand_dims(state, 0).astype(np.float32)).to(
            utils.device
        )

    def process_act(self, act):
        return np.squeeze(act.clamp(-1.0, 1.0).cpu().numpy(), 0)


def sunrise(
    agent,
    buffer,
    train_env,
    test_env,
    num_steps=1_000_000,
    transitions_per_step=1,
    max_episode_steps=100_000,
    batch_size=512,
    tau=0.005,
    actor_lr=1e-4,
    critic_lr=1e-4,
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
    name="sunrise_run",
    render=False,
    save_to_disk=True,
    log_to_disk=True,
    verbosity=0,
    gradient_updates_per_step=1,
    init_alpha=0.1,
    weighted_bellman_temp=20.0,
    infinite_bootstrap=True,
    **kwargs,
):
    """
    "SUNRISE: A Simple Unified Framework for Ensemble Learning
    in Deep Reinforcement Learning", Lee et al., 2020.

    SUNRISE extends SAC by adding:
    1. An ensemble of actors and critics
        - Less explicitly focused on the bias reduction of the Q
          network than something like REDQ or Maxmin Q.
    2. UCB Exploration
        - Leverage ensemble of critics to encourage exploration
          in uncertain states.
    3. Weighted Bellman Backups
        - Similar motivation to DisCor but without the extra error
          predicting networks. Much simpler.
    4. Ensembled Inference
        - Each actor trains to maximize one critic, but their action
          distributions are averaged at inference time. This is slower
          than other ensembling methods, where the actor trains on all
          of the critics.

    Reference: https://arxiv.org/abs/2007.04938
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
    target_agent = copy.deepcopy(agent)
    # initialize all of the critic targets
    for target_critics, agent_critics in zip(target_agent.critics, agent.critics):
        for target_critic, agent_critic in zip(target_critics, agent_critics):
            utils.hard_update(target_critic, agent_critic)
    target_agent.train()

    critic_optimizer = torch.optim.Adam(
        agent.critic_params,
        lr=critic_lr,
        weight_decay=critic_l2,
        betas=(0.9, 0.999),
    )

    actor_optimizer = torch.optim.Adam(
        agent.actor_params,
        lr=actor_lr,
        weight_decay=actor_l2,
        betas=(0.9, 0.999),
    )

    # create a separate entropy coeff for each agent in the ensemble
    log_alphas, alpha_optimizers = [], []
    for _ in range(len(agent.actors)):
        log_alpha = torch.Tensor([math.log(init_alpha)]).to(device)
        log_alpha.requires_grad = True
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=alpha_lr, betas=(0.5, 0.999))
        log_alphas.append(log_alpha)
        alpha_optimizers.append(alpha_optimizer)
    target_entropy = -train_env.action_space.shape[0]

    ###################
    ## TRAINING LOOP ##
    ###################
    run.warmup_buffer(buffer, train_env, warmup_steps, max_episode_steps)
    done = True
    steps_iter = range(num_steps)
    if verbosity:
        steps_iter = tqdm.tqdm(steps_iter)
    for step in steps_iter:
        for _ in range(transitions_per_step):
            if done:
                state = train_env.reset()
                steps_this_ep = 0
                done = False
            # UCB Exploration implemented in agent.sample_action method
            action = agent.sample_action(state)
            next_state, reward, done, info = train_env.step(action)
            if infinite_bootstrap and steps_this_ep + 1 == max_episode_steps:
                done = False
            # We are skipping the transition mask idea from the paper
            # (which is equivalent to setting \Beta = 1. in Alg 1 line 7)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            steps_this_ep += 1
            if steps_this_ep >= max_episode_steps:
                done = True

        for _ in range(gradient_updates_per_step):
            learn_sunrise(
                buffer=buffer,
                target_agent=target_agent,
                agent=agent,
                critic_optimizer=critic_optimizer,
                batch_size=batch_size,
                gamma=gamma,
                critic_clip=critic_clip,
                actor_optimizer=actor_optimizer,
                alpha_optimizers=alpha_optimizers,
                target_entropy=target_entropy,
                log_alphas=log_alphas,
                actor_clip=actor_clip,
                weighted_bellman_temp=weighted_bellman_temp,
            )

        if step % target_delay == 0:
            for (target_critics, agent_critics) in zip(
                target_agent.critics, agent.critics
            ):
                for target_critic, agent_critic in zip(target_critics, agent_critics):
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


def learn_sunrise(
    buffer,
    target_agent,
    agent,
    critic_optimizer,
    batch_size,
    gamma,
    critic_clip,
    actor_optimizer,
    alpha_optimizers,
    target_entropy,
    log_alphas,
    actor_clip,
    weighted_bellman_temp,
):
    per = isinstance(buffer, replay.PrioritizedReplayBuffer)
    if per:
        batch, imp_weights, priority_idxs = buffer.sample(batch_size)
        imp_weights = imp_weights.to(device)
    else:
        batch = buffer.sample(batch_size)

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
        # compute weighted bellman coeffs using SUNRISE Eq 5
        target_q_std = torch.stack(
            [
                clipped_double_q(critics, state_batch, action_batch)
                for critics in target_agent.critics
            ],
            dim=0,
        ).std(0)
        weights = torch.sigmoid(-target_q_std * weighted_bellman_temp) + 0.5

    # now we compute the MSBE of each critic relative to its own target
    critic_loss = 0.0
    total_abs_td_error = 0.0
    for i, critic_pair in enumerate(agent.critics):
        with torch.no_grad():
            # sample an action from actor i
            action_dist_s1 = agent.actors[i](next_state_batch)
            action_s1 = action_dist_s1.rsample()
            logp_a1 = action_dist_s1.log_prob(action_s1).sum(-1, keepdim=True)
            # generate target network's Q(s', a') prediction
            target_q_s1 = clipped_double_q(
                target_agent.critics[i], next_state_batch, action_s1
            )
            # compute TD target value for this critic
            td_target = reward_batch + gamma * (1.0 - done_batch) * (
                target_q_s1 - (log_alphas[i].exp() * logp_a1)
            )
        # compute MSBE for this critic
        for critic in critic_pair:
            agent_critic_pred = critic(state_batch, action_batch)
            td_error = td_target - agent_critic_pred
            # SUNRISE Eq 4
            critic_loss += 0.5 * (td_error ** 2)
            total_abs_td_error += abs(td_error)
    if per:
        # priority weights can be used in addition to bellman backup weights.
        # this is mentioned in the paper for Rainbow DQN but could apply here.
        critic_loss *= imp_weights
    critic_loss = (critic_loss * weights).mean() / (i + 1)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    if critic_clip:
        torch.nn.utils.clip_grad_norm_(agent.critic_params, critic_clip)
    critic_optimizer.step()

    ##########################
    ## ACTOR + ALPHA UPDATE ##
    ##########################
    actor_loss = 0.0
    for i, actor in enumerate(agent.actors):
        # sample an action for this actor
        dist = actor(state_batch)
        agent_actions = dist.rsample()
        logp_a = dist.log_prob(agent_actions).sum(-1, keepdim=True)
        # use corresponding critic to evaluate this action
        critic_pred = clipped_double_q(agent.critics[i], state_batch, agent_actions)
        actor_loss += -(critic_pred - (log_alphas[i].exp().detach() * logp_a)).mean()

        # each agent in the ensemble has its own alpha (entropy) coeff,
        # which we update inside the actor loop so we can use the logp_a terms
        alpha_loss = (-log_alphas[i].exp() * (logp_a + target_entropy).detach()).mean()
        alpha_optimizers[i].zero_grad()
        alpha_loss.backward()
        alpha_optimizers[i].step()

    # actor gradient step
    actor_optimizer.zero_grad()
    actor_loss.backward()
    if actor_clip:
        torch.nn.utils.clip_grad_norm_(agent.actor_params, actor_clip)
    actor_optimizer.step()

    if per:
        ensemble_size = float(len(agent.actors))
        avg_abs_td_error = total_abs_td_error / (ensemble_size * 2)
        new_priorities = (avg_abs_td_error + 1e-5).cpu().detach().squeeze(1).numpy()
        buffer.update_priorities(priority_idxs, new_priorities)


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
        "--gradient_updates_per_step",
        type=int,
        default=1,
        help="how many gradient updates to make per training step",
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
        "--ensemble_size",
        type=int,
        default=5,
        help="SUNRISE ensemble size",
    )
    parser.add_argument(
        "--ucb_bonus",
        type=float,
        default=5.0,
        help="coeff for std term in ucb exploration. higher values prioritize exploring uncertain actions",
    )
    parser.add_argument(
        "--weighted_bellman_temp",
        type=float,
        default=20.0,
        help="temperature in sunrise's weight adjustment. See equation 5 of the sunrise paper",
    )
