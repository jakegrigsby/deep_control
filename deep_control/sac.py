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

from . import envs, nets, replay, run, utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SACAgent:
    def __init__(
        self, obs_space_size, act_space_size, max_action, log_std_low, log_std_high
    ):
        self.actor = nets.StochasticActor(
            obs_space_size, act_space_size, max_action, log_std_low, log_std_high
        )
        self.critic1 = nets.BaselineCritic(obs_space_size, act_space_size)
        self.critic2 = nets.BaselineCritic(obs_space_size, act_space_size)
        self.max_act = max_action

    def to(self, device):
        self.actor = self.actor.to(device)
        self.critic1 = self.critic1.to(device)
        self.critic2 = self.critic2.to(device)

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def save(self, path):
        actor_path = os.path.join(path, "actor.pt")
        critic1_path = os.path.join(path, "critic1.pt")
        critic2_path = os.path.join(path, "critic2.pt")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic1.state_dict(), critic1_path)
        torch.save(self.critic2.state_dict(), critic2_path)

    def load(self, path):
        actor_path = os.path.join(path, "actor.pt")
        critic1_path = os.path.join(path, "critic1.pt")
        critic2_path = os.path.join(path, "critic2.pt")
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic1.load_state_dict(torch.load(critic1_path))
        self.critic2.load_state_dict(torch.load(critic2_path))

    def forward(self, state):
        state = self.process_state(state)
        self.actor.eval()
        with torch.no_grad():
            act_dist = self.actor.forward(state)
            act = act_dist.mean
        self.actor.train()
        return self.process_act(act)

    def sample_action(self, state):
        state = self.process_state(state)
        with torch.no_grad():
            act_dist = self.actor.forward(state)
            act = act_dist.sample()
        act = self.process_act(act)
        return act

    def process_state(self, state):
        return torch.from_numpy(np.expand_dims(state, 0).astype(np.float32)).to(
            utils.device
        )

    def process_act(self, act):
        return np.squeeze(act.cpu().numpy(), 0)


class SACDAgent(SACAgent):
    def __init__(self, obs_space_size, act_space_size):
        self.actor = nets.BaselineDiscreteActor(obs_space_size, act_space_size)
        self.critic1 = nets.BaselineDiscreteCritic(obs_space_size, act_space_size)
        self.critic2 = nets.BaselineDiscreteCritic(obs_space_size, act_space_size)


def sac(
    agent,
    buffer,
    train_env,
    test_env,
    num_steps=1_000_000,
    transitions_per_step=1,
    max_episode_steps=100_000,
    batch_size=128,
    tau=0.005,
    actor_lr=1e-4,
    critic_lr=1e-3,
    alpha_lr=1e-4,
    gamma=0.99,
    eval_interval=5000,
    eval_episodes=10,
    warmup_steps=1000,
    actor_clip=None,
    critic_clip=None,
    actor_l2=0.0,
    critic_l2=0.0,
    delay=1,
    save_interval=100_000,
    name="sac_run",
    render=False,
    save_to_disk=True,
    log_to_disk=True,
    verbosity=0,
    gradient_updates_per_step=1,
    init_alpha=0.1,
    discrete_actions=False,
    **kwargs,
):
    """
    Train `agent` on `env` with Soft Actor Critic algorithm.

    Reference: https://arxiv.org/abs/1801.01290 and https://arxiv.org/abs/1812.05905

    Also supports discrete action spaces (ref: https://arxiv.org/abs/1910.07207)
    """

    if save_to_disk or log_to_disk:
        save_dir = utils.make_process_dirs(name)
    if log_to_disk:
        # create tb writer, save hparams
        writer = tensorboardX.SummaryWriter(save_dir)
        writer.add_hparams(locals(), {})

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

    log_alpha = torch.Tensor([math.log(init_alpha)]).to(device)
    log_alpha.requires_grad = True
    log_alpha_optimizer = torch.optim.Adam([log_alpha], lr=alpha_lr, betas=(0.5, 0.999))

    if not discrete_actions:
        # continuous action learning
        target_entropy = -train_env.action_space.shape[0]
        learn_fn = learn
    else:
        # discrete action learning
        target_entropy = -math.log(1.0 / train_env.action_space.shape[0]) * 0.98
        learn_fn = learn_discrete

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
            # allow infinite bootstrapping
            if steps_this_ep + 1 == max_episode_steps:
                done = False
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            steps_this_ep += 1
            if steps_this_ep >= max_episode_steps:
                done = True

        update_policy = step % delay == 0
        for _ in range(gradient_updates_per_step):
            learn_fn(
                buffer=buffer,
                target_agent=target_agent,
                agent=agent,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                log_alpha=log_alpha,
                log_alpha_optimizer=log_alpha_optimizer,
                target_entropy=target_entropy,
                batch_size=batch_size,
                gamma=gamma,
                critic_clip=critic_clip,
                actor_clip=actor_clip,
                update_policy=update_policy,
            )

            # move target model towards training model
            if update_policy:
                utils.soft_update(target_agent.critic1, agent.critic1, tau)
                utils.soft_update(target_agent.critic2, agent.critic2, tau)

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
    target_agent,
    agent,
    actor_optimizer,
    critic_optimizer,
    log_alpha_optimizer,
    target_entropy,
    batch_size,
    log_alpha,
    gamma,
    critic_clip,
    actor_clip,
    update_policy=True,
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

    alpha = torch.exp(log_alpha)
    with torch.no_grad():
        action_dist_s1 = agent.actor(next_state_batch)
        action_s1 = action_dist_s1.rsample()
        logp_a1 = action_dist_s1.log_prob(action_s1).sum(-1, keepdim=True)
        target_action_value_s1 = torch.min(
            target_agent.critic1(next_state_batch, action_s1),
            target_agent.critic2(next_state_batch, action_s1),
        )
        td_target = reward_batch + gamma * (1.0 - done_batch) * (
            target_action_value_s1 - (alpha * logp_a1)
        )

    # update critics
    agent_critic1_pred = agent.critic1(state_batch, action_batch)
    td_error1 = td_target - agent_critic1_pred
    if per:
        critic1_loss = (imp_weights * 0.5 * (td_error1 ** 2)).mean()
    else:
        critic1_loss = 0.5 * (td_error1 ** 2).mean()
    agent_critic2_pred = agent.critic2(state_batch, action_batch)
    td_error2 = td_target - agent_critic2_pred
    if per:
        critic2_loss = (imp_weights * 0.5 * (td_error2 ** 2)).mean()
    else:
        critic2_loss = 0.5 * (td_error2 ** 2).mean()
    critic_loss = critic1_loss + critic2_loss
    critic_optimizer.zero_grad()
    critic_loss.backward()
    if critic_clip:
        torch.nn.utils.clip_grad_norm_(
            chain(agent.critic1.parameters(), agent.critic2.parameters()), critic_clip
        )
    critic_optimizer.step()

    if update_policy:
        # actor update
        dist = agent.actor(state_batch)
        agent_actions = dist.rsample()
        logp_a = dist.log_prob(agent_actions).sum(-1, keepdim=True)
        actor_loss = -(
            torch.min(
                agent.critic1(state_batch, agent_actions),
                agent.critic2(state_batch, agent_actions),
            )
            - (alpha.detach() * logp_a)
        ).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        if actor_clip:
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), actor_clip)
        actor_optimizer.step()

        # alpha update
        alpha_loss = (-alpha * (logp_a + target_entropy).detach()).mean()
        log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        log_alpha_optimizer.step()

    if per:
        new_priorities = (abs(td_error1) + 1e-5).cpu().detach().squeeze(1).numpy()
        buffer.update_priorities(priority_idxs, new_priorities)


def learn_discrete(
    buffer,
    target_agent,
    agent,
    actor_optimizer,
    critic1_optimizer,
    critic2_optimizer,
    log_alpha_optimizer,
    target_entropy,
    batch_size,
    log_alpha,
    gamma,
    critic_clip,
    actor_clip,
    update_policy=True,
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

    alpha = torch.exp(log_alpha)
    with torch.no_grad():
        # create critic targets (clipped double Q learning)
        action_s1, logp_a1 = agent.stochastic_forward(
            next_state_batch, track_gradients=False, process_states=False
        )
        target_value_s1 = (
            logp_a1.exp()
            * (
                torch.min(
                    target_agent.critic1(next_state_batch),
                    target_agent.critic2(next_state_batch),
                )
                - (alpha * logp_a1)
            )
        ).sum(1, keepdim=True)
        td_target = reward_batch + gamma * (1.0 - done_batch) * (target_value_s1)

    # update first critic
    agent_critic1_pred = agent.critic1(state_batch).gather(1, action_batch.long())
    td_error1 = td_target - agent_critic1_pred
    if per:
        critic1_loss = (imp_weights * 0.5 * (td_error1 ** 2)).mean()
    else:
        critic1_loss = 0.5 * (td_error1 ** 2).mean()
    critic1_optimizer.zero_grad()
    critic1_loss.backward()
    if critic_clip:
        torch.nn.utils.clip_grad_norm_(agent.critic1.parameters(), critic_clip)
    critic1_optimizer.step()

    # update second critic
    agent_critic2_pred = agent.critic2(state_batch).gather(1, action_batch.long())
    td_error2 = td_target - agent_critic2_pred
    if per:
        critic2_loss = (imp_weights * 0.5 * (td_error2 ** 2)).mean()
    else:
        critic2_loss = 0.5 * (td_error2 ** 2).mean()
    critic2_optimizer.zero_grad()
    critic2_loss.backward()
    if critic_clip:
        torch.nn.utils.clip_grad_norm_(agent.critic2.parameters(), critic_clip)
    critic2_optimizer.step()

    if update_policy:
        # actor update
        agent_actions, logp_a = agent.stochastic_forward(
            state_batch, track_gradients=True, process_states=False
        )
        actor_loss = (
            (
                logp_a.exp()
                * (
                    (alpha.detach() * logp_a)
                    - torch.min(agent.critic1(state_batch), agent.critic2(state_batch))
                )
            )
            .sum(1, keepdim=True)
            .mean()
        )
        actor_optimizer.zero_grad()
        actor_loss.backward()
        if actor_clip:
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), actor_clip)
        actor_optimizer.step()

        # alpha update
        alpha_loss = torch.sum(
            logp_a.exp().detach() * (-alpha * (logp_a.detach() + target_entropy)),
            dim=1,
            keepdim=True,
        ).mean()
        log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        log_alpha_optimizer.step()

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
        "--batch_size", type=int, default=256, help="training batch size"
    )
    parser.add_argument(
        "--tau", type=float, default=0.005, help="for model parameter % update"
    )
    parser.add_argument(
        "--actor_lr", type=float, default=1e-4, help="actor learning rate"
    )
    parser.add_argument(
        "--critic_lr", type=float, default=1e-3, help="critic learning rate"
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
        "--name", type=str, default="sac_run", help="dir name for saves"
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
        "--delay",
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
        help="how many gradient updates to make per env step",
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
        "--discrete_actions",
        action="store_true",
        help="enable SAC Discrete update function for discrete action spaces",
    )
    parser.add_argument(
        "--log_std_low",
        type=int,
        default=-10,
        help="Lower bound for log std of action distribution.",
    )
    parser.add_argument(
        "--log_std_high",
        type=int,
        default=2,
        help="Upper bound for log std of action distribution.",
    )
