import argparse
import copy
import os
from itertools import chain

import numpy as np
import tensorboardX
import torch
import torch.nn.functional as F
import tqdm

from . import envs, nets, replay, run, utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3Agent:
    def __init__(
        self,
        obs_space_size,
        act_space_size,
        actor_net_cls=nets.BaselineActor,
        critic_net_cls=nets.BigCritic,
        hidden_size=256,
    ):
        self.actor = actor_net_cls(
            obs_space_size, act_space_size, hidden_size=hidden_size
        )
        self.critic1 = critic_net_cls(
            obs_space_size, act_space_size, hidden_size=hidden_size
        )
        self.critic2 = critic_net_cls(
            obs_space_size, act_space_size, hidden_size=hidden_size
        )

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

    def forward(self, state, from_cpu=True):
        if from_cpu:
            state = self.process_state(state)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)
        self.actor.train()
        if from_cpu:
            action = self.process_act(action)
        return action

    def process_state(self, state):
        return torch.from_numpy(np.expand_dims(state, 0).astype(np.float32)).to(
            utils.device
        )

    def process_act(self, act):
        return np.squeeze(act.cpu().numpy(), 0)


def td3(
    agent,
    train_env,
    test_env,
    buffer,
    num_steps=1_000_000,
    transitions_per_step=1,
    max_episode_steps=100_000,
    batch_size=256,
    tau=0.005,
    actor_lr=1e-4,
    critic_lr=1e-3,
    gamma=0.99,
    sigma_start=0.2,
    sigma_final=0.1,
    sigma_anneal=100_000,
    theta=0.15,
    eval_interval=5000,
    eval_episodes=10,
    warmup_steps=1000,
    actor_clip=None,
    critic_clip=None,
    actor_l2=0.0,
    critic_l2=0.0,
    delay=2,
    target_noise_scale=0.2,
    save_interval=100_000,
    c=0.5,
    name="td3_run",
    render=False,
    save_to_disk=True,
    log_to_disk=True,
    verbosity=0,
    gradient_updates_per_step=1,
    td_reg_coeff=0.0,
    td_reg_coeff_decay=0.9999,
    infinite_bootstrap=True,
    **_,
):
    """
    Train `agent` on `train_env` with Twin Delayed Deep Deterministic Policy
    Gradient algorithm, and evaluate on `test_env`.

    Reference: https://arxiv.org/abs/1802.09477
    """
    if save_to_disk or log_to_disk:
        save_dir = utils.make_process_dirs(name)
    if log_to_disk:
        # create tb writer, save hparams
        writer = tensorboardX.SummaryWriter(save_dir)
        writer.add_hparams(locals(), {})

    agent.to(device)

    # initialize target networks
    target_agent = copy.deepcopy(agent)
    target_agent.to(device)
    utils.hard_update(target_agent.actor, agent.actor)
    utils.hard_update(target_agent.critic1, agent.critic1)
    utils.hard_update(target_agent.critic2, agent.critic2)

    random_process = utils.OrnsteinUhlenbeckProcess(
        size=train_env.action_space.shape,
        sigma=sigma_start,
        sigma_min=sigma_final,
        n_steps_annealing=sigma_anneal,
        theta=theta,
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
        agent.actor.parameters(), lr=actor_lr, weight_decay=actor_l2
    )

    run.warmup_buffer(buffer, train_env, warmup_steps, max_episode_steps)

    done = True

    steps_iter = range(num_steps)
    if verbosity:
        steps_iter = tqdm.tqdm(steps_iter)

    for step in steps_iter:
        for _ in range(transitions_per_step):
            if done:
                state = train_env.reset()
                random_process.reset_states()
                steps_this_ep = 0
                done = False
            action = agent.forward(state)
            noisy_action = run.exploration_noise(action, random_process)
            next_state, reward, done, info = train_env.step(noisy_action)
            if infinite_bootstrap:
                # allow infinite bootstrapping
                if steps_this_ep + 1 == max_episode_steps:
                    done = False
            buffer.push(state, noisy_action, reward, next_state, done)
            state = next_state
            steps_this_ep += 1
            if steps_this_ep >= max_episode_steps:
                done = True

        update_policy = step % delay == 0
        for _ in range(gradient_updates_per_step):
            learn(
                buffer=buffer,
                target_agent=target_agent,
                agent=agent,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                batch_size=batch_size,
                target_noise_scale=target_noise_scale,
                c=c,
                gamma=gamma,
                critic_clip=critic_clip,
                actor_clip=actor_clip,
                td_reg_coeff=td_reg_coeff,
                update_policy=update_policy,
            )

            # move target model towards training model
            if update_policy:
                utils.soft_update(target_agent.actor, agent.actor, tau)
                # original td3 impl only updates critic targets with the actor...
                utils.soft_update(target_agent.critic1, agent.critic1, tau)
                utils.soft_update(target_agent.critic2, agent.critic2, tau)

        # decay td regularization
        td_reg_coeff *= td_reg_coeff_decay

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
    batch_size,
    target_noise_scale,
    c,
    gamma,
    critic_clip,
    actor_clip,
    td_reg_coeff,
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

    with torch.no_grad():
        # create critic targets (clipped double Q learning)
        target_action_s1 = target_agent.actor(next_state_batch)
        target_noise = torch.clamp(
            target_noise_scale * torch.randn(*target_action_s1.shape).to(device), -c, c
        )
        # target smoothing
        target_action_s1 = torch.clamp(
            target_action_s1 + target_noise,
            -1.0,
            1.0,
        )
        target_action_value_s1 = torch.min(
            target_agent.critic1(next_state_batch, target_action_s1),
            target_agent.critic2(next_state_batch, target_action_s1),
        )
        td_target = reward_batch + gamma * (1.0 - done_batch) * target_action_value_s1

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
        agent_actions = agent.actor(state_batch)
        actor_loss = -(
            agent.critic1(state_batch, agent_actions).mean()
            - td_reg_coeff * critic_loss.detach()
        )
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
        "--num_steps",
        type=int,
        default=10 ** 6,
        help="number of training steps",
    )
    parser.add_argument(
        "--transitions_per_step",
        type=int,
        default=1,
        help="env transitions collected per training step. Defaults to 1, in which case we're training for num_steps total env steps. But when looking for replay ratios < 1, this value will need to be set higher.",
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
    parser.add_argument("--sigma_final", type=float, default=0.1)
    parser.add_argument(
        "--sigma_anneal",
        type=float,
        default=100_000,
        help="How many steps to anneal sigma over.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.15,
        help="theta for Ornstein Uhlenbeck process computation",
    )
    parser.add_argument(
        "--sigma_start",
        type=float,
        default=0.2,
        help="sigma for Ornstein Uhlenbeck process computation",
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
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--actor_clip", type=float, default=None)
    parser.add_argument("--critic_clip", type=float, default=None)
    parser.add_argument("--name", type=str, default="td3_run")
    parser.add_argument("--actor_l2", type=float, default=0.0)
    parser.add_argument("--critic_l2", type=float, default=0.0)
    parser.add_argument("--delay", type=int, default=2)
    parser.add_argument("--target_noise_scale", type=float, default=0.2)
    parser.add_argument("--save_interval", type=int, default=100_000)
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--verbosity", type=int, default=1)
    parser.add_argument("--gradient_updates_per_step", type=int, default=1)
    parser.add_argument("--prioritized_replay", action="store_true")
    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument("--skip_save_to_disk", action="store_true")
    parser.add_argument("--skip_log_to_disk", action="store_true")
    parser.add_argument("--td_reg_coeff", type=float, default=0.0)
    parser.add_argument("--td_reg_coeff_decay", type=float, default=0.9999)
