import argparse
import copy
import time

import gym
import numpy as np
import tensorboardX
import torch
import torch.nn.functional as F
import tqdm

from . import run, utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ddpg(
    agent,
    env,
    num_steps=1_000_000,
    max_episode_steps=100_000,
    batch_size=64,
    tau=0.005,
    actor_lr=1e-4,
    critic_lr=1e-3,
    gamma=0.99,
    sigma_start=0.2,
    sigma_final=0.1,
    sigma_anneal=10_000,
    theta=0.15,
    buffer_size=1_000_000,
    eval_interval=5000,
    eval_episodes=10,
    warmup_steps=1000,
    render=False,
    actor_clip=None,
    critic_clip=None,
    name="ddpg_run",
    actor_l2=0.0,
    critic_l2=0.0,
    save_interval=10_000,
    log_to_disk=True,
    save_to_disk=True,
    verbosity=0,
):
    """
    Train `agent` on `env` with the Deep Deterministic Policy Gradient algorithm.

    Reference: https://arxiv.org/abs/1509.02971
    """
    agent.to(device)
    max_act = env.action_space.high[0]

    # initialize target networks
    target_agent = copy.deepcopy(agent)
    target_agent.to(device)
    utils.hard_update(target_agent.actor, agent.actor)
    utils.hard_update(target_agent.critic, agent.critic)

    random_process = utils.GaussianExplorationNoise(
        size=env.action_space.shape,
        start_scale=sigma_start,
        final_scale=sigma_final,
        steps_annealed=sigma_anneal,
    )

    buffer = utils.ReplayBuffer(buffer_size)
    critic_optimizer = torch.optim.Adam(
        agent.critic.parameters(), lr=critic_lr, weight_decay=critic_l2
    )
    actor_optimizer = torch.optim.Adam(
        agent.actor.parameters(), lr=actor_lr, weight_decay=actor_l2
    )

    if save_to_disk:
        # create save directory for this run
        save_dir = utils.make_process_dirs(name)
    if log_to_disk:
        # create tb writer, save hparams
        writer = tensorboardX.SummaryWriter(save_dir)

    utils.warmup_buffer(buffer, env, warmup_steps, max_episode_steps)

    done = True
    learning_curve = []

    steps_iter = range(num_steps)
    if verbosity:
        steps_iter = tqdm.tqdm(steps_iter)

    for step in steps_iter:
        if done:
            state = env.reset()
            random_process.reset_states()
            steps_this_ep = 0
            done = False
        action = agent.forward(state)
        noisy_action = utils.exploration_noise(action, random_process, max_act)
        next_state, reward, done, info = env.step(noisy_action)
        buffer.push(state, noisy_action, reward, next_state, done)
        state = next_state
        steps_this_ep += 1
        if steps_this_ep >= max_episode_steps:
            done = True

        _ddpg_learn(
            buffer,
            target_agent,
            agent,
            actor_optimizer,
            critic_optimizer,
            batch_size,
            gamma,
            critic_clip,
            actor_clip,
        )
        # move target model towards training model
        utils.soft_update(target_agent.actor, agent.actor, tau)
        utils.soft_update(target_agent.critic, agent.critic, tau)

        if step % eval_interval == 0:
            mean_return = utils.evaluate_agent(
                agent, env, eval_episodes, max_episode_steps, render
            )
            if log_to_disk:
                writer.add_scalar("return", mean_return, step)
            learning_curve.append((step, mean_return))

        if step % save_interval == 0 and save_to_disk:
            agent.save(save_dir)

    if save_to_disk:
        agent.save(save_dir)
    return agent, learning_curve


def _ddpg_learn(
    buffer,
    target_agent,
    agent,
    actor_optimizer,
    critic_optimizer,
    batch_size,
    gamma,
    critic_clip,
    actor_clip,
):
    """
    DDPG inner optimization loop
    """
    batch = buffer.sample(args.batch_size)
    # batch will be None if not enough experience has been collected yet
    if not batch:
        return

    # prepare transitions for models
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

    cat_tuple = lambda t: torch.cat(t).to(device)
    list_to_tensor = lambda t: torch.tensor(t).unsqueeze(0).to(device)
    state_batch = cat_tuple(state_batch)
    next_state_batch = cat_tuple(next_state_batch)
    action_batch = cat_tuple(action_batch)
    reward_batch = list_to_tensor(reward_batch).T
    done_batch = list_to_tensor(done_batch).T

    agent.train()
    target_agent.train()

    # critic update
    with torch.no_grad():
        target_action_s2 = target_agent.actor(next_state_batch)
        target_action_value_s2 = target_agent.critic(next_state_batch, target_action_s2)
        td_target = (
            reward_batch + args.gamma * (1.0 - done_batch) * target_action_value_s2
        )
    agent_critic_pred = agent.critic(state_batch, action_batch)
    critic_loss = F.mse_loss(td_target, agent_critic_pred)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    if args.critic_clip:
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), args.critic_clip)
    critic_optimizer.step()

    # actor update
    agent_actions = agent.actor(state_batch)
    actor_loss = -agent.critic(state_batch, agent_actions).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    if args.actor_clip:
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), args.actor_clip)
    actor_optimizer.step()


def parse_args():
    parser = argparse.ArgumentParser(description="Train agent with DDPG")
    parser.add_argument(
        "--env", type=str, default="Pendulum-v0", help="training environment"
    )
    parser.add_argument(
        "--num_steps", type=int, default=1000000, help="number of episodes for training"
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=100000,
        help="maximum steps per episode",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="training batch size"
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
        default=100000,
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
        "--buffer_size", type=int, default=1000000, help="replay buffer size"
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=5000,
        help="how often to test the agent without exploration (in steps)",
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
    parser.add_argument("--name", type=str, default="ddpg_run")
    parser.add_argument("--actor_l2", type=float, default=0.0)
    parser.add_argument("--critic_l2", type=float, default=0.0)
    parser.add_argument("--save_interval", type=int, default=10000)
    parser.add_argument("--verbosity", type=int, default=1)
    parser.add_argument("--skip_save_to_disk", action="store_true")
    parser.add_argument("--skip_log_to_disk", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    agent, env = run.load_env(args.env, "ddpg")
    print(f"Using Device: {device}")
    agent = ddpg(
        agent,
        env,
        num_steps=args.num_steps,
        max_episode_steps=args.max_episode_steps,
        batch_size=args.batch_size,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        sigma_start=args.sigma_start,
        sigma_final=args.sigma_final,
        sigma_anneal=args.sigma_anneal,
        theta=args.theta,
        buffer_size=args.buffer_size,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        warmup_steps=args.warmup_steps,
        actor_clip=args.actor_clip,
        critic_clip=args.critic_clip,
        actor_l2=args.actor_l2,
        critic_l2=args.critic_l2,
        save_interval=args.save_interval,
        render=args.render,
        name=args.name,
        save_to_disk=not args.skip_save_to_disk,
        log_to_disk=not args.skip_log_to_disk,
        verbosity=args.verbosity,
    )
