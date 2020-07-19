import argparse
import time
import copy

import torch
import torch.nn.functional as F
import numpy as np
import gym
import tensorboardX

from . import utils
from . import run


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def td3(agent,
        env,
        num_steps=1_000_000,
        max_episode_steps=100_000,
        batch_size=64,
        tau=.005,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=.99,
        sigma_start=.2,
        sigma_final=.1,
        sigma_anneal=10_000,
        theta=.15,
        buffer_size=1_000_000,
        eval_interval=5000,
        eval_episodes=10,
        warmup_steps=1000,
        actor_clip=None,
        critic_clip=None,
        actor_l2=0.,
        critic_l2=0.,
        delay=2,
        target_noise_scale=.2,
        save_interval=10_000,
        c=.5,
        name='td3_run',
        render=False,
        ):

    agent.to(device)
    max_act = env.action_space.high[0]

    # initialize target networks
    target_agent = copy.deepcopy(agent)
    target_agent.to(device)
    utils.hard_update(target_agent.actor, agent.actor)
    utils.hard_update(target_agent.critic1, agent.critic1)
    utils.hard_update(target_agent.critic2, agent.critic2)

    random_process = utils.OrnsteinUhlenbeckProcess(size=env.action_space.shape, sigma=sigma_start, sigma_min=sigma_final, n_steps_annealing=sigma_anneal, theta=theta)

    buffer = utils.ReplayBuffer(buffer_size)
    critic1_optimizer = torch.optim.Adam(agent.critic1.parameters(), lr=critic_lr, weight_decay=critic_l2)
    critic2_optimizer = torch.optim.Adam(agent.critic2.parameters(), lr=critic_lr, weight_decay=critic_l2)
    actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=actor_lr, weight_decay=actor_l2)

    save_dir = utils.make_process_dirs(name)
    # create tb writer, save hparams
    writer = tensorboardX.SummaryWriter(save_dir)

    utils.warmup_buffer(buffer, env, warmup_steps, max_episode_steps)

    done = True
    learning_curve = []
    for step in range(num_steps):
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
        if steps_this_ep >= max_episode_steps: done = True

        update_policy = (step  % delay == 0)
        _td3_learn(buffer, target_agent, agent, actor_optimizer, critic1_optimizer, critic2_optimizer, env.action_space.high[0], batch_size, target_noise_scale, c, gamma, critic_clip, actor_clip, update_policy)

        # move target model towards training model
        if update_policy:
            utils.soft_update(target_agent.actor, agent.actor, tau)
        utils.soft_update(target_agent.critic1, agent.critic1, tau)
        utils.soft_update(target_agent.critic2, agent.critic2, tau)
        
        if step % eval_interval == 0:
            mean_return = utils.evaluate_agent(agent, env, eval_episodes, max_episode_steps, render)
            writer.add_scalar('return', mean_return, step)
            learning_curve.append((step, mean_return))

        if step % save_interval == 0:
            agent.save(save_dir)
   
    agent.save(save_dir)
    return agent

def _td3_learn(buffer, 
                target_agent, 
                agent, 
                actor_optimizer, 
                critic1_optimizer, 
                critic2_optimizer, 
                max_act, 
                batch_size, 
                target_noise_scale,
                c,
                gamma,
                critic_clip,
                actor_clip,
                update_policy=True,
                ):
    batch = buffer.sample(args.batch_size)
    # batch will be None if not enough experience has been collected yet
    if not batch:
        return
    
    # prepare transitions for models
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
    
    cat_tuple = lambda t : torch.cat(t).to(device)
    list_to_tensor = lambda t : torch.tensor(t).unsqueeze(0).to(device)
    state_batch = cat_tuple(state_batch)
    next_state_batch = cat_tuple(next_state_batch)
    action_batch = cat_tuple(action_batch)
    reward_batch = list_to_tensor(reward_batch).T
    done_batch = list_to_tensor(done_batch).T

    agent.train()

    with torch.no_grad():
        # create critic targets (clipped double Q learning)
        target_action_s2 = target_agent.actor(next_state_batch)
        target_noise = torch.clamp(target_noise_scale*torch.randn(*target_action_s2.shape).to(device), -c, c)
        # target smoothing
        target_action_s2 = torch.clamp(target_action_s2 + target_noise, -max_act, max_act)
        target_action_value_s2 = torch.min(target_agent.critic1(next_state_batch, target_action_s2), target_agent.critic2(next_state_batch, target_action_s2))
        td_target = reward_batch + gamma*(1.-done_batch)*target_action_value_s2

    # update first critic
    agent_critic1_pred = agent.critic1(state_batch, action_batch)
    critic1_loss = F.mse_loss(td_target, agent_critic1_pred)
    critic1_optimizer.zero_grad()
    critic1_loss.backward()
    if critic_clip:
        torch.nn.utils.clip_grad_norm_(agent.critic1.parameters(), critic_clip)
    critic1_optimizer.step()

    # update second critic
    agent_critic2_pred = agent.critic2(state_batch, action_batch)
    critic2_loss = F.mse_loss(td_target, agent_critic2_pred)
    critic2_optimizer.zero_grad()
    critic2_loss.backward()
    if critic_clip:
        torch.nn.utils.clip_grad_norm_(agent.critic2.parameters(), critic_clip)
    critic2_optimizer.step()

    if update_policy:
        # actor update
        agent_actions = agent.actor(state_batch)
        actor_loss = -agent.critic1(state_batch, agent_actions).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        if actor_clip:
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), actor_clip)
        actor_optimizer.step()

def parse_args():
    parser = argparse.ArgumentParser(description='Train agent with DDPG')
    parser.add_argument('--env', type=str, default='Pendulum-v0', help='training environment')
    parser.add_argument('--num_steps', type=int, default=10**6,
                        help='number of episodes for training')
    parser.add_argument('--max_episode_steps', type=int, default=100000,
                        help='maximum steps per episode')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='training batch size')
    parser.add_argument('--tau', type=float, default=.005,
                        help='for model parameter % update')
    parser.add_argument('--actor_lr', type=float, default=1e-4,
                        help='actor learning rate')
    parser.add_argument('--critic_lr', type=float, default=1e-3,
                        help='critic learning rate')
    parser.add_argument('--gamma', type=float, default=.99,
                        help='gamma, the discount factor')
    parser.add_argument('--sigma_final', type=float, default=.1)
    parser.add_argument('--sigma_anneal', type=float, default=10000, help='How many steps to anneal sigma over.')
    parser.add_argument('--theta', type=float, default=.15,
        help='theta for Ornstein Uhlenbeck process computation')
    parser.add_argument('--sigma_start', type=float, default=.2,
        help='sigma for Ornstein Uhlenbeck process computation')
    parser.add_argument('--buffer_size', type=int, default=1000000,
        help='replay buffer size')
    parser.add_argument('--eval_interval', type=int, default=5000,
        help='how often to test the agent without exploration (in episodes)')
    parser.add_argument('--eval_episodes', type=int, default=10,
        help='how many episodes to run for when testing')
    parser.add_argument('--warmup_steps', type=int, default=1000,
        help='warmup length, in steps')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--actor_clip', type=float, default=None)
    parser.add_argument('--critic_clip', type=float, default=None)
    parser.add_argument('--name', type=str, default='ddpg_run')
    parser.add_argument('--actor_l2', type=float, default=0.)
    parser.add_argument('--critic_l2', type=float, default=0.)
    parser.add_argument('--delay', type=int, default=2)
    parser.add_argument('--target_noise_scale', type=float, default=.2)
    parser.add_argument('--save_interval', type=int, default=10000)
    parser.add_argument('--c', type=float, default=.5)
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    agent, env = run.load_env(args.env, 'td3')
    print(f"Using Device: {device}")
    agent = td3(agent, 
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
            delay=args.delay,
            target_noise_scale=args.target_noise_scale,
            save_interval=args.save_interval,
            c=args.c,
            name=args.name,
            render=args.render,
        )


