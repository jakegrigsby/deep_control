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

def td3(agent, train_env, args):
    agent.to(device)

    # initialize target networks
    target_agent = copy.deepcopy(agent)
    target_agent.to(device)
    utils.hard_update(target_agent.actor, agent.actor)
    utils.hard_update(target_agent.critic1, agent.critic1)
    utils.hard_update(target_agent.critic2, agent.critic2)
    test_env = copy.deepcopy(train_env)

    random_process = utils.OrnsteinUhlenbeckProcess(size=train_env.action_space.shape, sigma=args.sigma_start, sigma_min=args.sigma_final, n_steps_annealing=args.sigma_anneal, theta=args.theta)

    buffer = utils.ReplayBuffer(args.buffer_size)
    critic1_optimizer = torch.optim.Adam(agent.critic1.parameters(), lr=args.critic_lr, weight_decay=args.critic_l2)
    critic2_optimizer = torch.optim.Adam(agent.critic2.parameters(), lr=args.critic_lr, weight_decay=args.critic_l2)
    actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=args.actor_lr, weight_decay=args.actor_l2)

    save_dir = utils.make_process_dirs(args.name)
    # create tb writer, save hparams
    writer = tensorboardX.SummaryWriter(save_dir)
    hparams_dict = utils.clean_hparams_dict(vars(args))
    writer.add_hparams(hparams_dict, {})

    utils.warmup_buffer(buffer, train_env, args.warmup_steps, args.max_episode_steps)

    done = True
    learning_curve = []
    for step in range(args.num_steps):
        if done: 
            state = train_env.reset()
            random_process.reset_states()
            steps_this_ep = 0
            done = False
        action = agent.forward(state)
        noisy_action = utils.exploration_noise(action, random_process)
        next_state, reward, done, info = train_env.step(noisy_action)
        buffer.push(state, noisy_action, reward, next_state, done)
        next_state = state
        steps_this_ep += 1
        if steps_this_ep >= args.max_episode_steps: done = True

        update_policy = (step  % args.delay == 0)
        _td3_learn(args, buffer, target_agent, agent, actor_optimizer, critic1_optimizer, critic2_optimizer, update_policy)

        # move target model towards training model
        if update_policy:
            utils.soft_update(target_agent.actor, agent.actor, args.tau)
        utils.soft_update(target_agent.critic1, agent.critic1, args.tau)
        utils.soft_update(target_agent.critic2, agent.critic2, args.tau)
        
        if step % args.eval_interval == 0:
            mean_return = utils.evaluate_agent(agent, test_env, args)
            writer.add_scalar('return', mean_return, step)
            learning_curve.append((step, mean_return))

        if step % args.save_interval == 0:
            agent.save(save_dir)
   
    agent.save(save_dir)
    return agent

def _td3_learn(args, buffer, target_agent, agent, actor_optimizer, critic1_optimizer, critic2_optimizer, update_policy=True):
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
        # target smoothing
        target_action_s2 += torch.clamp(args.target_noise_scale*torch.randn(*target_action_s2.shape), -args.c, args.c)
        target_action_value_s2 = torch.min(target_agent.critic1(next_state_batch, target_action_s2), target_agent.critic2(next_state_batch, target_action_s2))
        td_target = reward_batch + args.gamma*(1.-done_batch)*target_action_value_s2

    # update first critic
    agent_critic1_pred = agent.critic1(state_batch, action_batch)
    critic1_loss = F.mse_loss(td_target, agent_critic1_pred)
    critic1_optimizer.zero_grad()
    critic1_loss.backward()
    if args.critic_clip:
        torch.nn.utils.clip_grad_norm_(agent.critic1.parameters(), args.critic_clip)
    critic1_optimizer.step()

    # update second critic
    agent_critic2_pred = agent.critic2(state_batch, action_batch)
    critic2_loss = F.mse_loss(td_target, agent_critic2_pred)
    critic2_optimizer.zero_grad()
    critic2_loss.backward()
    if args.critic_clip:
        torch.nn.utils.clip_grad_norm_(agent.critic2.parameters(), args.critic_clip)
    critic2_optimizer.step()

    if update_policy:
        # actor update
        agent_actions = agent.actor(state_batch)
        actor_loss = -agent.critic1(state_batch, agent_actions).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        if args.actor_clip:
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), args.actor_clip)
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
    parser.add_argument('--sigma_final', type=float, default=.2)
    parser.add_argument('--sigma_anneal', type=float, default=10000, help='How many steps to anneal sigma over.')
    parser.add_argument('--theta', type=float, default=.15,
        help='theta for Ornstein Uhlenbeck process computation')
    parser.add_argument('--sigma_start', type=float, default=.2,
        help='sigma for Ornstein Uhlenbeck process computation')
    parser.add_argument('--buffer_size', type=int, default=100000,
        help='replay buffer size')
    parser.add_argument('--eval_interval', type=int, default=5000,
        help='how often to test the agent without exploration (in episodes)')
    parser.add_argument('--eval_episodes', type=int, default=10,
        help='how many episodes to run for when testing')
    parser.add_argument('--warmup_steps', type=int, default=25000,
        help='warmup length, in steps')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--actor_clip', type=float, default=None)
    parser.add_argument('--critic_clip', type=float, default=None)
    parser.add_argument('--name', type=str, default='ddpg_run')
    parser.add_argument('--actor_l2', type=float, default=0.)
    parser.add_argument('--critic_l2', type=float, default=1e-4)
    parser.add_argument('--delay', type=int, default=2)
    parser.add_argument('--target_noise_scale', type=float, default=.2)
    parser.add_argument('--save_interval', type=int, default=10000)
    parser.add_argument('--c', type=float, default=.5)
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    agent, env = run.load_env(args.env, 'td3')
    agent = td3(agent, env, args)


