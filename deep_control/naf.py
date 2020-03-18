import argparse
import copy
import time

import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import tensorboardX

import utils
import run

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def naf(agent, env, args):
    agent.to(device)

    # initialize target networks
    target_agent = type(agent)()
    target_agent.to(device)
    utils.hard_update(target_agent.network, agent.network)

    random_process = utils.OrnsteinUhlenbeckProcess(size=env.action_space.shape, sigma=args.sigma, theta=args.theta)

    buffer = utils.ReplayBuffer(args.buffer_size)
    optimizer = torch.optim.Adam(agent.network.parameters(), lr=args.lr, weight_decay=args.l2)

    save_dir = utils.make_process_dirs(args.name)

    sample_state = env.reset()
    if isinstance(sample_state, dict):
        return _dict_based_naf(agent, target_agent, random_process, buffer, optimizer, save_dir, env, args)
    else:
        return _array_based_naf(agent, target_agent, random_process, buffer, optimizer, save_dir, env, args)


def _array_based_naf(agent, target_agent, random_process, buffer, optimizer, save_dir, env, args):
    eps = args.eps_start
    # use warmp up steps to add random transitions to the buffer
    state = env.reset()
    done = False
    for _ in range(args.warmup_steps):
        if done: state = env.reset(); done = False
        rand_action = env.action_space.sample()
        next_state, reward, done, info = env.step(rand_action)
        buffer.push(state, rand_action, reward, next_state, done)
        state = next_state

    for episode in range(args.num_episodes):
        rollout = utils.collect_rollout(agent, random_process, eps, env, args)
        eps = max(args.eps_final, eps - (args.eps_start - args.eps_final)/args.eps_anneal)

        for (state, action, rew, next_state, done, info) in rollout:
            buffer.push(state, action, rew, next_state, done)

            if args.her:
                raise ValueError(f"HER Not Supported on standard (non goal-based environments). For now, this means HER only works with the robotics sims.")

        for optimization_step in range(args.opt_steps):
            _naf_optimization_step(args, buffer, target_agent, agent, optimizer)
            # move target model towards training model
            utils.soft_update(target_agent.network, agent.network, args.tau)
        
        if episode % args.eval_interval == 0:
            mean_return = utils.evaluate_agent(agent, env, args)
            print(f"Episodes of training: {episode+1}, mean reward in test mode: {mean_return}")
   
    agent.save(save_dir)
    return agent


def _dict_based_naf(agent, target_agent, random_process, buffer, optimizer, save_dir, env, args):
    eps = args.eps_start
    # use warmp up steps to add random transitions to the buffer
    state = env.reset()
    done = False
    for _ in range(args.warmup_steps):
        if done: state = env.reset(); done = False
        rand_action = env.action_space.sample()
        next_state, reward, done, info = env.step(rand_action)
        state_goal = np.concatenate((state['observation'], state['desired_goal']))
        next_state_goal = np.concatenate((next_state['observation'], next_state['desired_goal']))
        buffer.push(state_goal, rand_action, reward, next_state_goal, done)
        state = next_state

    for episode in range(args.num_episodes):
        rollout = utils.collect_rollout(agent, random_process, eps, env, args)
        eps = max(args.eps_final, eps - (args.eps_start - args.eps_final)/args.eps_anneal)

        # add rollout to buffer
        substitute_goal = rollout[-1][3]['achieved_goal'] # last state reached on the rollout
        for (state, action, rew, next_state, done, info) in rollout:
            state_goal = np.concatenate((state['observation'], state['desired_goal']))
            next_state_goal = np.concatenate((next_state['observation'], next_state['desired_goal']))
            buffer.push(state_goal, action, rew, next_state_goal, done)

            if args.her:
                new_rew = env.compute_reward(state['achieved_goal'], substitute_goal, info)
                new_done = True if new_rew == 0. else False
                new_state_goal = np.concatenate((state['observation'], substitute_goal))
                new_next_state_goal = np.concatenate((next_state['observation'], substitute_goal))
                buffer.push(new_state_goal, action, new_rew, new_next_state_goal, new_done)

        for optimization_step in range(args.opt_steps):
            _naf_optimization_step(args, buffer, target_agent, agent, optimizer)
            # move target model towards training model
            utils.soft_update(target_agent.network, agent.network, args.tau)
        
        if episode % args.eval_interval == 0:
            mean_return = utils.evaluate_agent(agent, env, args)
            print(f"Episodes of training: {episode+1}, mean reward in test mode: {mean_return}")
   
    agent.save(save_dir)
    return agent

def _naf_optimization_step(args, buffer, target_agent, agent, optimizer):
    batch = buffer.sample(args.batch_size)
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

    _, _, next_state_values = target_agent.network(next_state_batch)
    td_targets = reward_batch + args.gamma*(1.-done_batch)*next_state_values

    mu, l, v = agent.network(state_batch)

    # calculate Q values from network outputs
    l *= agent.network.tril_mask.expand_as(l) + torch.exp(l) * agent.network.diag_mask.expand_as(l)
    p = torch.bmm(l, l.transpose(2, 1))
    u_mu = (action_batch - mu).unsqueeze(2)
    a = -.5 * torch.bmm(torch.bmm(u_mu.transpose(2, 1), p), u_mu)[:,:,0]
    agent_qval_preds = v + a

    optimizer.zero_grad()
    loss = F.mse_loss(td_targets, agent_qval_preds)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.network.parameters(), args.clip)
    optimizer.step()

def parse_args():

    parser = argparse.ArgumentParser(description='Train agent with NAF',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='naf_run', help='base name of save dir (a unique number will be added)')
    parser.add_argument('--env', type=str, default='brightpoint3x3', help='training environment')
    parser.add_argument('--actor_arch', default='standard', help='Architecture of Actor Network')
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='number of episodes for training')
    parser.add_argument('--max_episode_steps', type=int, default=150,
                        help='maximum steps per episode')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--tau', type=float, default=.001,
                        help='for model parameter perc update')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='network learning rate')
    parser.add_argument('--gamma', type=float, default=.99,
                        help='gamma, the discount factor')
    parser.add_argument('--eps_start', type=float, default=1.)
    parser.add_argument('--eps_final', type=float, default=.5)
    parser.add_argument('--eps_anneal', type=float, default=10000, help='How many **episodes** to anneal eps over.')
    parser.add_argument('--theta', type=float, default=.15,
        help='theta for Ornstein Uhlenbeck process computation')
    parser.add_argument('--sigma', type=float, default=.2,
        help='sigma for Ornstein Uhlenbeck process computation')
    parser.add_argument('--buffer_size', type=int, default=10**6,
        help='replay buffer size')
    parser.add_argument('--eval_interval', type=int, default=1000,
        help='how often to test the agent without exploration (in episodes)')
    parser.add_argument('--eval_episodes', type=int, default=1,
        help='how many episodes to run for when testing')
    parser.add_argument('--warmup_steps', type=int, default=1000,
        help='warmup length, in steps')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--log_interval', type=int, default=10,
        help='How often to log training metrics like actor loss and critic loss (in steps)')
    parser.add_argument('--l2', type=float, default=.0)
    parser.add_argument('--save_interval', type=int, default=1000,
        help='How often (in episodes) to save the actor and critic models to disk')
    parser.add_argument('--metric', type=str, default='l2')
    parser.add_argument('--target_class', type=int, default=None)
    parser.add_argument('--clip', type=float, default=100.)
    parser.add_argument('--her', action='store_true')
    parser.add_argument('--opt_steps', type=int, default=50)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    agent, env = run.load_env(args.env, 'naf')
    agent = naf(agent, env, args)
