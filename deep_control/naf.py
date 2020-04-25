import argparse
import copy
import time

import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import tensorboardX

from . import utils
from . import run

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def naf(agent, train_env, args):
    """
    Train `agent` on `env` with the Normalized Advantage Function algorithm.

    Reference: https://arxiv.org/abs/1603.00748
    """
    agent.to(device)

    # initialize target networks
    target_agent = copy.deepcopy(agent)
    target_agent.to(device)
    utils.hard_update(target_agent.network, agent.network)

    random_process = utils.OrnsteinUhlenbeckProcess(size=train_env.action_space.shape, sigma=args.sigma_start, sigma_min=args.sigma_final, n_steps_annealing=args.sigma_anneal, theta=args.theta)

    buffer = utils.ReplayBuffer(args.buffer_size)
    optimizer = torch.optim.Adam(agent.network.parameters(), lr=args.lr, weight_decay=args.l2)

    save_dir = utils.make_process_dirs(args.name)
    test_env = copy.deepcopy(train_env)

    # use warmp up steps to add random transitions to the buffer
    state = train_env.reset()
    done = False
    for _ in range(args.warmup_steps):
        if done: state = train_env.reset(); done = False
        rand_action = train_env.action_space.sample()
        next_state, reward, done, info = train_env.step(rand_action)
        buffer.push(state, rand_action, reward, next_state, done)
        state = next_state

    for episode in range(args.num_episodes):
        rollout = utils.collect_rollout(agent, random_process, train_env, args)

        for (state, action, rew, next_state, done, info) in rollout:
            buffer.push(state, action, rew, next_state, done)

        for optimization_step in range(args.opt_steps):
            _naf_learn(args, buffer, target_agent, agent, optimizer)
            # move target model towards training model
            utils.soft_update(target_agent.network, agent.network, args.tau)
        
        if episode % args.eval_interval == 0:
            mean_return = utils.evaluate_agent(agent, test_env, args)
            print(f"Episodes of training: {episode+1}, mean reward in test mode: {mean_return}")
   
    agent.save(save_dir)
    return agent


def _naf_learn(args, buffer, target_agent, agent, optimizer):
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
    if args.clip:
        torch.nn.utils.clip_grad_norm_(agent.network.parameters(), args.clip)
    optimizer.step()

def parse_args():

    parser = argparse.ArgumentParser(description='Train agent with NAF',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='naf_run', help='base name of save dir (a unique number will be added)')
    parser.add_argument('--env', type=str, default='brightpoint3x3', help='training environment')
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
    parser.add_argument('--sigma_final', type=float, default=.2)
    parser.add_argument('--sigma_anneal', type=float, default=10000, help='How many steps to anneal sigma over.')
    parser.add_argument('--theta', type=float, default=.15,
        help='theta for Ornstein Uhlenbeck process computation')
    parser.add_argument('--sigma_start', type=float, default=.2,
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
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--l2', type=float, default=.0)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--clip', type=float, default=None)
    parser.add_argument('--opt_steps', type=int, default=50)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    agent, env = run.load_env(args.env, 'naf')
    agent = naf(agent, env, args)
