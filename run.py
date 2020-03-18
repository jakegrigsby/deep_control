import argparse

import gym
import torch
import utils
import agents

def run(agent, env, episodes, max_steps, render=False, verbosity=1):
    episode_return_history = []
    for episode in range(episodes):
        episode_return = 0
        state = env.reset()
        done, info = False, {}
        for _ in range(max_steps):
            if done: break
            action = agent.forward(state)
            state, reward, done, info = env.step(action)
            if render: env.render()
            episode_return += reward
        if verbosity:
            print(f"Episode {episode}:: {episode_return}")
        episode_return_history.append(episode_return)
    return torch.tensor(episode_return_history)

def load_env(env_id, algo_type):
    if algo_type not in ['naf', 'ddpg']:
        raise ValueError(f"'{algo_type}' algorithm code not recognized.")
    if env_id == 'Pendulum-v0':
        if algo_type == 'ddpg':
            agent = agents.PendulumACAgent()
        else:
            agent = agents.PendulumNAFAgent()
    elif env_id == 'MountaincarContinuous-v0':
        if algo_type == 'ddpg':
            agent = agents.MountaincarACAgent()
        else:
            agent = agents.MountaincarNAFAgent()
    elif env_id == 'Ant-v2':
        if algo_type == 'ddpg':
            agent = agents.AntACAgent()
        else:
            agent = agents.AntNAFAgent()
    elif env_id == 'Walker2d-v2':
        if algo_type == 'ddpg':
            agent = agents.WalkerACAgent()
        else:
            agent = agents.WalkerNAFAgent()
    elif env_id == 'Swimmer-v2':
        if algo_type == 'ddpg':
            agent = agents.SwimmerACAgent()
        else:
            agent = agents.SwimmerNAFAgent()
    elif env_id == 'Reacher-v2':
        if algo_type == 'ddpg':
            agent = agents.ReacherACAgent()
        else:
            agent = agents.ReacherNAFAgent()
    elif env_id == 'Hopper-v2':
        if algo_type == 'ddpg':
            agent = agents.HopperACAgent()
        else:
            agent = agents.HopperNAFAgent()
    elif env_id == 'Humanoid-v2' or env_id == 'HumanoidStandup-v2':
        if algo_type == 'ddpg':
            agent = agents.HumanoidACAgent()
        else:
            agent = agents.HumanoidNAFAgent()
    elif env_id == 'HalfCheetah-v2':
        if algo_type == 'ddpg':
            agent = agents.CheetahACAgent()
        else:
            agent = agents.CheetahNAFAgent()
    else:
        raise ValueError(f"'{env_id}' environment code not recognized.")
    env = gym.make(env_id)
    return agent, env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', type=int, default=1)
    parser.add_argument('--env', type=str)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--save', type=str)
    parser.add_argument('--algo', type=str)
    parser.add_argument('--max_steps', type=int, default=300)
    args = parser.parse_args()

    agent, env = load_env(args.env, args.algo)
    agent.load(args.agent)
    run(agent, env, args.episodes, args.max_steps, args.render, verbosity=1)
