import argparse

import gym
import pybullet
import pybullet_envs
import torch

from . import agents, utils


def run(agent, env, episodes, max_steps, render=False, verbosity=1):
    episode_return_history = []
    if render:
        env.render()
    for episode in range(episodes):
        episode_return = 0
        state = env.reset()
        done, info = False, {}
        for _ in range(max_steps):
            if done:
                break
            action = agent.forward(state)
            state, reward, done, info = env.step(action)
            if render:
                env.render()
            episode_return += reward
        if verbosity:
            print(f"Episode {episode}:: {episode_return}")
        episode_return_history.append(episode_return)
    return torch.tensor(episode_return_history)


def load_env(env_id, algo_type):
    env = gym.make(env_id)
    shape = (env.observation_space.shape[0], env.action_space.shape[0])
    max_action = env.action_space.high[0]
    if algo_type == "ddpg":
        agent = agents.DDPGAgent(*shape, max_action)
    elif algo_type == "sac":
        agent = agents.SACAgent(*shape, max_action)
    elif algo_type == "td3":
        agent = agents.TD3Agent(*shape, max_action)
    return agent, env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--env", type=str)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--save", type=str)
    parser.add_argument("--algo", type=str)
    parser.add_argument("--max_steps", type=int, default=300)
    args = parser.parse_args()

    agent, env = load_env(args.env, args.algo)
    agent.load(args.agent)
    run(agent, env, args.episodes, args.max_steps, args.render, verbosity=1)
