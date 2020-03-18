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
    if env_id == 'Pendulum-v0':
        agent = algo_switch('Pendulum', algo_type)
        env = gym.make(env_id)
    elif env_id == 'MountaincarContinuous-v0':
        agent = algo_switch('Mountaincar', algo_type)
        env = gym.make(env_id)
    elif env_id == 'Ant-v2':
        agent = algo_switch('Ant', algo_type)
        env = gym.make(env_id)
    elif env_id == 'Walker2d-v2':
        agent = algo_switch('Walker', algo_type)
        env = gym.make(env_id)
    elif env_id == 'Swimmer-v2':
        agent = algo_switch('Swimmer', algo_type)
        env = gym.make(env_id)
    elif env_id == 'Reacher-v2':
        agent = algo_switch('Reacher', algo_type)
        env = gym.make(env_id)
    elif env_id == 'Hopper-v2':
        agent = algo_switch('Hopper', algo_type)
        env = gym.make(env_id)
    elif env_id == 'Humanoid-v2':
        agent = algo_switch('Humanoid', algo_type)
        env = gym.make(env_id)
    elif env_id == 'HumanoidStandup-v2':
        agent = algo_switch('HumanoidStandup', algo_type)
        env = gym.make(env_id)
    elif env_id == 'HalfCheetah-v2':
        agent = algo_switch('Cheetah', algo_type)
        env = gym.make(env_id)
    elif env_id == 'FetchPush-v1':
        agent = algo_switch('FetchPush', algo_type)
        env = utils.FlattenRoboticsDictWrapper(gym.make(env_id))
    elif env_id == 'FetchReach-v1':
        agent = algo_switch('FetchReach', algo_type)
        env = utils.FlattenRoboticsDictWrapper(gym.make(env_id))
    elif env_id == 'FetchSlide-v1':
        agent = algo_switch('FetchSlide', algo_type)
        env = utils.FlattenRoboticsDictWrapper(gym.make(env_id))
    elif env_id == 'FetchPickAndPlace-v1':
        agent = algo_switch('FetchPickAndPlace', algo_type)
        env = utils.FlattenRoboticsDictWrapper(gym.make(env_id))
    elif env_id == 'HandReach-v0':
        agent = algo_switch('HandReach', algo_type)
        env = utils.FlattenRoboticsDictWrapper(gym.make(env_id))
    elif env_id == 'HandManipulateBlockRotate-v0':
        agent = algo_switch('HandManipulateBlockRotate', algo_type)
        env = utils.FlattenRoboticsDictWrapper(gym.make(env_id))
    elif env_id == 'HandManipulateBlockRotateParallel-v0':
        agent = algo_switch('HandManipulateBlockRotateParallel', algo_type)
        env = utils.FlattenRoboticsDictWrapper(gym.make(env_id))
    elif env_id == 'HandManipulateBlockRotateXYZ-v0':
        agent = algo_switch('HandManipulateBlockRotateXYZ', algo_type)
        env = utils.FlattenRoboticsDictWrapper(gym.make(env_id))
    elif env_id == 'HandManipulateBlockFull-v0':
        agent = algo_switch('HandManipulateBlockFull', algo_type)
        env = utils.FlattenRoboticsDictWrapper(gym.make(env_id))
    elif env_id == 'HandManipulateEgg-v0':
        agent = algo_switch('HandManipulateEgg', algo_type)
        env = utils.FlattenRoboticsDictWrapper(gym.make(env_id))
    elif env_id == 'HandManipulateEggRotate-v0':
        agent = algo_switch('HandManipulateEggRotate', algo_type)
        env = gym.make(env_id)
    elif env_id == 'HandManipulateEggFull-v0':
        agent = algo_switch('HandManipulateEggFull', algo_type)
        env = utils.FlattenRoboticsDictWrapper(gym.make(env_id))
    elif env_id == 'HandManipulatePen-v0':
        agent = algo_switch('HandManipulatePen', algo_type)
        env = utils.FlattenRoboticsDictWrapper(gym.make(env_id))
    elif env_id == 'HandManipulatePenRotate-v0':
        agent = algo_switch('HandManipulatePenRotate', algo_type)
        env = utils.FlattenRoboticsDictWrapper(gym.make(env_id))
    elif env_id == 'HandManipulatePenFull-v0':
        agent = algo_switch('HandManipulatePenFull', algo_type)
        env = utils.FlattenRoboticsDictWrapper(gym.make(env_id))
    else:
        raise ValueError(f"'{env_id}' environment code not recognized.")
    print(type(agent))
    return agent, env


def algo_switch(prefix, algo_type):
    if algo_type == 'ddpg':
        return eval(f"agents.{prefix}DDPGAgent")()
    elif algo_type == 'naf':
        return eval(f"agents.{prefix}NAFAgent")()
    else:
        raise ValueError(f"Unrecognized algorithm id: {algo_type}. 'ddpg' and 'naf' are currently supported.")

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
