import argparse

import gym
import d4rl
import numpy as np

import deep_control as dc


def train_d4rl_sbc(args):
    test_env = gym.make(args.env_id)
    test_env.seed(args.seed)
    state_space = test_env.observation_space
    action_space = test_env.action_space

    # create agent
    agent = dc.sbc.SBCAgent(
        state_space.shape[0],
        action_space.shape[0],
        args.log_std_low,
        args.log_std_high,
    )

    # get offline datset
    dset = d4rl.qlearning_dataset(test_env)
    dset_size = dset["observations"].shape[0]
    # create replay buffer
    buffer = dc.replay.ReplayBuffer(
        size=dset_size,
        state_shape=state_space.shape,
        state_dtype=float,
        action_shape=action_space.shape,
    )
    buffer.load_experience(
        dset["observations"],
        dset["actions"],
        dset["rewards"],
        dset["next_observations"],
        dset["terminals"],
    )

    # run sbc
    dc.sbc.sbc(agent=agent, test_env=test_env, buffer=buffer, **vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dc.envs.add_gym_args(parser)
    dc.sbc.add_args(parser)
    args = parser.parse_args()
    train_d4rl_sbc(args)
