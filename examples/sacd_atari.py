import argparse

import gym

import deep_control as dc


def train_atari_sacd(args):
    # create envs. Note use of the same seed
    train_env = dc.envs.load_atari(**vars(args))
    test_env = dc.envs.load_atari(**vars(args))

    # create agent
    obs_shape = train_env.observation_space.shape
    actions = train_env.action_space.n
    agent = dc.agents.PixelSACDAgent(obs_shape, actions)

    # create replay buffer
    if args.prioritized_replay:
        buffer_t = dc.replay.PrioritizedReplayBuffer
    else:
        buffer_t = dc.replay.ReplayBuffer
    buffer = buffer_t(
        args.buffer_size,
        state_shape=train_env.observation_space.shape,
        state_dtype=int,
        action_shape=(1,),
    )

    # run SAC
    dc.sac.sac(
        agent=agent, train_env=train_env, test_env=test_env, buffer=buffer, **vars(args)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add atari-related cl args
    dc.envs.add_atari_args(parser)
    # add sac-related cl args
    dc.sac.add_args(parser)
    args = parser.parse_args()
    args.discrete_actions = True

    train_atari_sacd(args)
