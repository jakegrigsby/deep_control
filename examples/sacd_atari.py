import argparse

import gym

import deep_control as dc


def main():
    parser = argparse.ArgumentParser()

    # add atari-related cl args
    dc.envs.add_atari_args(parser)

    # add sac-related cl args
    dc.sac.add_args(parser)
    args = parser.parse_args()
    args.discrete_actions = True

    env = dc.envs.load_atari(**vars(args))

    obs_shape = env.observation_space.shape
    actions = env.action_space.n

    agent = dc.agents.PixelSACDAgent(obs_shape, actions)

    # select a replay buffer
    if args.prioritized_replay:
        buffer_t = dc.replay.PrioritizedReplayBuffer
    else:
        buffer_t = dc.replay.ReplayBuffer
    buffer = buffer_t(
        args.buffer_size,
        state_shape=env.observation_space.shape,
        state_dtype=int,
        action_shape=(1,),
    )

    print(f"Using device: {dc.device}")

    # run SAC
    dc.sac.sac(agent=agent, env=env, buffer=buffer, **vars(args))


if __name__ == "__main__":
    main()
