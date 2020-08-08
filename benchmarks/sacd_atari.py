import argparse

import gym

import deep_control as dc


def main():
    parser = argparse.ArgumentParser()
    """
    # add dmc-related cl args
    dc.envs.add_dmc_args(parser)
    """
    # add sac-related cl args
    dc.sac.add_args(parser)
    args = parser.parse_args()
    args.discrete_actions = True

    env = dc.envs.DiscreteWrapper(gym.make("MountainCar-v0"))

    obs_shape = env.observation_space.shape
    actions = env.action_space.n

    """
    # select an agent architecture
    if args.from_pixels:
        agent = dc.agents.PixelSACDAgent(obs_shape, action_shape[0], max_action)
    else:
    """
    agent = dc.agents.SACDAgent(obs_shape[0], actions)

    # select a replay buffer
    if args.prioritized_replay:
        buffer_t = dc.replay.PrioritizedReplayBuffer
    else:
        buffer_t = dc.replay.ReplayBuffer
    buffer = buffer_t(
        args.buffer_size,
        state_shape=env.observation_space.shape,
        state_dtype=float,
        action_shape=(1,),
    )

    print(f"Using device: {dc.device}")

    # run SAC
    dc.sac.sac(agent=agent, env=env, buffer=buffer, **vars(args))


if __name__ == "__main__":
    main()
