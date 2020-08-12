import argparse

import deep_control as dc


def main():
    parser = argparse.ArgumentParser()
    # add dmc-related cl args
    dc.envs.add_dmc_args(parser)
    # add sac-related cl args
    dc.sac.add_args(parser)
    args = parser.parse_args()

    env = dc.envs.load_dmc(**vars(args))

    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    max_action = env.action_space.high[0]

    # select an agent architecture
    if args.from_pixels:
        agent = dc.agents.PixelSACAgent(obs_shape, action_shape[0], max_action)
    else:
        agent = dc.agents.SACAgent(obs_shape[0], action_shape[0], max_action)

    # select a replay buffer
    if args.prioritized_replay:
        buffer_t = dc.replay.PrioritizedReplayBuffer
    else:
        buffer_t = dc.replay.ReplayBuffer
    buffer = buffer_t(
        args.buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
    )

    print(f"Using device: {dc.device}")

    # run SAC
    dc.sac.sac(agent=agent, env=env, buffer=buffer, **vars(args))


if __name__ == "__main__":
    main()
