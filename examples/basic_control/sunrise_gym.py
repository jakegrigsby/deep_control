import argparse

import deep_control as dc


def train_gym_sunrise(args):
    train_env = dc.envs.load_gym(**vars(args))
    test_env = dc.envs.load_gym(**vars(args))

    obs_shape = train_env.observation_space.shape
    action_shape = train_env.action_space.shape
    max_action = train_env.action_space.high[0]

    agent = dc.sunrise.SunriseAgent(
        obs_shape[0],
        action_shape[0],
        args.log_std_low,
        args.log_std_high,
        args.ensemble_size,
        args.ucb_bonus,
    )

    # select a replay buffer
    if args.prioritized_replay:
        buffer_t = dc.replay.PrioritizedReplayBuffer
    else:
        buffer_t = dc.replay.ReplayBuffer
    buffer = buffer_t(
        args.buffer_size,
        state_dtype=float,
        state_shape=train_env.observation_space.shape,
        action_shape=train_env.action_space.shape,
    )

    agent = dc.sunrise.sunrise(
        agent=agent, train_env=train_env, test_env=test_env, buffer=buffer, **vars(args)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dc.envs.add_gym_args(parser)
    # add sunrise-related cl args
    dc.sunrise.add_args(parser)
    args = parser.parse_args()
    train_gym_sunrise(args)
