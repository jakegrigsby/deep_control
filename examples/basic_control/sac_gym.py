import argparse

import deep_control as dc


def train_gym_sac(args):
    # same training and testing seed
    train_env = dc.envs.load_gym(args.env, seed=231)
    test_env = dc.envs.load_gym(args.env, seed=231)

    state_space = train_env.observation_space
    action_space = train_env.action_space

    # create agent
    agent = dc.agents.SACAgent(
        state_space.shape[0], action_space.shape[0], max_action=action_space.high[0]
    )

    # create replay buffer
    if args.prioritized_replay:
        buffer_type = dc.replay.PrioritizedReplayBuffer
    else:
        buffer_type = dc.replay.ReplayBuffer
    buffer = buffer_type(
        args.buffer_size,
        state_shape=state_space.shape,
        state_dtype=float,
        action_shape=action_space.shape,
    )

    # run sac
    dc.sac.sac(
        agent=agent, train_env=train_env, test_env=test_env, buffer=buffer, **vars(args)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dc.envs.add_gym_args(parser)
    dc.sac.add_args(parser)
    args = parser.parse_args()

    train_gym_sac(args)
