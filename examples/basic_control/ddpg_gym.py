import argparse

import deep_control as dc


def train_gym_ddpg(args):
    # same training and testing seed
    train_env = dc.envs.load_gym(args.env_id, args.seed)
    test_env = dc.envs.load_gym(args.env_id, args.seed)

    state_space = train_env.observation_space
    action_space = train_env.action_space

    # create agent
    agent = dc.ddpg.DDPGAgent(state_space.shape[0], action_space.shape[0])

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

    # run ddpg
    dc.ddpg.ddpg(agent=agent, train_env=train_env, test_env=test_env, buffer=buffer, **vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dc.envs.add_gym_args(parser)
    dc.ddpg.add_args(parser)
    args = parser.parse_args()
    train_gym_ddpg(args)
