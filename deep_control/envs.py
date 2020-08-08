import argparse
import random

import gym
import numpy as np

from . import agents


class ChannelsFirstWrapper(gym.ObservationWrapper):
    """
    Some pixel-based gym environments use a (Height, Width, Channel) image format.
    This wrapper rolls those axes to (Channel, Height, Width) to work with pytorch
    Conv2D layers.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space.shape = (
            env.observation_space.shape[-1],
        ) + env.observation_space.shape[:-1]

    def observation(self, frame):
        frame = np.transpose(frame, (2, 0, 1))
        # this is a trick to make numpy put this array in contiguous memory
        return frame - np.zeros_like(frame)


class DiscreteWrapper(gym.ActionWrapper):
    def action(self, action):
        if isinstance(action, np.ndarray):
            if len(action.shape) > 0:
                action = action[0]
        return int(action)


class GoalBasedWrapper(gym.ObservationWrapper):
    """
    Some goal-based envs (like the Gym Robotics suite) use dictionary observations
    with one entry for the current state and another to describe the goal. This
    wrapper concatenates those into a single vector so it can be used just like
    any other env.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space.shape = (
            env.observation_space["observation"].shape[0]
            + env.observation_space["desired_goal"].shape[0],
        )

    def observation(self, obs_dict):
        return self._flatten_obs(obs_dict)

    def _flatten_obs(self, obs_dict):
        return np.concatenate((obs_dict["observation"], obs_dict["desired_goal"]))


def add_gym_args(parser):
    """
    Add a --env cl flag to an argparser
    """
    parser.add_argument("--env", type=str, default="Pendulum-v0")


def load_gym(env_id):
    """
    Load an environment from OpenAI gym (or pybullet_gym, if installed)
    """
    # optional pybullet import
    try:
        import pybullet
        import pybulletgym
    except ImportError:
        pass
    return gym.make(env_id)


def add_dmc_args(parser):
    """
    Add cl flags associated with the deepmind control suite to a parser
    """
    parser.add_argument("--domain_name", type=str, default="fish")
    parser.add_argument("--task_name", type=str, default="swim")
    parser.add_argument("--from_pixels", action="store_true")
    parser.add_argument("--height", type=int, default=84)
    parser.add_argument("--width", type=int, default=84)
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--channels_last", action="store_true")


def load_dmc(
    domain_name,
    task_name,
    seed=None,
    from_pixels=False,
    height=84,
    width=84,
    camera_id=0,
    frame_skip=1,
    channels_last=False,
    **kwargs
):
    """
    Load a task from the deepmind control suite. 

    Uses dmc2gym (https://github.com/denisyarats/dmc2gym)

    Note that setting seed=None (the default) picks a random seed
    """
    import dmc2gym

    if seed is None:
        seed = random.randint(1, 100)
    return dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        from_pixels=from_pixels,
        height=height,
        width=width,
        camera_id=camera_id,
        visualize_reward=False,
        frame_skip=frame_skip,
        channels_first=not channels_last,
    )


def load_env(env_id, algo_type):
    env = load_gym(env_id)

    # decide if env is a goal based (dict) env
    if isinstance(env.observation_space, gym.spaces.dict.Dict):
        env = GoalBasedWrapper(env)

    # decide if we are learning from state or pixels
    if len(env.observation_space.shape) > 1:
        from_state = False
        if env.observation_space.shape[0] > env.observation_space.shape[-1]:
            # assume channels-last env and wrap to channels-first
            env = ChannelsFirstWrapper(env)
        obs_shape = env.observation_space.shape
    else:
        from_state = True
        obs_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    if from_state:
        if algo_type == "ddpg":
            agent = agents.DDPGAgent(obs_shape, action_shape, max_action)
        elif algo_type == "sac":
            agent = agents.SACAgent(obs_shape, action_shape, max_action)
        elif algo_type == "td3":
            agent = agents.TD3Agent(obs_shape, action_shape, max_action)
    else:
        if algo_type == "ddpg":
            agent = agents.PixelDDPGAgent(obs_shape, action_shape, max_action)
        elif algo_type == "sac":
            agent = agents.PixelSACAgent(obs_shape, action_shape, max_action)
        elif algo_type == "td3":
            agent = agents.PixelTD3Agent(obs_shape, action_shape, max_action)
    return agent, env
