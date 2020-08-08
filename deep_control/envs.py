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


class DiscreteActionWrapper(gym.ActionWrapper):
    """
    This is intended to let the action be any scalar
    (float or int) or np array (float or int) of size 1.

    floats are cast to ints using python's standard rounding.
    """

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


def load_gym(env_id, seed=None, **_):
    """
    Load an environment from OpenAI gym (or pybullet_gym, if installed)
    """
    # optional pybullet import
    try:
        import pybullet
        import pybulletgym
    except ImportError:
        pass
    env = gym.make(env_id)
    if seed is None:
        seed = random.randint(1, 100)
    env.seed(seed)
    return env


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


def add_atari_args(parser):
    parser.add_argument("--game_id", type=str, default="Boxing-v0")
    parser.add_argument("--noop_max", type=int, default=30)
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--screen_size", type=int, default=84)
    parser.add_argument("--terminal_on_life_loss", action="store_true")
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--frame_stack", type=int, default=4)


def load_atari(
    game_id,
    seed=None,
    noop_max=30,
    frame_skip=1,
    screen_size=84,
    terminal_on_life_loss=False,
    rgb=False,
    normalize=False,
    frame_stack=4,
    clip_reward=True,
    **_,
):
    """
    Load a game from the Atari benchmark, with the usual settings

    Note that the simplest game ids (e.g. Boxing-v0) come with frame
    skipping by default, and you'll get an error if the frame_skp arg > 1.
    Use `BoxingNoFrameskip-v0` with frame_skip > 1.
    """
    env = gym.make(game_id)
    if seed is None:
        seed = random.randint(1, 100)
    env.seed(seed)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=noop_max,
        frame_skip=frame_skip,
        screen_size=screen_size,
        terminal_on_life_loss=terminal_on_life_loss,
        grayscale_obs=not rgb,
        scale_obs=normalize,
    )
    if frame_stack > 1:
        env = gym.wrappers.FrameStack(env, num_stack=frame_stack)
    if clip_reward:
        env = ClipReward(env)
    env = DiscreteActionWrapper(env)
    return env


class ClipReward(gym.RewardWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self._clip_low = low
        self._clip_high = high

    def reward(self, rew):
        return max(min(rew, self._clip_high), self._clip_low)


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
    **_,
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
