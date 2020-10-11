import argparse
import random
from collections import deque

import gym
import numpy as np


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
        return np.ascontiguousarray(frame)


class DiscreteActionWrapper(gym.ActionWrapper):
    """
    This is intended to let the action be any scalar
    (float or int) or np array (float or int) of size 1.

    floats are cast to ints using python's standard rounding.
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space.shape = (env.action_space.n,)

    def action(self, action):
        if isinstance(action, np.ndarray):
            if len(action.shape) > 0:
                action = action[0]
        return int(action)


class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack):
        gym.Wrapper.__init__(self, env)
        self._k = num_stack
        self._frames = deque([], maxlen=num_stack)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * num_stack,) + shp[1:]),
            dtype=env.observation_space.dtype,
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


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
    Add a --env_id cl flag to an argparser
    """
    parser.add_argument("--env_id", type=str, default="Pendulum-v0")
    parser.add_argument("--seed", type=int, default=231)


def load_gym(env_id="CartPole-v1", seed=None, **_):
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
        seed = random.randint(1, 100000)
    env.seed(seed)
    return env


def add_dmc_args(parser):
    """
    Add cl flags associated with the deepmind control suite to a parser
    """
    parser.add_argument("--domain_name", type=str, default="fish")
    parser.add_argument("--task_name", type=str, default="swim")
    parser.add_argument(
        "--from_pixels", action="store_true", help="Use image observations"
    )
    parser.add_argument("--height", type=int, default=84)
    parser.add_argument("--width", type=int, default=84)
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--frame_stack", type=int, default=3)
    parser.add_argument("--channels_last", action="store_true")
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--seed", type=int, default=231)


def add_atari_args(parser):
    parser.add_argument("--game_id", type=str, default="Boxing-v0")
    parser.add_argument("--noop_max", type=int, default=30)
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--screen_size", type=int, default=84)
    parser.add_argument("--terminal_on_life_loss", action="store_true")
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--seed", type=int, default=231)


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
        seed = random.randint(1, 100000)
    env.seed(seed)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=noop_max,
        frame_skip=frame_skip,
        screen_size=screen_size,
        terminal_on_life_loss=terminal_on_life_loss,
        grayscale_obs=False,  # use GrayScale wrapper instead...
        scale_obs=normalize,
    )
    if not rgb:
        env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
    if clip_reward:
        env = ClipReward(env)
    env = ChannelsFirstWrapper(env)
    env = FrameStack(env, num_stack=frame_stack)
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
    frame_stack=1,
    height=84,
    width=84,
    camera_id=0,
    frame_skip=1,
    channels_last=False,
    rgb=False,
    **_,
):
    """
    Load a task from the deepmind control suite. 

    Uses dmc2gym (https://github.com/denisyarats/dmc2gym)

    Note that setting seed=None (the default) picks a random seed
    """
    import dmc2gym

    if seed is None:
        seed = random.randint(1, 100000)
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        from_pixels=from_pixels,
        height=height,
        width=width,
        camera_id=camera_id,
        visualize_reward=False,
        frame_skip=frame_skip,
        channels_first=not channels_last
        if rgb
        else False,  # if we're using RGB, set the channel order here
    )
    if not rgb and from_pixels:
        env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
        env = ChannelsFirstWrapper(env)
    if from_pixels:
        env = FrameStack(env, num_stack=frame_stack)
    return env
