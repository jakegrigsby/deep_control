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


def load_env(env_id, algo_type):
    env = gym.make(env_id)

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
