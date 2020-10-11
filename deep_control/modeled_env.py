import random

import gym
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModeledEnv(gym.Env):
    """
    A ModeledEnv looks and acts like a regular Gym environemnt, but instead of
    determining the game logic with a physics engine, it uses the predictions of
    a (an ensemble of) dynamics model(s) (see models.py).

    Every time this env is reset, it picks a random state from `starting_states` to restart
    from. If starting_states is set to every state from a replay buffer, this creates
    the 'branching rollout' behavior used in MBPO, where we are doing (short) rollouts
    with the model after pretending to follow the real env for some amount of time. If
    starting_states is set to only be starting states from the real env, then we are also
    approximating the env's initial state distribution. This might be better for more
    rollout-based algorithms like PPO?
    """

    def __init__(self, buffer, model, action_space=None, observation_space=None):
        assert (
            action_space
        ), "Must provide ModeledEnv with action_sapce object from real Gym Env being modeled"
        assert (
            observation_space
        ), "Must provide ModeledEnv with observation_sapce object from real Gym Env being modeled"
        self.buffer = buffer
        model.to(device)
        self.model = model
        self._current_state = None
        self.action_space = action_space
        self.observation_space = observation_space

    def reset(self):
        s, *_ = self.buffer.sample(1)
        self._current_state = s.to(device)
        return self._current_state.cpu().numpy()

    def step(self, action):
        state = self._current_state.unsqueeze(0)
        action = torch.Tensor(action).to(device).unsqueeze(0)
        with torch.no_grad():
            pred_next_state, pred_rew, pred_done = self.model(state, action)
        self._current_state = pred_next_state
        pred_next_state_np = pred_next_state.cpu().numpy()
        pred_rew_float = float(pred_rew.cpu())
        pred_done_bool = bool(pred_done.cpu().round())
        return pred_next_state_np, pred_rew_float, pred_done_bool, {}  # empty info dict


class ParallelModeledEnv(ModeledEnv):
    """
    This runs the modeled env in parallel by stacking (s, a, r, s_1, d) tensors
    and doing batch prediction on the GPU

    Uses a different API than a standard gym env. You should call current_state()
    to get the batch of states before each action prediction. step(action_batch) returns
    next_state_batch, reward_batch, done_batch, empty_info_dict.

    Call reset_if_done() after each prediction to reset the environments that happen to be done.
    reset_on_step(max_step), resets environments that have been running for max_step steps. Both
    return an integer representing how many environments they reset.
    """

    def __init__(
        self, buffer, model, parallel_envs=1, action_space=None, observation_space=None
    ):
        super().__init__(buffer, model, action_space, observation_space)
        self.parallel_envs = parallel_envs
        self.dones = torch.zeros((parallel_envs,)).bool().to(device)
        self.steps = torch.zeros((parallel_envs,)).to(device)
        self._current_state = torch.zeros((parallel_envs, *observation_space.shape)).to(
            device
        )
        self.reset_all()

    def _copy_state(self):
        return self._current_state.clone()

    def current_state(self):
        return self._copy_state()

    def _reset_idxs(self, idxs):
        s, *_ = self.buffer.sample(len(idxs))
        self._current_state[idxs] = s.to(device)
        self.steps[idxs] = 0.0
        self.dones[idxs] = False

    def reset_all(self):
        self._reset_idxs(torch.arange(self.parallel_envs))
        return self.parallel_envs

    def reset_if_done(self):
        idxs_to_reset = torch.where(self.dones == 1)[0]
        if idxs_to_reset.shape[0]:
            self._reset_idxs(idxs_to_reset)
        return len(idxs_to_reset)

    def reset_on_step(self, max_step):
        idxs_to_reset = torch.where(self.steps >= max_step)[0]
        if idxs_to_reset.shape[0]:
            self._reset_idxs(idxs_to_reset)
        return len(idxs_to_reset)

    def step(self, action):
        """
        Assumes action is a torch Tensor of shape (parallel_envs, *action_space.shape),
        and on the correct device

        returns torch tensors, and keeps them on the current device!
        """
        state = self._current_state
        with torch.no_grad():
            pred_next_state, pred_rew, pred_done = self.model(state, action)
        self._current_state = pred_next_state
        pred_done_bool = pred_done.round().bool()
        self.steps += 1
        self.dones = pred_done_bool
        return pred_next_state, pred_rew, pred_done_bool, {}  # empty info dict
