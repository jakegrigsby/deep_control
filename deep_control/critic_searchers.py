import deep_control as dc

import torch
import numpy as np

"""
This is code is from https://github.com/stanford-iprl-lab/GRAC/blob/master/ES.py
"""


class _CEM:
    def __init__(
        self,
        num_params,
        mu_init=None,
        batch_size=256,
        sigma_init=1e-3,
        clip=0.5,
        pop_size=256,
        damp=1e-3,
        damp_limit=1e-5,
        parents=None,
        elitism=True,
        device=dc.device,
    ):

        # misc
        self.num_params = num_params
        self.batch_size = batch_size
        self.device = device
        # distribution parameters
        if mu_init is None:
            self.mu = torch.zeros([self.batch_size, self.num_params], device=device)
        else:
            self.mu = mu_init.clone()
        self.sigma = sigma_init
        self.damp = damp
        self.damp_limit = damp_limit
        self.tau = 0.95
        self.cov = self.sigma * torch.ones(
            [self.batch_size, self.num_params], device=device
        )
        self.clip = clip

        # elite stuff
        self.elitism = elitism
        self.elite = torch.sqrt(torch.tensor(self.sigma, device=device)) * torch.rand(
            self.batch_size, self.num_params, device=device
        )
        self.elite_score = None

        # sampling stuff
        self.pop_size = pop_size
        if parents is None or parents <= 0:
            self.parents = pop_size // 2
        else:
            self.parents = parents
        self.weights = torch.FloatTensor(
            [np.log((self.parents + 1) / i) for i in range(1, self.parents + 1)]
        ).to(device)
        self.weights /= self.weights.sum()

    def ask(self, pop_size):
        """
        Returns a list of candidates parameters
        """
        epsilon = torch.randn(
            self.batch_size, pop_size, self.num_params, device=self.device
        )
        inds = self.mu.unsqueeze(1) + (
            epsilon * torch.sqrt(self.cov).unsqueeze(1)
        ).clamp(-self.clip, self.clip)
        if self.elitism:
            inds[:, -1] = self.elite
        return inds

    def tell(self, solutions, scores):
        """
        Updates the distribution
        returns the best solution
        """
        scores = scores.clone().squeeze()
        scores *= -1
        if len(scores.shape) == 1:
            scores = scores[None, :]
        _, idx_sorted = torch.sort(scores, dim=1)

        old_mu = self.mu.clone()
        self.damp = self.damp * self.tau + (1 - self.tau) * self.damp_limit
        idx_sorted = idx_sorted[:, : self.parents]
        top_solutions = torch.gather(
            solutions,
            1,
            idx_sorted.unsqueeze(2).expand(*idx_sorted.shape, solutions.shape[-1]),
        )
        self.mu = self.weights @ top_solutions
        z = top_solutions - old_mu.unsqueeze(1)
        self.cov = 1 / self.parents * self.weights @ (z * z) + self.damp * torch.ones(
            [self.batch_size, self.num_params], device=self.device
        )

        self.elite = top_solutions[:, 0, :]

        return top_solutions[:, 0, :]

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return self.mu.clone(), self.cov.clone()


class CEM:
    def __init__(
        self,
        action_dim,
        max_action,
        batch_size=256,
        sigma_init=1e-3,
        clip=0.5,
        pop_size=25,
        damp=0.1,
        damp_limit=0.05,
        parents=5,
        device=dc.device,
    ):
        self.sigma_init = sigma_init
        self.clip = clip
        self.pop_size = pop_size
        self.damp = damp
        self.damp_limit = damp_limit
        self.parents = parents
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.max_action = max_action
        self.device = device

    def search(
        self, state, action_init, critic, batch_size=None, n_iter=2, action_bound=True
    ):
        if batch_size is None:
            batch_size = self.batch_size
        cem = _CEM(
            self.action_dim,
            action_init,
            batch_size,
            self.sigma_init,
            self.clip,
            self.pop_size,
            self.damp,
            self.damp_limit,
            self.parents,
            device=self.device,
        )
        with torch.no_grad():
            for iter in range(n_iter):
                actions = cem.ask(self.pop_size)
                if action_bound:
                    actions = actions.clamp(-self.max_action, self.max_action)
                actions_temp = actions.clone().view(self.pop_size * batch_size, -1)
                Qs = critic(
                    state.unsqueeze(1)
                    .repeat(1, self.pop_size, 1)
                    .view(self.pop_size * batch_size, -1),
                    actions_temp,
                ).view(batch_size, self.pop_size)
                best_action = cem.tell(actions, Qs)
                if iter == n_iter - 1:
                    best_Q = critic(state, best_action)
                    ori_Q = critic(state, action_init)

                    action_index = (best_Q < ori_Q).squeeze()
                    best_action[action_index] = action_init[action_index]

                    return best_action
