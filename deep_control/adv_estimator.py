import torch
from torch import nn


class AdvantageEstimator(nn.Module):
    def __init__(
        self, actor, critics, popart=False, method="mean", ensembling="mean", n=4
    ):
        super().__init__()
        assert method in ["mean", "max"]
        assert ensembling in ["min", "mean"]
        self.actor = actor
        self.critics = critics
        self.method = method
        self.ensembling = ensembling
        self.val_s = None
        self.popart = popart
        self._n = n

    def pop(self, q, s, a):
        if self.popart:
            return self.popart(q(s, a))
        else:
            return q(s, a)

    def get_hparams(self):
        return {"adv_method": self.method, "adv_ensembling_method": self.method}

    def estimate_value(self, state):
        # get an action distribution from the policy
        act_dist = self.actor(state)
        actions = [act_dist.sample() for _ in range(self._n)]

        # get the q value for each of the n actions
        qs = []
        for act in actions:
            q_preds = torch.stack(
                [self.pop(critic, state, act) for critic in self.critics], dim=0
            )
            if self.ensembling == "min":
                q_preds = q_preds.min(0).values
            elif self.ensembling == "mean":
                q_preds = q_preds.mean(0)
            qs.append(q_preds)

        if self.method == "mean":
            # V(s) = E_{a ~ \pi(s)} [Q(s, a)]
            value = torch.stack(qs, dim=0).mean(0)
        elif self.method == "max":
            # Optimisitc value estimate: V(s) = max_{a1, a2, a3, ..., aN}(Q(s, a))
            value = torch.stack(qs, dim=0).max(0).values
        self.val_s = value
        return value

    def forward(self, state, action, use_computed_val=False):
        with torch.no_grad():
            q_preds = torch.stack(
                [self.pop(critic, state, action) for critic in self.critics], dim=0
            )
            if self.ensembling == "min":
                q_preds = q_preds.min(0).values
            elif self.ensembling == "mean":
                q_preds = q_preds.mean(0)
            # reuse the expensive value computation if it has already been done
            if use_computed_val:
                assert self.val_s is not None
            else:
                # do the value computation
                self.estimate_value(state)
            # A(s, a) = Q(s, a) - V(s)
            adv = q_preds - self.val_s
        return adv


class AdvEstimatorFilter(nn.Module):
    def __init__(self, adv_estimator, filter_type="binary", beta=1.0):
        super().__init__()
        self.adv_estimator = adv_estimator
        self.filter_type = filter_type
        self.beta = beta
        self._norm_a2 = 0.5

    def get_hparams(self):
        return {"filter_type": self.filter_type, "filter_beta": self.beta}

    def forward(self, s, a, step_num=None):
        adv = self.adv_estimator(s, a)
        if self.filter_type == "exp":
            filter_val = (self.beta * adv.clamp(-5.0, 5.0)).exp()
        elif self.filter_type == "binary":
            filter_val = (adv >= 0.0).float()
        elif self.filter_type == "exp_norm":
            self._norm_a2 += 1e-5 * (adv.mean() ** 2 - self._norm_a2)
            norm_a = a / ((self._norm_a2).sqrt() + 1e-5)
            filter_val = (self.beta * norm_a).exp()
        elif self.filter_type == "softmax":
            batch_size = s.shape[0]
            filter_val = batch_size * F.softmax(self.beta * adv, dim=0)
        elif self.filter_type == "identity":
            filter_val = torch.ones_like(adv)
        else:
            raise ValueError(f"Unrecognized filter type '{self.filter_type}'")
        # final clip for numerical stability (only applies to exp filters)
        return filter_val.clamp(-100.0, 100.0)
