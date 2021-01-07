import random
import os

import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torch import distributions as pyd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DynamicsModel(LightningModule):
    """
    A DynamicsModel is a network that learns the transition, reward
    and terminal functions of an RL environment.
    """

    def __init__(self):
        super().__init__()
        self._val_dset = None
        self._train_dset = None
        self._lr = None
        self._l2 = None
        self._batch_size = None

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        s, a, r, s1, d = batch
        pred_s1_del_mean, pred_s1_del_log_var, pred_r, pred_d = self(s, a)
        d_loss = F.binary_cross_entropy(pred_d, d)
        r_loss = F.mse_loss(pred_r, r)

        s1_targets = s1 - s
        s_loss = self._compute_state_loss(
            pred_s1_del_mean, pred_s1_del_log_var, s1_targets
        )

        loss = d_loss + s_loss + r_loss
        return {"loss": loss, "log": {"train_loss": loss}}

    def _compute_state_loss(self, pred_s1_del_mean, pred_s1_del_log_var, s1_targets):
        inv_var = (-pred_s1_del_log_var).exp()
        deltas = pred_s1_del_mean - s1_targets
        s_loss = (deltas ** 2 * inv_var).sum(-1) + pred_s1_del_log_var.sum(-1)
        s_loss = s_loss.mean()
        s_loss += 0.01 * self.max_logvar.sum() - 0.01 * self.min_logvar.sum()
        return s_loss

    def validation_step(self, batch, batch_idx):
        s, a, r, s1, d = batch
        pred_s1_del_mean, pred_s1_del_log_var, pred_r, pred_d = self(s, a)
        d_loss = F.binary_cross_entropy(pred_d, d)
        r_loss = F.mse_loss(pred_r, r)

        s1_targets = s1 - s
        s_loss = self._compute_state_loss(
            pred_s1_del_mean, pred_s1_del_log_var, s1_targets
        )

        loss = d_loss + s_loss + r_loss

        return {
            "val_loss": loss,
            "done_loss": d_loss,
            "state_loss": s_loss,
            "rew_loss": r_loss,
        }

    def configure_optimizers(self):
        assert self._lr is not None
        assert self._l2 is not None
        return torch.optim.Adam(self.parameters(), lr=self._lr, weight_decay=self._l2)

    def train_dataloader(self):
        assert self._batch_size is not None
        assert self._train_dset is not None
        return torch.utils.data.DataLoader(
            self._train_dset, batch_size=self._batch_size, num_workers=4, shuffle=True
        )

    def val_dataloader(self):
        assert self._batch_size is not None
        assert self._val_dset is not None
        return torch.utils.data.DataLoader(
            self._val_dset, batch_size=self._batch_size, num_workers=4, shuffle=False
        )

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_d_loss = torch.stack([x["done_loss"] for x in outputs]).mean()
        avg_r_loss = torch.stack([x["rew_loss"] for x in outputs]).mean()
        avg_s_loss = torch.stack([x["state_loss"] for x in outputs]).mean()
        return {
            "val_loss": avg_loss,
            "log": {
                "val_loss": avg_loss,
                "done_loss": avg_d_loss,
                "state_loss": avg_s_loss,
                "rew_loss": avg_r_loss,
            },
        }

    def fit(
        self,
        transitions,
        val_split=0.2,
        lr=0.001,
        epochs=100,
        early_stopping_patience=3,
        batch_size=1000,
        l2=0.0,
        gpus=1,
        log_dir=None,
    ):
        """
        Train the dynamics model on transitions from the real RL env.

        transitions is a tuple of (states, actions, rewards, next_states, dones).
        (just use buffer.get_all_transitions())

        pytorch_lightning handles the training loop.
        """
        s, a, r, s1, d = transitions
        d = d.float()
        # update normalization stats with new batch
        self._state_mean = torch.mean(s, axis=0).to(device)
        self._state_std = torch.std(s, axis=0).to(device)
        self._action_mean = torch.mean(a, axis=0).to(device)
        self._action_std = torch.std(a, axis=0).to(device)
        num_samples = len(s)
        val_set_size = int(num_samples * val_split)
        train_set_size = num_samples - val_set_size
        transition_dset = torch.utils.data.TensorDataset(s, a, r, s1, d)
        train_set, val_set = torch.utils.data.random_split(
            transition_dset, [train_set_size, val_set_size]
        )
        self._train_dset = train_set
        self._val_dset = val_set
        self._lr = lr
        self._l2 = l2
        self._batch_size = batch_size
        early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=early_stopping_patience
        )
        if log_dir is not None:
            logger = pl_loggers.TensorBoardLogger("model_training_logs", name=log_dir)
        else:
            logger = False
        trainer = Trainer(
            gpus=gpus,
            num_nodes=1,
            early_stop_callback=early_stop_callback,
            max_epochs=epochs,
            logger=logger,
        )
        trainer.fit(self)


class SimpleFeedForwardModel(DynamicsModel):
    """
    This is a baseline dynamics model. It's a simple
    feedforward network that takes in the current state
    and action and outputs a prediction for the *change in the state*,
    the reward and the done bool of that transition.
    """

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._state_mean = torch.zeros((1, obs_dim)).to(device)
        self._action_mean = torch.zeros((1, act_dim)).to(device)
        self._state_std = torch.ones((1, obs_dim)).to(device)
        self._action_std = torch.ones((1, act_dim)).to(device)
        self.max_logvar = torch.ones((1, obs_dim), requires_grad=True).to(device)
        self.min_logvar = -torch.ones((1, obs_dim), requires_grad=True).to(device)

        self.fc1 = nn.Linear(obs_dim + act_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.next_state_delta_fc = nn.Linear(1024, 2 * obs_dim)
        self.reward_fc = nn.Linear(1024, 1)
        self.done_fc = nn.Linear(1024, 1)

    def forward(self, state, action):
        state = (state - self._state_mean) / (self._state_std + 1e-5)
        action = (action - self._action_mean) / (self._action_std + 1e-5)
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        next_state_del_mean, next_state_del_log_var = self.next_state_delta_fc(x).chunk(
            2, dim=1
        )
        # softplus trick
        next_state_del_log_var = self.max_logvar - F.softplus(
            self.max_logvar - next_state_del_log_var
        )
        next_state_del_log_var = self.min_logvar + F.softplus(
            next_state_del_log_var - self.min_logvar
        )

        reward = self.reward_fc(x)
        done = torch.sigmoid(self.done_fc(x))
        return next_state_del_mean, next_state_del_log_var, reward, done


class SimpleModelEnsemble(nn.Module):
    """
    A risk in model-based RL is that the agent will learn
    to exploit inaccuracies in the simulated/modeled env.
    (e.g. actions that give unrealisticly high rewards). To
    combat this, most methods use an ensemble of models. When
    doing rollouts with the simulated env, we pick a random model
    from the ensemble to make each transition prediction. This
    limits the agent's ability to overfit to any one particular model's
    bad predictions.
    """

    def __init__(self, ensemble):
        super().__init__()
        self.ensemble = ensemble

    def forward(self, state, action, *args, **kwargs):
        model = random.choice(self.ensemble)
        return model(state, action, *args, **kwargs)

    def to(self, device):
        for model in self.ensemble:
            model.to(device)

    def sample_with_replacement(self, transitions):
        num_samples = len(transitions[0])
        indxs = torch.randint(num_samples, (num_samples,))
        return tuple([x[indxs] for x in transitions])

    def fit(self, transitions, *args, **kwargs):
        for model in self.ensemble:
            transitions_i = self.sample_with_replacement(transitions)
            model.fit(transitions_i, *args, **kwargs)

    def save(self, path):
        for i, model in enumerate(self.ensemble):
            model_path = os.path.join(path, f"model_{i}.pt")
            torch.save(model.state_dict(), model_path)

    def load(self, path):
        for i, model in enumerate(self.ensemble):
            model_path = os.path.join(path, f"model_{i}.pt")
            model.load_state_dict(torch.load(model_path))
