import random

import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import loggers as pl_loggers
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from torch import nn

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
        """
        The network is minimizing the MSE between
        the predicted next state and the true next state,
        according to the real transitions. It also has to
        predict the reward and whether this transition ended
        the episode. Right now, I'm just adding those three losses
        up, but this is a bit of an open question because
        many of the papers in this area only learn the state dynamics
        and not the rewards or dones...
        """
        s, a, r, s1, d = batch
        pred_s1, pred_r, pred_d = self(s, a)
        loss = (
            F.binary_cross_entropy(pred_d, d)
            + F.mse_loss(pred_s1, s1)
            + F.mse_loss(pred_r, r)
        )
        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        s, a, r, s1, d = batch
        pred_s1, pred_r, pred_d = self(s, a)
        d_loss = F.binary_cross_entropy(pred_d, d)
        s_loss = F.mse_loss(pred_s1, s1)
        r_loss = F.mse_loss(pred_r, r)
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
        self._state_var = torch.var(s, axis=0).to(device)
        self._action_mean = torch.mean(a, axis=0).to(device)
        self._action_var = torch.mean(a, axis=0).to(device)
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
    and action and outputs a prediction for the next state,
    the reward and the done bool of that transition.

    Loosely based on the model used in ME-TRPO
    """

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._state_mean = torch.zeros((1, obs_dim)).to(device)
        self._action_mean = torch.zeros((1, act_dim)).to(device)
        self._state_var = torch.ones((1, obs_dim)).to(device)
        self._action_var = torch.ones((1, act_dim)).to(device)

        self.fc1 = nn.Linear(obs_dim + act_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.next_state_delta_fc = nn.Linear(1024, obs_dim)
        self.reward_fc = nn.Linear(1024, 1)
        self.done_fc = nn.Linear(1024, 1)

    def forward(self, state, action):
        state = (state - self._state_mean) / (self._state_var + 1e-5)
        action = (action - self._action_mean) / (self._action_var + 1e-5)
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        next_state = state + self.next_state_delta_fc(x)
        reward = self.reward_fc(x)
        done = torch.sigmoid(self.done_fc(x))
        return next_state, reward, done


def swish(x):
    return x * torch.sigmoid(x)


@variational_estimator
class BNN(DynamicsModel):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.bl1 = BayesianLinear(obs_dim + act_dim, 200)
        self.bl2 = BayesianLinear(200, 200)
        self.bl3 = BayesianLinear(200, 200)
        self.next_state_delta_bl = BayesianLinear(200, obs_dim)
        self.reward_bl = BayesianLinear(200, 1)
        self.done_bl = BayesianLinear(200, 1)

    def compute_loss(self, outputs, labels):
        pred_next_state, pred_reward, pred_done = outputs
        next_state, reward, done = labels
        loss = (
            F.binary_cross_entropy(pred_done, done)
            + F.mse_loss(pred_next_state, next_state)
            + F.mse_loss(pred_reward, reward)
        )
        return loss

    def forward(self, state_action, dummy_action=None):
        if dummy_action is None:
            state, action = state_action
        else:
            state, action = state_action, dummy_action
        state = (state - self._state_mean) / (self._state_var + 1e-5)
        action = (action - self._action_mean) / (self._action_var + 1e-5)
        x = torch.cat((state, action), dim=1)
        x = swish(self.bl1(x))
        x = swish(self.bl2(x))
        x = swish(self.bl3(x))
        next_state = state + self.next_state_delta_bl(x)
        reward = self.reward_bl(x)
        done = torch.sigmoid(self.done_bl(x))
        return next_state, reward, done

    def training_step(self, batch, batch_idx):
        s, a, r, s1, d = batch
        loss = self.sample_elbo(
            inputs=(s, a),
            labels=(s1, r, d),
            criterion=self.compute_loss,
            sample_nbr=3,
            complexity_cost_weight=1 / len(self._train_dset),
        )
        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        s, a, r, s1, d = batch
        pred_next_states, pred_rews, pred_dones = [], [], []
        for sample in range(10):
            pred_next_state, pred_rew, pred_done = self((s, a))
            pred_next_states.append(pred_next_state)
            pred_rews.append(pred_rew)
            pred_dones.append(pred_done)
        mean_pred_next_state = torch.stack(pred_next_states).mean(0)
        mean_pred_rew = torch.stack(pred_rews).mean(0)
        mean_pred_done = torch.stack(pred_dones).mean(0)

        d_loss = F.binary_cross_entropy(mean_pred_done, d)
        s_loss = F.mse_loss(mean_pred_next_state, s1)
        r_loss = F.mse_loss(mean_pred_rew, r)
        loss = d_loss + s_loss + r_loss

        return {
            "val_loss": loss,
            "done_loss": d_loss,
            "state_loss": s_loss,
            "rew_loss": r_loss,
        }


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

    def forward(self, state, action):
        model = random.choice(self.ensemble)
        return model(state, action)

    def to(self, device):
        for model in self.ensemble:
            model.to(device)

    def fit(self, *args, **kwargs):
        # TODO: parallelize this?
        for model in self.ensemble:
            model.fit(*args, **kwargs)
