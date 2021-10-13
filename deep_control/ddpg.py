import argparse
import copy
import os

import numpy as np
import tensorboardX
import torch
import torch.nn.functional as F
import tqdm

from . import envs, nets, replay, run, utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    def __init__(
        self,
        obs_space_size,
        action_space_size,
        actor_net_cls=nets.BaselineActor,
        critic_net_cls=nets.BaselineCritic,
        hidden_size=256,
    ):
        self.actor = actor_net_cls(
            obs_space_size, action_space_size, hidden_size=hidden_size
        )
        self.critic = critic_net_cls(
            obs_space_size, action_space_size, hidden_size=hidden_size
        )

    def to(self, device):
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def train(self):
        self.actor.train()
        self.critic.train()

    def save(self, path):
        actor_path = os.path.join(path, "actor.pt")
        critic_path = os.path.join(path, "critic.pt")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load(self, path):
        actor_path = os.path.join(path, "actor.pt")
        critic_path = os.path.join(path, "critic.pt")
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))

    def forward(self, state):
        state = self.process_state(state)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)
        self.actor.train()
        return np.squeeze(action.cpu().numpy(), 0)

    def process_state(self, state):
        return torch.from_numpy(np.expand_dims(state, 0).astype(np.float32)).to(
            utils.device
        )


def ddpg(
    agent,
    train_env,
    test_env,
    buffer,
    num_steps=1_000_000,
    transitions_per_step=1,
    max_episode_steps=100_000,
    batch_size=256,
    tau=0.005,
    actor_lr=1e-4,
    critic_lr=1e-3,
    gamma=0.99,
    sigma_start=0.2,
    sigma_final=0.1,
    sigma_anneal=100_000,
    theta=0.15,
    eval_interval=5000,
    eval_episodes=10,
    warmup_steps=1000,
    render=False,
    actor_clip=None,
    critic_clip=None,
    name="ddpg_run",
    actor_l2=0.0,
    critic_l2=0.0,
    save_interval=100_000,
    log_to_disk=True,
    save_to_disk=True,
    verbosity=0,
    gradient_updates_per_step=1,
    infinite_bootstrap=True,
    **_,
):
    """
    Train `agent` on `train_env` with the Deep Deterministic Policy Gradient algorithm,
    and evaluate on `test_env`.

    Reference: https://arxiv.org/abs/1509.02971
    """
    if save_to_disk or log_to_disk:
        # create save directory for this run
        save_dir = utils.make_process_dirs(name)
    if log_to_disk:
        # create tb writer, save hparams
        writer = tensorboardX.SummaryWriter(save_dir)
        writer.add_hparams(locals(), {})

    agent.to(device)

    # initialize target networks
    target_agent = copy.deepcopy(agent)
    target_agent.to(device)
    utils.hard_update(target_agent.actor, agent.actor)
    utils.hard_update(target_agent.critic, agent.critic)

    # Ornstein-Uhlenbeck is a controlled random walk used
    # to introduce noise for exploration. The DDPG paper
    # picks it over the simpler gaussian noise alternative,
    # but later work has shown this is an unnecessary detail.
    random_process = utils.OrnsteinUhlenbeckProcess(
        theta=theta,
        size=train_env.action_space.shape,
        sigma=sigma_start,
        sigma_min=sigma_final,
        n_steps_annealing=sigma_anneal,
    )

    critic_optimizer = torch.optim.Adam(
        agent.critic.parameters(), lr=critic_lr, weight_decay=critic_l2
    )
    actor_optimizer = torch.optim.Adam(
        agent.actor.parameters(), lr=actor_lr, weight_decay=actor_l2
    )

    # the replay buffer is filled with a few thousand transitions by
    # sampling from a uniform random policy, so that learning can begin
    # from a buffer that is >> the batch size.
    run.warmup_buffer(buffer, train_env, warmup_steps, max_episode_steps)

    done = True

    steps_iter = range(num_steps)
    if verbosity:
        # fancy progress bar
        steps_iter = tqdm.tqdm(steps_iter)

    for step in steps_iter:
        for _ in range(transitions_per_step):
            # collect experience from the environment, sampling from
            # the current policy (with added noise for exploration)
            if done:
                # reset the environment
                state = train_env.reset()
                random_process.reset_states()
                steps_this_ep = 0
                done = False
            action = agent.forward(state)
            noisy_action = run.exploration_noise(action, random_process)
            next_state, reward, done, info = train_env.step(noisy_action)
            if infinite_bootstrap:
                # allow infinite bootstrapping. Many envs terminate
                # (done = True) after an arbitrary number of steps
                # to let the agent reset and avoid getting stuck in
                # a failed position. infinite bootstrapping prevents
                # this from impacting our Q function calculation. This
                # can be harmful in edge cases where the environment really
                # would have ended (task failed) regardless of the step limit,
                # and makes no difference if the environment is not set up
                # to enforce a limit by itself (but many common benchmarks are).
                if steps_this_ep + 1 == max_episode_steps:
                    done = False
            # add this transition to the replay buffer
            buffer.push(state, noisy_action, reward, next_state, done)
            state = next_state
            steps_this_ep += 1
            if steps_this_ep >= max_episode_steps:
                # enforce max step limit from the agent's perspective
                done = True

        for _ in range(gradient_updates_per_step):
            # update the actor and critics using the replay buffer
            learn(
                buffer=buffer,
                target_agent=target_agent,
                agent=agent,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                batch_size=batch_size,
                gamma=gamma,
                critic_clip=critic_clip,
                actor_clip=actor_clip,
            )

            # move target models towards the online models
            # CC algorithms typically use a moving average rather
            # than the full copy of a DQN.
            utils.soft_update(target_agent.actor, agent.actor, tau)
            utils.soft_update(target_agent.critic, agent.critic, tau)

        if step % eval_interval == 0 or step == num_steps - 1:
            mean_return = run.evaluate_agent(
                agent, test_env, eval_episodes, max_episode_steps, render
            )
            if log_to_disk:
                writer.add_scalar("return", mean_return, step * transitions_per_step)
        if step % save_interval == 0 and save_to_disk:
            agent.save(save_dir)

    if save_to_disk:
        agent.save(save_dir)
    return agent


def learn(
    buffer,
    target_agent,
    agent,
    actor_optimizer,
    critic_optimizer,
    batch_size,
    gamma,
    critic_clip,
    actor_clip,
):
    """
    DDPG inner optimization loop. The simplest deep
    actor critic update.
    """
    # support for prioritized experience replay is
    # included in almost every algorithm in this repo. however,
    # it is somewhat rarely used in recent work because of its
    # extra hyperparameters and implementation complexity.
    per = isinstance(buffer, replay.PrioritizedReplayBuffer)
    if per:
        batch, imp_weights, priority_idxs = buffer.sample(batch_size)
        imp_weights = imp_weights.to(device)
    else:
        batch = buffer.sample(batch_size)

    # send transitions to the gpu
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
    state_batch = state_batch.to(device)
    next_state_batch = next_state_batch.to(device)
    action_batch = action_batch.to(device)
    reward_batch = reward_batch.to(device)
    done_batch = done_batch.to(device)

    ###################
    ## Critic Update ##
    ###################

    # compute target values
    with torch.no_grad():
        target_action_s1 = target_agent.actor(next_state_batch)
        target_action_value_s1 = target_agent.critic(next_state_batch, target_action_s1)
        # bootstrapped estimate of Q(s, a) based on reward and target network
        td_target = reward_batch + gamma * (1.0 - done_batch) * target_action_value_s1

    # compute mean squared bellman error (MSE(Q(s, a), td_target))
    agent_critic_pred = agent.critic(state_batch, action_batch)
    td_error = td_target - agent_critic_pred
    if per:
        critic_loss = (imp_weights * 0.5 * (td_error ** 2)).mean()
    else:
        critic_loss = 0.5 * (td_error ** 2).mean()
    critic_optimizer.zero_grad()
    # gradient descent step on critic network
    critic_loss.backward()
    if critic_clip:
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), critic_clip)
    critic_optimizer.step()

    ##################
    ## Actor Update ##
    ##################

    # actor's objective is to maximize (or minimize the negative of)
    # the expectation of the critic's opinion of its action choices
    agent_actions = agent.actor(state_batch)
    actor_loss = -agent.critic(state_batch, agent_actions).mean()
    actor_optimizer.zero_grad()
    # gradient descent step on actor network
    actor_loss.backward()
    if actor_clip:
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), actor_clip)
    actor_optimizer.step()

    if per:
        # update prioritized replay distribution
        new_priorities = (abs(td_error) + 1e-5).cpu().detach().squeeze(1).numpy()
        buffer.update_priorities(priority_idxs, new_priorities)


def add_args(parser):
    parser.add_argument(
        "--num_steps", type=int, default=1000000, help="number of training steps"
    )
    parser.add_argument(
        "--transitions_per_step",
        type=int,
        default=1,
        help="number of env steps per training step",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=100000,
        help="maximum steps per episode",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="training batch size"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="controls the speed that the target networks converge to the online networks",
    )
    parser.add_argument(
        "--actor_lr", type=float, default=1e-4, help="actor network learning rate"
    )
    parser.add_argument(
        "--critic_lr", type=float, default=1e-3, help="critic network learning rate"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="gamma, the MDP discount factor that determines emphasis on long-term rewards",
    )
    parser.add_argument(
        "--sigma_final",
        type=float,
        default=0.1,
        help="final sigma value for Ornstein Uhlenbeck exploration process",
    )
    parser.add_argument(
        "--sigma_anneal",
        type=float,
        default=100_000,
        help="How many steps to anneal sigma over.",
    )
    parser.add_argument(
        "--sigma_start",
        type=float,
        default=0.2,
        help="sigma for Ornstein Uhlenbeck exploration process",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.15,
        help="theta for Ornstein Uhlenbeck exploration process",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=5000,
        help="how often to test the agent without exploration (in steps)",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=10,
        help="how many episodes to run for when testing. results are averaged over this many episodes",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="how many random steps to take before learning begins",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="render the environment during training. can slow training significantly",
    )
    parser.add_argument(
        "--actor_clip",
        type=float,
        default=None,
        help="clip actor gradients based on this norm. less commonly used in actor critic algs than DQN",
    )
    parser.add_argument(
        "--critic_clip",
        type=float,
        default=None,
        help="clip critic gradients based on this norm. less commonly used in actor critic algs than DQN",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="ddpg_run",
        help="we will save the results of this training run in a directory called dc_saves/{this name}",
    )
    parser.add_argument(
        "--actor_l2",
        type=float,
        default=0.0,
        help="actor network L2 regularization coeff. Typically not helpful in single-environment settings",
    )
    parser.add_argument(
        "--critic_l2",
        type=float,
        default=0.0,
        help="critic network L2 regularization coeff. Typically not helpful in single-environment settings",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100_000,
        help="how often (in terms of steps) to save the network weights to disk",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="set to 0 for quiet mode (limit printing to std out). 1 shows a progress bar",
    )
    parser.add_argument(
        "--skip_save_to_disk",
        action="store_true",
        help="do not save the agent weights to disk during this training run",
    )
    parser.add_argument(
        "--skip_log_to_disk",
        action="store_true",
        help="do not write results to tensorboard during this training run",
    )
    parser.add_argument(
        "--gradient_updates_per_step",
        type=int,
        default=1,
        help="learning updates per training step (aka replay ratio denominator)",
    )
    parser.add_argument(
        "--prioritized_replay",
        action="store_true",
        help="Flag to enable prioritized experience replay",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=1_000_000,
        help="Maximum size of the replay buffer before oldest transitions are overwritten. Note that the default deep_control buffer allocates all of this space at the start of training to fail fast when there won't be enough space.",
    )
