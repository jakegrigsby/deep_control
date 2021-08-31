import argparse
import copy
import math
import os
from itertools import chain

import gym
import numpy as np
import tensorboardX
import torch
import torch.nn.functional as F
import tqdm

from deep_control import envs, nets, replay, run, sac, utils
from deep_control.augmentations import AugmentationSequence, DrqAug

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PixelSACAgent(sac.SACAgent):
    def __init__(self, obs_shape, act_space_size, log_std_low, log_std_high):
        self.encoder = nets.BigPixelEncoder(obs_shape, out_dim=50)
        self.actor = nets.StochasticActor(50, act_space_size, log_std_low, log_std_high)
        self.critic1 = nets.BigCritic(50, act_space_size)
        self.critic2 = nets.BigCritic(50, act_space_size)
        self.log_std_low = log_std_low
        self.log_std_high = log_std_high

    def forward(self, obs):
        # eval forward (don't sample from distribution)
        obs = self.process_state(obs)
        self.encoder.eval()
        self.actor.eval()
        with torch.no_grad():
            state_rep = self.encoder.forward(obs)
            act_dist = self.actor.forward(state_rep)
            act = act_dist.mean
        self.encoder.train()
        self.actor.train()
        return self.process_act(act)

    def sample_action(self, obs):
        obs = self.process_state(obs)
        self.encoder.eval()
        self.actor.eval()
        with torch.no_grad():
            state_rep = self.encoder.forward(obs)
            act_dist = self.actor.forward(state_rep)
            act = act_dist.sample()
        self.encoder.train()
        self.actor.train()
        act = self.process_act(act)
        return act

    def to(self, device):
        self.encoder = self.encoder.to(device)
        super().to(device)

    def eval(self):
        self.encoder.eval()
        super().eval()

    def train(self):
        self.encoder.train()
        super().train()

    def save(self, path):
        encoder_path = os.path.join(path, "encoder.pt")
        torch.save(self.encoder.state_dict(), encoder_path)
        super().save(path)

    def load(self, path):
        encoder_path = os.path.join(path, "encoder.pt")
        self.encoder.load_state_dict(torch.load(encoder_path))
        super().load(path)


def sac_aug(
    agent,
    buffer,
    train_env,
    test_env,
    augmenter,
    num_steps=250_000,
    transitions_per_step=1,
    max_episode_steps=100_000,
    batch_size=256,
    mlp_tau=0.01,
    encoder_tau=0.05,
    actor_lr=1e-3,
    critic_lr=1e-3,
    encoder_lr=1e-3,
    alpha_lr=1e-4,
    gamma=0.99,
    eval_interval=10_000,
    test_eval_episodes=10,
    train_eval_episodes=0,
    warmup_steps=1000,
    actor_clip=None,
    critic_clip=None,
    actor_l2=0.0,
    critic_l2=0.0,
    encoder_l2=0.0,
    delay=2,
    save_interval=10_000,
    name="sac_aug_run",
    render=False,
    save_to_disk=True,
    log_to_disk=True,
    verbosity=0,
    gradient_updates_per_step=1,
    init_alpha=0.1,
    feature_matching_imp=0.0,
    aug_mix=1.0,
    infinite_bootstrap=True,
    **kwargs,
):
    if save_to_disk or log_to_disk:
        save_dir = utils.make_process_dirs(name)
    # create tb writer, save hparams
    if log_to_disk:
        writer = tensorboardX.SummaryWriter(save_dir)
        writer.add_hparams(locals(), {})

    agent.to(device)
    agent.train()

    # initialize target networks (target actor isn't used in SAC)
    target_agent = copy.deepcopy(agent)
    target_agent.to(device)
    utils.hard_update(target_agent.critic1, agent.critic1)
    utils.hard_update(target_agent.critic2, agent.critic2)
    utils.hard_update(target_agent.encoder, agent.encoder)
    target_agent.train()

    # create network optimizers
    critic_optimizer = torch.optim.Adam(
        chain(
            agent.critic1.parameters(),
            agent.critic2.parameters(),
        ),
        lr=critic_lr,
        weight_decay=critic_l2,
        betas=(0.9, 0.999),
    )
    encoder_optimizer = torch.optim.Adam(
        agent.encoder.parameters(),
        lr=encoder_lr,
        weight_decay=encoder_l2,
        betas=(0.9, 0.999),
    )
    actor_optimizer = torch.optim.Adam(
        agent.actor.parameters(),
        lr=actor_lr,
        weight_decay=actor_l2,
        betas=(0.9, 0.999),
    )

    # initialize learnable alpha param
    log_alpha = torch.Tensor([math.log(init_alpha)]).to(device)
    log_alpha.requires_grad = True
    log_alpha_optimizer = torch.optim.Adam([log_alpha], lr=alpha_lr, betas=(0.5, 0.999))

    target_entropy = -train_env.action_space.shape[0]

    run.warmup_buffer(buffer, train_env, warmup_steps, max_episode_steps)

    done = True
    steps_this_ep = 0

    steps_iter = range(num_steps)
    if verbosity:
        steps_iter = tqdm.tqdm(steps_iter)

    for step in steps_iter:
        for _ in range(transitions_per_step):
            if done:
                obs = train_env.reset()
                steps_this_ep = 0
                done = False
            # batch the actions
            action = agent.sample_action(obs)
            next_obs, reward, done, info = train_env.step(action)
            if infinite_bootstrap and steps_this_ep + 1 == max_episode_steps:
                # allow infinite bootstrapping
                done = False
            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            steps_this_ep += 1
            if steps_this_ep >= max_episode_steps:
                done = True

        update_policy = step % delay == 0
        for _ in range(gradient_updates_per_step):
            learn_from_pixels(
                buffer=buffer,
                target_agent=target_agent,
                agent=agent,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                encoder_optimizer=encoder_optimizer,
                log_alpha=log_alpha,
                log_alpha_optimizer=log_alpha_optimizer,
                target_entropy=target_entropy,
                batch_size=batch_size,
                gamma=gamma,
                critic_clip=critic_clip,
                actor_clip=actor_clip,
                update_policy=update_policy,
                augmenter=augmenter,
                feature_matching_imp=feature_matching_imp,
                aug_mix=aug_mix,
            )

            # move target model towards training model
            if update_policy:
                utils.soft_update(target_agent.critic1, agent.critic1, mlp_tau)
                utils.soft_update(target_agent.critic2, agent.critic2, mlp_tau)
                utils.soft_update(target_agent.encoder, agent.encoder, encoder_tau)

        if step % eval_interval == 0 or step == num_steps - 1:
            mean_test_return = run.evaluate_agent(
                agent, test_env, test_eval_episodes, max_episode_steps, render
            )
            mean_train_return = run.evaluate_agent(
                agent, train_env, train_eval_episodes, max_episode_steps, render
            )
            if log_to_disk:
                writer.add_scalar(
                    "performance/test_return",
                    mean_test_return,
                    step * transitions_per_step,
                )
                writer.add_scalar(
                    "performance/train_return",
                    mean_train_return,
                    step * transitions_per_step,
                )

        if step % save_interval == 0 and save_to_disk:
            agent.save(save_dir)

    if save_to_disk:
        agent.save(save_dir)
    return agent


def learn_from_pixels(
    buffer,
    target_agent,
    agent,
    actor_optimizer,
    critic_optimizer,
    encoder_optimizer,
    log_alpha_optimizer,
    target_entropy,
    log_alpha,
    augmenter,
    batch_size=128,
    gamma=0.99,
    critic_clip=None,
    actor_clip=None,
    update_policy=True,
    feature_matching_imp=1.0,
    aug_mix=0.75,
):

    per = isinstance(buffer, replay.PrioritizedReplayBuffer)
    if per:
        batch, imp_weights, priority_idxs = buffer.sample(batch_size)
        imp_weights = imp_weights.to(device)
    else:
        batch = buffer.sample(batch_size)

    # sample unaugmented transitions from the buffer
    og_obs_batch, action_batch, reward_batch, og_next_obs_batch, done_batch = batch
    og_obs_batch = og_obs_batch.to(device)
    og_next_obs_batch = og_next_obs_batch.to(device)
    # at this point, the obs batches are float32s [0., 255.] on the gpu

    # created an augmented version of each transition
    # the augmenter applies a random transition to each batch index,
    # but keep the random params consistent between obs and next_obs batches
    aug_obs_batch, aug_next_obs_batch = augmenter(og_obs_batch, og_next_obs_batch)

    # mix the augmented versions in with the standard
    # no need to shuffle because the replay buffer handles that
    aug_mix_idx = int(batch_size * aug_mix)
    obs_batch = og_obs_batch.clone()
    obs_batch[:aug_mix_idx] = aug_obs_batch[:aug_mix_idx]
    next_obs_batch = og_next_obs_batch.clone()
    next_obs_batch[:aug_mix_idx] = aug_next_obs_batch[:aug_mix_idx]

    action_batch = action_batch.to(device)
    reward_batch = reward_batch.to(device)
    done_batch = done_batch.to(device)

    alpha = torch.exp(log_alpha)

    with torch.no_grad():
        # create critic targets (clipped double Q learning)
        next_state_rep = target_agent.encoder(next_obs_batch)
        action_dist_s1 = agent.actor(next_state_rep)
        action_s1 = action_dist_s1.rsample()
        logp_a1 = action_dist_s1.log_prob(action_s1).sum(-1, keepdim=True)

        target_action_value_s1 = torch.min(
            target_agent.critic1(next_state_rep, action_s1),
            target_agent.critic2(next_state_rep, action_s1),
        )
        td_target = reward_batch + gamma * (1.0 - done_batch) * (
            target_action_value_s1 - (alpha * logp_a1)
        )

    # update critics with Bellman MSE
    state_rep = agent.encoder(obs_batch)
    agent_critic1_pred = agent.critic1(state_rep, action_batch)
    td_error1 = td_target - agent_critic1_pred
    if per:
        critic1_loss = (imp_weights * 0.5 * (td_error1 ** 2)).mean()
    else:
        critic1_loss = 0.5 * (td_error1 ** 2).mean()

    agent_critic2_pred = agent.critic2(state_rep, action_batch)
    td_error2 = td_target - agent_critic2_pred
    if per:
        critic2_loss = (imp_weights * 0.5 * (td_error2 ** 2)).mean()
    else:
        critic2_loss = 0.5 * (td_error2 ** 2).mean()

    # optional feature matching loss to make state_rep invariant to augs
    if feature_matching_imp > 0.0:
        aug_rep = agent.encoder(aug_obs_batch)
        with torch.no_grad():
            og_rep = agent.encoder(og_obs_batch)
        fm_loss = torch.norm(aug_rep - og_rep)
    else:
        fm_loss = 0.0

    critic_loss = critic1_loss + critic2_loss + feature_matching_imp * fm_loss

    critic_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    critic_loss.backward()
    if critic_clip:
        torch.nn.utils.clip_grad_norm_(
            chain(
                agent.critic1.parameters(),
                agent.critic2.parameters(),
            ),
            critic_clip,
        )
    critic_optimizer.step()
    encoder_optimizer.step()

    if update_policy:
        # actor update
        dist = agent.actor(state_rep.detach())
        agent_actions = dist.rsample()
        logp_a = dist.log_prob(agent_actions).sum(-1, keepdim=True)

        actor_loss = -(
            torch.min(
                agent.critic1(state_rep.detach(), agent_actions),
                agent.critic2(state_rep.detach(), agent_actions),
            )
            - (alpha.detach() * logp_a)
        ).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        if actor_clip:
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), actor_clip)
        actor_optimizer.step()

        # alpha update
        alpha_loss = (-alpha * (logp_a + target_entropy).detach()).mean()
        log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        log_alpha_optimizer.step()

    if per:
        new_priorities = (abs(td_error1) + 1e-5).cpu().data.squeeze(1).numpy()
        buffer.update_priorities(priority_idxs, new_priorities)


def add_args(parser):
    parser.add_argument(
        "--num_steps",
        type=int,
        default=250_000,
        help="Number of training steps.",
    )
    parser.add_argument(
        "--transitions_per_step",
        type=int,
        default=1,
        help="Env transitions per training step. Defaults to 1, but will need to \
        be set higher for repaly ratios < 1",
    )
    parser.add_argument(
        "--max_episode_steps_start",
        type=int,
        default=1000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--max_episode_steps_final",
        type=int,
        default=1000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--max_episode_steps_anneal",
        type=float,
        default=0.4,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument(
        "--mlp_tau",
        type=float,
        default=0.01,
        help="Determines how quickly the target agent's critic networks params catch up to the trained agent.",
    )
    parser.add_argument(
        "--encoder_tau",
        type=float,
        default=0.05,
        help="Determines how quickly the target agent's encoder network params catch up to the trained agent. This is typically set higher than mlp_tau because the encoder is used in both actor and critic updates.",
    )
    parser.add_argument(
        "--actor_lr",
        type=float,
        default=1e-3,
        help="Actor network learning rate",
    )
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=1e-3,
        help="Critic networks' learning rate",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="POMDP discount factor",
    )
    parser.add_argument(
        "--init_alpha",
        type=float,
        default=0.1,
        help="Initial entropy regularization coefficeint.",
    )
    parser.add_argument(
        "--alpha_lr",
        type=float,
        default=1e-4,
        help="Alpha (entropy regularization coefficeint) learning rate",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=100_000,
        help="Replay buffer maximum capacity. Note that image observations can take up a lot of memory, especially when using frame stacking. The buffer allocates a large tensor of zeros to fail fast if it will not have enough memory to complete the training run.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=10_000,
        help="How often to test the agent without exploration (in training steps)",
    )
    parser.add_argument(
        "--test_eval_episodes",
        type=int,
        default=10,
        help="How many episodes to run for when evaluating on the testing set",
    )
    parser.add_argument(
        "--train_eval_episodes",
        type=int,
        default=10,
        help="How many episodes to run for when evaluating on the training set",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Number of uniform random actions to take at the beginning of training",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Flag to enable env rendering during training",
    )
    parser.add_argument(
        "--actor_clip",
        type=float,
        default=None,
        help="Gradient clipping for actor updates",
    )
    parser.add_argument(
        "--critic_clip",
        type=float,
        default=None,
        help="Gradient clipping for critic updates",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="pixel_sac_run",
        help="Dir name for saves, (look in ./dc_saves/{name})",
    )
    parser.add_argument(
        "--actor_l2",
        type=float,
        default=0.0,
        help="L2 regularization coeff for actor network",
    )
    parser.add_argument(
        "--critic_l2",
        type=float,
        default=0.0,
        help="L2 regularization coeff for critic network",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=2,
        help="How many steps to go between actor and target agent updates",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10_000,
        help="How many steps to go between saving the agent params to disk",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="Verbosity > 0 displays a progress bar during training",
    )
    parser.add_argument(
        "--gradient_updates_per_step",
        type=int,
        default=1,
        help="How many gradient updates to make per training step",
    )
    parser.add_argument(
        "--prioritized_replay",
        action="store_true",
        help="Flag that enables use of prioritized experience replay",
    )
    parser.add_argument(
        "--skip_save_to_disk",
        action="store_true",
        help="Flag to skip saving agent params to disk during training",
    )
    parser.add_argument(
        "--skip_log_to_disk",
        action="store_true",
        help="Flag to skip saving agent performance logs to disk during training",
    )
    parser.add_argument(
        "--feature_matching_imp",
        type=float,
        default=0.0,
        help="Coefficient for feature matching loss",
    )
    parser.add_argument(
        "--encoder_lr",
        type=float,
        default=1e-3,
        help="Learning rate for the encoder network",
    )
    parser.add_argument(
        "--encoder_l2",
        type=float,
        default=0.0,
        help="Weight decay coefficient for pixel encoder network",
    )
    parser.add_argument(
        "--aug_mix",
        type=float,
        default=1.0,
        help="Fraction of each update batch that is made up of augmented samples",
    )
    parser.add_argument(
        "--log_std_low",
        type=int,
        default=-10,
        help="Lower bound for log std of action distribution.",
    )
    parser.add_argument(
        "--log_std_high",
        type=int,
        default=2,
        help="Upper bound for log std of action distribution.",
    )
    parser.add_argument(
        "--augmentations",
        type=str,
        default="[DrqAug]",
        help="Sequence of image data augmentations to perform during training. e.g [ColorJitterAug,DrQAug]",
    )
