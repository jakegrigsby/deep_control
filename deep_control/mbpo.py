import argparse
import copy
import math
from itertools import chain

import numpy as np
import tensorboardX
import torch
import tqdm

import deep_control as dc

from . import modeled_env, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_experience_from_model(
    agent, fake_env, policy_buffer, rollouts, max_rollout_length
):
    fake_env.reset_all()
    rollouts_complete = 0
    while rollouts_complete < rollouts:
        # generate the next transition
        state = fake_env.current_state()
        actions = agent.sample_action(state, from_cpu=False)
        next_states, rewards, dones, _ = fake_env.step(actions)
        mrn = rollouts - rollouts_complete  # max_rollouts_needed
        s = state[:mrn]
        a = actions[:mrn]
        r = rewards[:mrn]
        s1 = next_states[:mrn]
        d = dones[:mrn]
        policy_buffer.push(s, a, r, s1, d)

        # reset the env if done or if max_rollout_length reached
        reset_by_done = fake_env.reset_if_done()
        reset_by_step = fake_env.reset_on_step(max_rollout_length)
        rollouts_complete += reset_by_done + reset_by_step


def mbpo(
    agent,
    train_env,
    test_env,
    buffer,
    modelCls,
    ensemble_size=7,
    num_steps=1_000_000,
    model_buffer_size=100_000,
    warmup_steps=5000,
    real_env_steps_per_epoch=1000,
    model_rollouts_per_real_env_step=400,
    model_parallel_rollouts=400,
    policy_updates_per_real_env_step=20,
    max_rollout_length_start=1,
    max_rollout_length_final=15,
    max_rollout_length_anneal_start=20,
    max_rollout_length_anneal_end=100,
    tau=0.005,
    actor_lr=1e-4,
    critic_lr=1e-4,
    alpha_lr=1e-4,
    init_alpha=0.1,
    gamma=0.99,
    batch_size=256,
    eval_interval=5000,
    eval_episodes=10,
    actor_clip=None,
    critic_clip=None,
    actor_l2=0.0,
    critic_l2=0.0,
    target_delay=1,
    actor_delay=1,
    save_interval=100_000,
    name="mbpo_run",
    render=False,
    save_to_disk=True,
    log_to_disk=True,
    verbosity=0,
    max_episode_steps=100_000,
    model_val_split=0.2,
    model_lr=1e-3,
    model_early_stopping_patience=5,
    model_batch_size=256,
    model_max_epochs=100,
    model_l2=0.0001,
    infinite_bootstrap=True,
    **_,
):
    """
    Train `agent` on `train_env` using MBPO, and evaluate on `test_env`.

    Reference: https://arxiv.org/abs/1906.08253
    """
    if save_to_disk or log_to_disk:
        save_dir = dc.utils.make_process_dirs(name)
    if log_to_disk:
        writer = tensorboardX.SummaryWriter(save_dir)
        writer.add_hparams(locals(), {})

    assert max_rollout_length_start <= max_rollout_length_final

    # use an ensemble of models to prevent the policy from overfitting to any one
    # model's inaccuracies.
    ensemble = models.SimpleModelEnsemble(
        [
            modelCls(
                train_env.observation_space.shape[0], train_env.action_space.shape[0]
            )
            for i in range(ensemble_size)
        ]
    )

    # initialize target networks
    target_agent = copy.deepcopy(agent)
    target_agent.to(device)
    agent.to(device)
    dc.utils.hard_update(target_agent.critic1, agent.critic1)
    dc.utils.hard_update(target_agent.critic2, agent.critic2)

    # create optimizers
    critic_optimizer = torch.optim.Adam(
        chain(agent.critic1.parameters(), agent.critic2.parameters()),
        lr=critic_lr,
        weight_decay=critic_l2,
    )
    actor_optimizer = torch.optim.Adam(
        agent.actor.parameters(), lr=actor_lr, weight_decay=actor_l2
    )

    log_alpha = torch.Tensor([math.log(init_alpha)]).to(device)
    log_alpha.requires_grad = True

    log_alpha_optimizer = torch.optim.Adam([log_alpha], lr=alpha_lr)

    # the model buffer holds transitions from the real env that are used to
    # train the dynamics model(s)
    model_buffer = dc.replay.ReplayBuffer(
        model_buffer_size,
        train_env.observation_space.shape,
        train_env.action_space.shape,
        state_dtype=buffer.state_dtype,
    )

    # the policy buffer holds transitions from the fake env that are used to
    # train the agent, as in normal Soft Actor Critic
    policy_buffer = buffer

    # take random actions to collect experience for the dynamics model to train on the first time
    env_state = train_env.reset()
    env_done = False
    env_steps_this_ep = 0

    # the MBPO paper's theory component suggests short model rollouts starting
    # from any real state ("branching rollouts"). This rollout length can be
    # increased over time.
    current_max_rollout_length = int(max_rollout_length_start)
    max_rollout_length_slope = float(
        max_rollout_length_final - max_rollout_length_start
    ) / max((max_rollout_length_anneal_end - max_rollout_length_anneal_start), 1)

    # progress bar
    steps_iter = range(1, num_steps + 1)
    if verbosity > 0:
        steps_iter = tqdm.tqdm(steps_iter)

    # initial_agent_eval
    mean_return = dc.run.evaluate_agent(
        agent, test_env, eval_episodes, max_episode_steps, render
    )
    if log_to_disk:
        writer.add_scalar("return", mean_return, 0)

    env_done = True
    epoch = 0
    for step in steps_iter:
        # collect experience
        if env_done:
            env_state = train_env.reset()
            env_steps_this_ep = 0
            env_done = False
        action = agent.sample_action(env_state, from_cpu=True)
        env_next_state, env_reward, env_done, info = train_env.step(action)
        if infinite_bootstrap and env_steps_this_ep + 1 == max_episode_steps:
            # allow infinite bootstrapping
            env_done = False
        model_buffer.push(env_state, action, env_reward, env_next_state, env_done)
        env_state = env_next_state
        env_steps_this_ep += 1
        if env_steps_this_ep >= max_episode_steps:
            done = True

        if step < warmup_steps:
            continue

        if step % real_env_steps_per_epoch == 0 or step == warmup_steps:
            # time to retrain the dynamics models with new transitions from the model buffer
            ensemble.fit(
                model_buffer.get_all_transitions(),
                val_split=model_val_split,
                lr=model_lr,
                epochs=model_max_epochs,
                early_stopping_patience=model_early_stopping_patience,
                batch_size=model_batch_size,
                l2=model_l2,
                gpus=1,
            )

            if epoch >= max_rollout_length_anneal_start:
                # update max rollout length
                current_max_rollout_length = int(
                    min(
                        current_max_rollout_length + max_rollout_length_slope,
                        max_rollout_length_final,
                    )
                )
            epoch += 1

        fake_env = modeled_env.ParallelModeledEnv(
            buffer=model_buffer,
            model=ensemble,
            parallel_envs=model_parallel_rollouts,
            action_space=train_env.action_space,
            observation_space=train_env.observation_space,
        )
        collect_experience_from_model(
            agent,
            fake_env,
            policy_buffer,
            rollouts=model_rollouts_per_real_env_step,
            max_rollout_length=current_max_rollout_length,
        )

        for update_num in range(policy_updates_per_real_env_step):
            # use the inner optimization loop of the Soft Actor Critic algorithm
            dc.sac.learn(
                buffer=policy_buffer,
                target_agent=target_agent,
                agent=agent,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                log_alpha_optimizer=log_alpha_optimizer,
                batch_size=batch_size,
                log_alpha=log_alpha,
                target_entropy=-train_env.action_space.shape[0],
                gamma=gamma,
                critic_clip=critic_clip,
                actor_clip=actor_clip,
                update_policy=update_num % actor_delay == 0,
            )

            # move target model towards training model
            if update_num % target_delay == 0:
                dc.utils.soft_update(target_agent.critic1, agent.critic1, tau)
                dc.utils.soft_update(target_agent.critic2, agent.critic2, tau)

        if step % eval_interval == 0:
            # evaluate the agent on the real environemnt
            mean_return = dc.run.evaluate_agent(
                agent, test_env, eval_episodes, max_episode_steps, render
            )
            if log_to_disk:
                writer.add_scalar("return", mean_return, step)

        if step % save_interval == 0 and save_to_disk:
            agent.save(save_dir)

    if save_to_disk:
        agent.save(save_dir)

    return agent


def add_args(parser):
    parser.add_argument(
        "--modelCls",
        type=str,
        default="SimpleFeedForwardModel",
        help="Class name for dynamics model. See models.py",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100_000,
        help="number of steps in the training process",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=100000,
        help="maximum steps per episode",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=7,
        help="number of dynamics models in the ensemble",
    )
    parser.add_argument(
        "--real_env_steps_per_epoch",
        type=int,
        default=1000,
        help="how often (in steps) to retrain the dynamics model",
    )
    parser.add_argument(
        "--model_rollouts_per_real_env_step",
        type=int,
        default=400,
        help="how many ficticious (modeled) rollouts to collect per training step",
    )
    parser.add_argument(
        "--model_parallel_rollouts",
        type=int,
        default=400,
        help="how many rollouts of the model to perform at one time",
    )
    parser.add_argument(
        "--policy_updates_per_real_env_step",
        type=int,
        default=20,
        help="how many SAC gradient updates to make per training step",
    )
    parser.add_argument(
        "--max_rollout_length_start",
        type=int,
        default=1,
        help="length of modeled rollouts when training begins",
    )
    parser.add_argument(
        "--max_rollout_length_final",
        type=int,
        default=1,
        help="length of modeled rollouts after `max_rollout_length_anneal_end` epochs.",
    )
    parser.add_argument(
        "--max_rollout_length_anneal_start",
        type=int,
        default=1,
        help="Epoch # to start linearly increasing the rollout length. An epoch ends every `real_env_steps_per_epoch` training steps",
    )
    parser.add_argument(
        "--max_rollout_length_anneal_end",
        type=int,
        default=1,
        help="Epoch # that the rollout length ends its linear increase.",
    )
    parser.add_argument(
        "--model_buffer_size",
        type=int,
        default=100_000,
        help="How many transitions to store for training the dynamics models",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="SAC training batch size",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="target network polyak averaging value",
    )
    parser.add_argument(
        "--actor_lr",
        type=float,
        default=1e-4,
        help="actor learning rate",
    )
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=1e-4,
        help="critic learning rate",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="gamma, the discount factor",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=1_000_000,
        help="How many transitions to store for training the policy (as in normal SAC)",
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
        help="how many episodes to run for when testing. Reports average over these trials.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5000,
        help="Fill the model buffer with this many steps before training it the first time.",
    )
    parser.add_argument("--render", action="store_true", help="Render the env (slow)")
    parser.add_argument(
        "--actor_clip",
        type=float,
        default=None,
        help="Clip the actor gradients inside the SAC update.",
    )
    parser.add_argument(
        "--critic_clip",
        type=float,
        default=None,
        help="Clip the critic gradients inside the SAC update.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="mbpo_run",
        help="log files will be saved to `dc_saves/{name}`",
    )
    parser.add_argument(
        "--actor_l2",
        type=float,
        default=0.0,
        help="L2 regularization on agent's actor network.",
    )
    parser.add_argument(
        "--critic_l2",
        type=float,
        default=0.0,
        help="L2 regularization on agent's critic networks.",
    )
    parser.add_argument(
        "--target_delay",
        type=int,
        default=1,
        help="How many steps to go between target network updates",
    )
    parser.add_argument(
        "--actor_delay",
        type=int,
        default=1,
        help="How many steps to go between actor network updates",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10000,
        help="How often (in steps) to save the agent weights to disk",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="verbosity > 0 turns on a progress bar to watch during training.",
    )
    parser.add_argument(
        "--prioritized_replay",
        action="store_true",
        help="Enable prioritized experience replay",
    )
    parser.add_argument(
        "--skip_save_to_disk",
        action="store_true",
        help="Skips saving the agent weights to disk",
    )
    parser.add_argument(
        "--skip_log_to_disk",
        action="store_true",
        help="Skips saving the agent's training logs to disk.",
    )
    parser.add_argument(
        "--model_val_split",
        type=float,
        default=0.2,
        help="pct of transition data to use as val set during model training",
    )
    parser.add_argument(
        "--model_lr",
        type=float,
        default=1e-4,
        help="Learning rate for model training",
    )
    parser.add_argument(
        "--model_early_stopping_patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs) for model training",
    )
    parser.add_argument(
        "--model_batch_size",
        type=int,
        default=256,
        help="Batch size to use when training the model",
    )
    parser.add_argument(
        "--model_max_epochs",
        type=int,
        default=250,
        help="Max number of epochs to train each model for (if early stopping has not already ended training)",
    )
    parser.add_argument(
        "--alpha_lr",
        type=float,
        default=1e-4,
        help="Learning rate for alpha (entropy) coeff",
    )
    parser.add_argument(
        "--init_alpha",
        type=float,
        default=0.1,
        help="Initial value for alpha (entropy) coeff",
    )
    parser.add_argument(
        "--log_std_low",
        type=float,
        default=-10,
        help="Lower bound for log std of action distribution",
    )
    parser.add_argument(
        "--log_std_high",
        type=float,
        default=-2,
        help="Upper bound for log std of action distribution",
    )
