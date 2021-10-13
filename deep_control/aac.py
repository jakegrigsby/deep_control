"""
A (temporarily) messy port of the Automatic Actor Critic codebase (https://arxiv.org/abs/2106.08918)
into the rest of deep_control.
"""

from functools import total_ordering
from dataclasses import dataclass
import random
import os
import math
import warnings
from itertools import chain
from collections import namedtuple

warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
)

import tensorboardX
import tqdm
import torch
from torch import multiprocessing as mp
import numpy as np

import deep_control as dc
from deep_control import run, replay, nets, device
from deep_control.envs import PersistenceAwareWrapper


def learn_critics(member, buffer, batch_size, gamma):

    agent = member.agent
    batch = buffer.sample(batch_size)

    # prepare transitions for models
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
    state_batch = state_batch.to(device)
    next_state_batch = next_state_batch.to(device)
    action_batch = action_batch.to(device)
    reward_batch = reward_batch.to(device)
    done_batch = done_batch.to(device)

    agent.train()

    ###################
    ## CRITIC UPDATE ##
    ###################
    alpha = torch.exp(agent.log_alpha)
    with torch.no_grad():
        action_dist_s1 = agent.actor(next_state_batch)
        action_s1 = action_dist_s1.rsample()
        logp_a1 = action_dist_s1.log_prob(action_s1).sum(-1, keepdim=True)
        y1 = agent.critic1(next_state_batch, action_s1)
        y2 = agent.critic2(next_state_batch, action_s1)
        clipped_double_q_s1 = torch.min(y1, y2)
        gammas = (
            torch.Tensor([gamma ** i for i in range(reward_batch.shape[-1])])
            .unsqueeze(0)
            .to(device)
        )
        discounted_rews = (gammas * reward_batch).sum(1, keepdim=True)
        action_repeat = state_batch[:, 0]
        multistep_gamma = (gamma ** action_repeat).unsqueeze(1)
        td_target = discounted_rews + multistep_gamma * (1.0 - done_batch) * (
            clipped_double_q_s1 - (alpha * logp_a1)
        )

    # standard bellman error
    a_critic1_pred = agent.critic1(state_batch, action_batch)
    a_critic2_pred = agent.critic2(state_batch, action_batch)
    td_error1 = td_target - a_critic1_pred
    td_error2 = td_target - a_critic2_pred

    # constraints that discourage large changes in Q(s_{t+1}, a_{t+1}),
    a1_critic1_pred = agent.critic1(next_state_batch, action_s1)
    a1_critic2_pred = agent.critic2(next_state_batch, action_s1)
    a1_constraint1 = y1 - a1_critic1_pred
    a1_constraint2 = y2 - a1_critic2_pred

    elementwise_critic_loss = (
        (td_error1 ** 2)
        + (td_error2 ** 2)
        + (a1_constraint1 ** 2)
        + (a1_constraint2 ** 2)
    )
    critic_loss = 0.5 * elementwise_critic_loss.mean()
    agent.critic_optimizer.zero_grad()
    critic_loss.backward()
    agent.critic_optimizer.step()


def learn_actor(
    member,
    buffer,
    batch_size,
    target_entropy_mul,
):
    agent = member.agent
    agent.train()

    batch = buffer.sample(batch_size)
    s, a, *_ = batch
    s = s.to(device)
    a = a.to(device)

    dist = agent.actor(s)
    agent_a = dist.rsample()
    logp_a = dist.log_prob(agent_a).sum(-1, keepdim=True)
    vals = torch.min(agent.critic1(s, agent_a), agent.critic2(s, agent_a))
    entropy_bonus = agent.log_alpha.exp() * logp_a
    actor_loss = -(vals - entropy_bonus).mean()

    optimizer = agent.online_actor_optimizer
    optimizer.zero_grad()
    actor_loss.backward()
    optimizer.step()

    ##################
    ## ALPHA UPDATE ##
    ##################
    target_entropy = target_entropy_mul * -float(a.shape[1])
    alpha_loss = (-agent.log_alpha.exp() * (logp_a + target_entropy).detach()).mean()
    agent.log_alpha_optimizer.zero_grad()
    alpha_loss.backward()
    agent.log_alpha_optimizer.step()


def collect_experience(member, env_wrapper, buffer=None):
    """
    AAC's custom collect_experience function depends
    on the EnvironmentWrapper interface (see below)
    """
    if env_wrapper.done:
        env_wrapper.state = env_wrapper.env.reset()
        env_wrapper.step_count = 0
        env_wrapper.done = False

    state = env_wrapper.state
    action = member.agent.sample_action(state)
    next_state, reward, done, info = env_wrapper.env.step(action)

    # infinite bootstrapping
    if env_wrapper.step_count + 1 == env_wrapper.max_episode_steps:
        done = False
    if buffer:
        # go ahead and push to the buffer
        buffer.push(state, action, reward, next_state, done)

    # prep for next iteration
    env_wrapper.state = next_state
    env_wrapper.step_count += 1
    env_wrapper.done = done
    if env_wrapper.step_count >= env_wrapper.max_episode_steps:
        env_wrapper.done = True

    if buffer:
        return None
    else:
        return state, action, reward, next_state, done


class AACAgent:
    def __init__(
        self,
        obs_space_size,
        act_space_size,
        log_std_low=-10.0,
        log_std_high=2.0,
        actor_net_cls=nets.StochasticActor,
        critic_net_cls=nets.BigCritic,
        hidden_size=256,
    ):
        self.actor = actor_net_cls(
            obs_space_size,
            act_space_size,
            log_std_low,
            log_std_high,
            dist_impl="pyd",
            hidden_size=hidden_size,
        )
        self.critic1 = critic_net_cls(obs_space_size, act_space_size, hidden_size)
        self.critic2 = critic_net_cls(obs_space_size, act_space_size, hidden_size)

        self.critic_optimizer = torch.optim.Adam(
            chain(
                self.critic1.parameters(),
                self.critic2.parameters(),
            ),
            lr=3e-4,
        )
        self.online_actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=3e-4,
        )
        # Trick to make it easier to reload `log_alpha`'s optimizer when changing devices.
        self._log_alpha = torch.nn.Linear(1, 1, bias=False)
        self.log_alpha = torch.Tensor([math.log(0.1)])
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=1e-4, betas=(0.5, 0.999)
        )
        self.target_entropy = -float(act_space_size)

    @property
    def log_alpha(self):
        return self._log_alpha.weight

    @log_alpha.setter
    def log_alpha(self, v):
        assert isinstance(v, torch.Tensor)
        self._log_alpha.weight = torch.nn.Parameter(v)

    def to(self, device):
        self.actor.to(device)
        self.critic1.to(device)
        self.critic2.to(device)

        # Reload state_dict of optimizer to account for device change
        # From https://github.com/pytorch/pytorch/issues/8741
        self.critic_optimizer.load_state_dict(self.critic_optimizer.state_dict())
        self.online_actor_optimizer.load_state_dict(
            self.online_actor_optimizer.state_dict()
        )
        self._log_alpha.to(device)
        self.log_alpha_optimizer.load_state_dict(self.log_alpha_optimizer.state_dict())

    def share_memory_(self):
        self.actor.share_memory()
        self.critic1.share_memory()
        self.critic2.share_memory()
        self.log_alpha.share_memory_()

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        if not self.log_alpha.requires_grad:
            self.log_alpha.requires_grad = True

    def save(self, path, id_):
        actor_path = os.path.join(path, f"actor_{id_}.pt")
        critic1_path = os.path.join(path, f"critic1_{id_}.pt")
        critic2_path = os.path.join(path, f"critic2_{id_}.pt")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic1.state_dict(), critic1_path)
        torch.save(self.critic2.state_dict(), critic2_path)

    def load(self, path, id_):
        actor_path = os.path.join(path, f"actor_{id_}.pt")
        critic1_path = os.path.join(path, f"critic1_{id_}.pt")
        critic2_path = os.path.join(path, f"critic2_{id_}.pt")
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic1.load_state_dict(torch.load(critic1_path))
        self.critic2.load_state_dict(torch.load(critic2_path))

    def forward(self, state, from_cpu=True):
        if from_cpu:
            state = self.process_state(state)
        self.actor.eval()
        with torch.no_grad():
            act_dist = self.actor.forward(state)
            act = act_dist.mean
        self.actor.train()
        if from_cpu:
            act = self.process_act(act)
        return act

    def sample_action(self, state, from_cpu=True):
        if from_cpu:
            state = self.process_state(state)
        self.actor.eval()
        with torch.no_grad():
            act_dist = self.actor.forward(state)
            act = act_dist.sample()
        self.actor.train()
        if from_cpu:
            act = self.process_act(act)
        return act

    def process_state(self, state):
        return torch.from_numpy(np.expand_dims(state, 0).astype(np.float32)).to(device)

    def process_act(self, act):
        return np.squeeze(act.clamp(-1.0, 1.0).cpu().numpy(), 0)


@dataclass
class Hparams:
    # actor updates per step
    a: int
    # critic updates per step
    c: int
    # action persistence
    k: int
    # target entropy
    h: float
    # discount factor
    g: float


@total_ordering
class Member:
    """Each population member contains actor (which contains actor and critic(s)) and hparams."""

    def __init__(self, uid, agent, hparams):
        self.id = uid
        self.agent = agent
        self.hparams = hparams
        self.fitness = -float("inf")

    def __eq__(self, other):
        return self.fitness == other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness


class EnvironmentWrapper:
    def __init__(self, uid, make_env_function, max_episode_steps):
        self.id = uid
        self.make_env_function = make_env_function
        self._env = None
        self.done = True
        self.step_count = 0
        self._max_episode_steps = max_episode_steps

    @property
    def env(self):
        # Lazy instantiation of environment
        if self._env is None:
            self._env = self.make_env_function()
            self.state = self._env.reset()
        return self._env

    @property
    def max_episode_steps(self):
        return round(self._max_episode_steps / self.env.k)

    def set_k(self, new_k):
        self.env.set_k(new_k)


class Worker(mp.Process):
    """Each worker is reponsible for collecting experience, training, and evaluating each member."""

    def __init__(
        self,
        uid,
        make_env_function,
        max_episode_steps,
        replay_buffer,
        member_queue,
        exp_queue,
        step_events,
        epoch_events,
        epochs,
        steps_per_epoch,
        batch_size,
        num_gpus,
        eval_episodes=10,
    ):
        super().__init__()
        self.id = uid
        self.train_env_wrapper = EnvironmentWrapper(
            uid, make_env_function, max_episode_steps
        )
        self.test_env_wrapper = EnvironmentWrapper(
            uid, make_env_function, max_episode_steps
        )
        self.replay_buffer = replay_buffer
        self.member_queue = member_queue
        self.exp_queue = exp_queue
        self.step_events = step_events
        self.epoch_events = epoch_events
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.eval_episodes = eval_episodes

    def run(self):
        # Set sharing strategy to avoid errors when sharing agents to main process.
        torch.multiprocessing.set_sharing_strategy("file_system")
        gpu_id = (
            torch.multiprocessing.current_process()._identity[0] - 1
        ) % self.num_gpus
        torch.cuda.set_device(gpu_id)

        for epoch in range(self.epochs):
            _uid, member = self.member_queue.get()
            assert _uid == self.id, "Worker id and member id mismatch."

            # `requires_grad` needs to be set to `False` before sending it to GPU.
            member.agent.log_alpha.requires_grad = False
            member.agent.to(dc.device)

            # Make sure environment uses member's k.
            self.train_env_wrapper.set_k(member.hparams.k)

            for step in range(self.steps_per_epoch):
                exp = collect_experience(member, self.train_env_wrapper)
                self.exp_queue.put((self.id, exp))

                # Wait until main process is done modifying the replay buffer
                if step % 2 == 0:
                    self.step_events[0].wait()
                else:
                    self.step_events[1].wait()

                # Do critic updates
                for _ in range(member.hparams.c):
                    learn_critics(
                        member=member,
                        buffer=self.replay_buffer,
                        batch_size=self.batch_size,
                        gamma=1.0 - math.exp(member.hparams.g),
                    )

                # Do actor updates
                for _ in range(member.hparams.a):
                    learn_actor(
                        member=member,
                        buffer=self.replay_buffer,
                        batch_size=self.batch_size,
                        target_entropy_mul=member.hparams.h,
                    )

            # Evaluate fitness of population members using the test env.
            self.test_env_wrapper.env.reset()
            self.test_env_wrapper.set_k(member.hparams.k)
            member.fitness = run.evaluate_agent(
                member.agent,
                self.test_env_wrapper.env,
                self.eval_episodes,
                self.train_env_wrapper.max_episode_steps,
            )

            member.agent.log_alpha.requires_grad = False
            member.agent.to(torch.device("cpu"))
            self.member_queue.put((_uid, member))

            # Wait until main process is done crossover-ing bad and elite members.
            if epoch % 2 == 0:
                self.epoch_events[0].wait()
            else:
                self.epoch_events[1].wait()


class WorkerPool:
    """Simple class to handle pool of workers"""

    def __init__(self, workers, replay_buffer, member_queues, exp_queue):
        self.workers = workers
        self.replay_buffer = replay_buffer
        self.member_queues = member_queues
        self.exp_queue = exp_queue

    def start(self):
        for w in self.workers:
            w.start()

    def join(self):
        for w in self.workers:
            w.join()

    def close(self):
        for w in self.workers:
            w.close()

    def get_population(self):
        population = []
        for worker in self.workers:
            _id, member = self.member_queues[worker.id].get()
            population.append(member)
        return population

    def collect_experiences(self):
        for _ in self.workers:
            _, result = self.exp_queue.get(block=True)
            self.replay_buffer.push(*result)


ParamSpace = namedtuple("ParamSpace", ["min", "max", "delta"])


def aac(
    make_env_function,
    epochs=1_000,
    steps_per_epoch=1_000,
    population_size=20,
    a_param=(1, 10, 2),
    c_param=(1, 40, 5),
    h_param=(0.25, 1.75, 0.25),
    k_param=(1, 15, 2),
    g_param=(-6.5, -1.0, 0.5),
    max_episode_steps=1000,
    eval_episodes=10,
    batch_size=512,
    hidden_size=256,
    name="aac_run",
    **_,
):
    """Parallel Implementation of AAC

    Args:
        make_env_function (callable):
            Zero-argument callable that returns environment as `PersistenceAwareWrapper`.
        epochs (int, optional):
            Evolutionary epochs Defaults to 1_000.
        steps_per_epoch (int, optional):
            Training steps per epoch. Defaults to 1_000.
        population_size (int, optional):
            Population size. Defaults to 20.
        a_param (tuple[int, int, int], optional):
            Tuple of min, max, and delta value for hyperparameter `a`. Defaults to (1, 10, 2).
        c_param (tuple, optional):
            Tuple of min, max, and delta value for hyperparameter `c`. Defaults to (1, 40, 5).
        h_param (tuple, optional):
            Tuple of min, max, and delta value for hyperparameter `h`. Defaults to (0.25, 1.75, 0.25).
        k_param (tuple, optional):
            Tuple of min, max, and delta value for hyperparameter `k`. Defaults to (1, 15, 2).
        g_param (tuple, optional):
            Tuple of min, max, and delta value for hyperparameter `g`. Defaults to (-6.5, -1.0, 0.5).
        max_episode_steps (int, optional):
            Maximum number of steps for an episode. Defaults to 1000.
        batch_size (int, optional):
            Batch size of experiences from replay buffer used for training. Defaults to 512.
        name (str, optional):
            Name of run. Used for logging to Tensorboard. Defaults to "parallel_pbt_ac".
    """
    a_param = ParamSpace(*a_param)
    c_param = ParamSpace(*c_param)
    h_param = ParamSpace(*h_param)
    k_param = ParamSpace(*k_param)
    g_param = ParamSpace(*g_param)

    test_env = make_env_function()
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    save_dir = dc.utils.make_process_dirs(name)
    writer = tensorboardX.SummaryWriter(save_dir)
    writer.add_hparams(locals(), {})

    # Lambda funcs related to genetic crossover
    clamp = lambda x, param: max(min(x, param.max), param.min)
    shift_int_by_add = lambda current, param: current + random.randint(
        -param.delta, param.delta
    )
    shift_float_by_add = lambda current, param: current + random.uniform(
        -param.delta, param.delta
    )
    make_int_range = lambda x: random.randint(x.min, x.max)
    make_float_range = lambda x: random.uniform(x.min, x.max)

    # Create a centralized replay buffer and add a few random samples to get started
    buffer_size = 2_000_000
    replay_buffer = PersistenceReplayBuffer(
        size=buffer_size,
        state_shape=obs_space.shape,
        action_repeat=k_param.max,
        state_dtype=float,
        action_shape=act_space.shape,
    )

    print("Warm up replay buffer...")
    warmup_size = 10_000
    rand_env = make_env_function()
    pbar = tqdm.tqdm(total=warmup_size, dynamic_ncols=True)
    while len(replay_buffer) < warmup_size:
        # Collect random samples at each action repeat value
        for k in range(1, k_param.max + 1):
            prev = len(replay_buffer)
            rand_env.reset()
            rand_env.set_k(k)
            max_steps = round(max_episode_steps / k)
            run.warmup_buffer(
                replay_buffer,
                rand_env,
                max_steps + 1,
                max_steps,
            )
            if len(replay_buffer) >= warmup_size:
                pbar.update(warmup_size - prev)
                break
            else:
                pbar.update(len(replay_buffer) - prev)
    pbar.close()

    # Initialize the population
    population = []
    for i in range(population_size):
        agent = AACAgent(
            obs_space.shape[0], act_space.shape[0], hidden_size=hidden_size
        )
        hparams = Hparams(
            make_int_range(a_param),
            make_int_range(c_param),
            make_int_range(k_param),
            make_float_range(h_param),
            make_float_range(g_param),
        )
        member = Member(i, agent, hparams)
        population.append(member)

    # Moving replay buffer to shared memory allows us to share it among workers without copying.
    replay_buffer.share_memory_()
    # Separate queue is created for each member to ensure correct member is sent to each worker.
    member_queues = {m.id: mp.Queue() for m in population}
    # Queue for sharing collect experiences
    exp_queue = mp.Queue()

    # Events for synchronization. Two different events are used to avoid possible race conditions.
    # Specifically, we need to clear each event before reusing it, but we don't want to clear it too early.
    step_events = (mp.Event(), mp.Event())
    epoch_events = (mp.Event(), mp.Event())
    num_gpus = torch.cuda.device_count()

    # Initialize workers.
    workers = [
        Worker(
            i,
            make_env_function=make_env_function,
            max_episode_steps=max_episode_steps,
            replay_buffer=replay_buffer,
            member_queue=member_queues[i],
            exp_queue=exp_queue,
            step_events=step_events,
            epoch_events=epoch_events,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            num_gpus=num_gpus,
            eval_episodes=eval_episodes,
        )
        for i in range(len(population))
    ]
    pool = WorkerPool(workers, replay_buffer, member_queues, exp_queue)
    pool.start()

    for epoch in range(epochs):
        print(f"EPOCH {epoch}")
        for member in population:
            member_queues[member.id].put((member.id, member))

        for step in tqdm.tqdm(range(steps_per_epoch), dynamic_ncols=True):
            # Push collected experiences to shared replay buffer
            pool.collect_experiences()

            # Notify workers that they're good to proceed with training.
            if step % 2 == 0:
                step_events[0].set()
                step_events[1].clear()
            else:
                step_events[1].set()
                step_events[0].clear()

        # Get members that have been trained
        population = pool.get_population()

        # Save final population to disk
        for member in population:
            member.agent.save(save_dir, member.id)

        # Sort the population by increasing average return
        population = sorted(population)
        # Gget the bottom and top 20% of the population and randomly shuffle them
        worst_members = population[: (population_size // 5)]
        best_members = population[-(population_size // 5) :]
        random.shuffle(worst_members)
        random.shuffle(best_members)
        # They were shuffled so that zip() creates a random pairing
        for bad, elite in zip(worst_members, best_members):
            # Copy the good agent's network weights
            dc.utils.hard_update(bad.agent.actor, elite.agent.actor)
            dc.utils.hard_update(bad.agent.critic1, elite.agent.critic1)
            dc.utils.hard_update(bad.agent.critic2, elite.agent.critic2)
            # Copy the good agent's optimizers (this may not be necessary)
            bad.agent.online_actor_optimizer.load_state_dict(
                elite.agent.online_actor_optimizer.state_dict()
            )
            bad.agent.critic_optimizer.load_state_dict(
                elite.agent.critic_optimizer.state_dict()
            )
            # Copy the good agent's max ent constraint and optimizer
            bad.agent.log_alpha_optimizer.load_state_dict(
                elite.agent.log_alpha_optimizer.state_dict()
            )
            bad.agent.log_alpha = elite.agent.log_alpha.clone()

            # Explore the param space, clamped within a specified range
            new_a = clamp(shift_int_by_add(elite.hparams.a, a_param), a_param)
            new_c = clamp(shift_int_by_add(elite.hparams.c, c_param), c_param)
            new_g = clamp(shift_float_by_add(elite.hparams.g, g_param), g_param)
            new_k = clamp(shift_int_by_add(elite.hparams.k, k_param), k_param)
            new_h = clamp(shift_float_by_add(elite.hparams.h, h_param), h_param)
            bad.hparams = Hparams(new_a, new_c, new_k, new_h, new_g)

        # Logging
        fitness_distrib = torch.Tensor([m.fitness for m in population]).float()
        a_distrib = torch.Tensor([m.hparams.a for m in population]).float()
        c_distrib = torch.Tensor([m.hparams.c for m in population]).float()
        k_distrib = torch.Tensor([m.hparams.k for m in population]).float()
        h_distrib = torch.Tensor([m.hparams.h for m in population]).float()
        g_distrib = torch.Tensor([m.hparams.g for m in population]).float()

        writer.add_histogram("Fitness", fitness_distrib, epoch)
        writer.add_histogram("A Param", a_distrib, epoch)
        writer.add_histogram("C Param", c_distrib, epoch)
        writer.add_histogram("K Param", k_distrib, epoch)
        writer.add_histogram("H Param", h_distrib, epoch)
        writer.add_histogram("G Param", g_distrib, epoch)

        best_return = population[-1].fitness
        best_a = population[-1].hparams.a
        best_c = population[-1].hparams.c
        best_k = population[-1].hparams.k
        best_h = population[-1].hparams.h
        best_g = population[-1].hparams.g

        writer.add_scalar("BestReturn", best_return, epoch)
        writer.add_scalar("BestA", best_a, epoch)
        writer.add_scalar("BestC", best_c, epoch)
        writer.add_scalar("BestK", best_k, epoch)
        writer.add_scalar("BestH", best_h, epoch)
        writer.add_scalar("BestG", best_g, epoch)
        with open(os.path.join(save_dir, "population_fitness.csv"), "a") as f:
            f.write(",".join([f"{m.fitness.item():.1f}" for m in population]) + "\n")

        # Notify workers they're good to proceed with next epoch.
        if epoch % 2 == 0:
            epoch_events[0].set()
            epoch_events[1].clear()
        else:
            epoch_events[1].set()
            epoch_events[0].clear()
    return population


class PersistenceReplayBufferStorage:
    def __init__(
        self, size, obs_shape, act_shape, action_repeat, obs_dtype=torch.float32
    ):
        self.s_dtype = obs_dtype

        # buffer arrays
        self.s_stack = torch.zeros((size,) + obs_shape, dtype=self.s_dtype)
        self.action_stack = torch.zeros((size,) + act_shape, dtype=torch.float32)
        self.reward_stack = torch.zeros((size, action_repeat), dtype=torch.float32)
        self.s1_stack = torch.zeros((size,) + obs_shape, dtype=self.s_dtype)
        self.done_stack = torch.zeros((size, 1), dtype=torch.int)

        self.obs_shape = obs_shape
        self.size = size
        self._next_idx = 0
        self._max_filled = 0

        self._shared = False

    def __len__(self):
        return self.max_filled

    @property
    def next_idx(self):
        if self._shared:
            return self._next_idx.value
        else:
            return self._next_idx

    @next_idx.setter
    def next_idx(self, v):
        if self._shared:
            self._next_idx.value = v
        else:
            self._next_idx = v

    @property
    def max_filled(self):
        if self._shared:
            return self._max_filled.value
        else:
            return self._max_filled

    @max_filled.setter
    def max_filled(self, v):
        if self._shared:
            self._max_filled.value = v
        else:
            self._max_filled = v

    def add(self, s, a, r, s_1, d):
        # this buffer supports batched experience
        if len(s.shape) > len(self.obs_shape):
            # there must be a batch dimension
            num_samples = len(s)
        else:
            num_samples = 1
            d = [d]

        if not isinstance(s, torch.Tensor):
            # convert states to numpy (checking for LazyFrames)
            if not isinstance(s, np.ndarray):
                s = np.asarray(s)
            if not isinstance(s_1, np.ndarray):
                s_1 = np.asarray(s_1)

            # convert to torch tensors
            s = torch.from_numpy(s)
            a = torch.from_numpy(a).float()
            r = torch.Tensor(r).float().unsqueeze(0)

            steps_short = self.reward_stack.shape[1] - r.shape[1]
            if steps_short > 0:
                r = torch.cat((r, torch.zeros(1, steps_short)), dim=1)
            s_1 = torch.from_numpy(s_1)
            d = torch.Tensor(d).int()

            # make sure tensors are floats not doubles
            if self.s_dtype is torch.float32:
                s = s.float()
                s_1 = s_1.float()

        else:
            # move to cpu
            s = s.cpu()
            a = a.cpu()
            r = r.cpu()
            s_1 = s_1.cpu()
            d = d.int().cpu()

        # Store at end of buffer. Wrap around if past end.
        R = np.arange(self.next_idx, self.next_idx + num_samples) % self.size
        self.s_stack[R] = s
        self.action_stack[R] = a
        self.reward_stack[R] = r
        self.s1_stack[R] = s_1
        self.done_stack[R] = d
        # Advance index.
        self.max_filled = min(
            max(self.next_idx + num_samples, self.max_filled), self.size
        )
        self.next_idx = (self.next_idx + num_samples) % self.size
        return R

    def __getitem__(self, indices):
        try:
            iter(indices)
        except ValueError:
            raise IndexError(
                "ReplayBufferStorage getitem called with indices object that is not iterable"
            )

        # converting states and actions to float here instead of inside the learning loop
        # of each agent seems fine for now.
        state = self.s_stack[indices].float()
        action = self.action_stack[indices].float()
        reward = self.reward_stack[indices]
        next_state = self.s1_stack[indices].float()
        done = self.done_stack[indices]
        return (state, action, reward, next_state, done)

    def __setitem__(self, indices, experience):
        s, a, r, s1, d = experience
        self.s_stack[indices] = s.float()
        self.action_stack[indices] = a.float()
        self.reward_stack[indices] = r
        self.s1_stack[indices] = s1.float()
        self.done_stack[indices] = d

    def get_all_transitions(self):
        return (
            self.s_stack[: self.max_filled],
            self.action_stack[: self.max_filled],
            self.reward_stack[: self.max_filled],
            self.s1_stack[: self.max_filled],
            self.done_stack[: self.max_filled],
        )

    def share_memory_(self):
        if self._shared:
            return

        self._shared = True
        self.s_stack.share_memory_()
        self.action_stack.share_memory_()
        self.reward_stack.share_memory_()
        self.s1_stack.share_memory_()
        self.done_stack.share_memory_()
        self._max_filled = mp.Value("i", self._max_filled)
        self._next_idx = mp.Value("i", self._next_idx)


class PersistenceReplayBuffer:
    def __init__(
        self,
        size,
        state_shape=None,
        action_shape=None,
        action_repeat=1,
        state_dtype=float,
    ):
        self._maxsize = size
        self.state_shape = state_shape
        self.state_dtype = self._convert_dtype(state_dtype)
        self.action_shape = action_shape
        self._storage = None
        self.action_repeat = action_repeat
        assert self.state_shape, "Must provide shape of state space to ReplayBuffer"
        assert self.action_shape, "Must provide shape of action space to ReplayBuffer"

    def _convert_dtype(self, dtype):
        if dtype in [int, np.uint8, torch.uint8]:
            return torch.uint8
        elif dtype in [float, np.float32, np.float64, torch.float32, torch.float64]:
            return torch.float32
        elif dtype in ["int32", np.int32]:
            return torch.int32
        else:
            raise ValueError(f"Uncreocgnized replay buffer dtype: {dtype}")

    def __len__(self):
        return len(self._storage) if self._storage is not None else 0

    def push(self, state, action, reward, next_state, done):
        if self._storage is None:
            self._storage = PersistenceReplayBufferStorage(
                self._maxsize,
                obs_shape=self.state_shape,
                act_shape=self.action_shape,
                action_repeat=self.action_repeat,
                obs_dtype=self.state_dtype,
            )
        return self._storage.add(state, action, reward, next_state, done)

    def sample(self, batch_size, get_idxs=False):
        random_idxs = torch.randint(len(self._storage), (batch_size,))
        if get_idxs:
            return self._storage[random_idxs], random_idxs.cpu().numpy()
        else:
            return self._storage[random_idxs]

    def get_all_transitions(self):
        return self._storage.get_all_transitions()

    def load_experience(self, s, a, r, s1, d):
        assert (
            s.shape[0] <= self._maxsize
        ), "Experience dataset is larger than the buffer."
        if len(r.shape) < 2:
            r = np.expand_dims(r, 1)
        if len(d.shape) < 2:
            d = np.expand_dims(d, 1)
        self.push(s, a, r, s1, d)

    def share_memory_(self):
        self._storage.share_memory_()


"""
Parallel implementation requires us to pass a the function to create
new environments between processes. These functions need to be picklable.
A quick solution is to make them global:
"""


import gym

try:
    import dmc2gym
except:
    "MuJoCo not found or dmc2gym not installed. Skipping..."
    pass


try:
    import or_gym
    from industrial_benchmark_python.IBGym import IBGym
except:
    "or-gym or industrial_benchmark_python packages not installed. Skipping..."
    pass


def fish_swim():
    return PersistenceAwareWrapper(dmc2gym.make("fish", "swim"))


def walker_run():
    return PersistenceAwareWrapper(dmc2gym.make("walker", "run"))


def swimmer_swimmer6():
    return PersistenceAwareWrapper(dmc2gym.make("swimmer", "swimmer6"))


def humanoid_stand():
    return PersistenceAwareWrapper(dmc2gym.make("humanoid", "stand"))


def reacher_hard():
    return PersistenceAwareWrapper(dmc2gym.make("reacher", "hard"))


def cheetah_run():
    return PersistenceAwareWrapper(dmc2gym.make("cheetah", "run"))


def bipedal_hardcore():
    # AAC gets good results in bipedal, although they weren't used in the paper
    # because the task is too similar to the DMC benchmarks.
    return PersistenceAwareWrapper(gym.make("BipedalWalkerHardcore-v3"))


def _ib(setpoint):
    return PersistenceAwareWrapper(
        IBGym(
            setpoint=setpoint,
            reward_type="classic",
            action_type="continuous",
            observation_type="include_past",
        )
    )


def industrial_benchmark_70():
    return _ib(70)


def industrial_benchmark_100():
    return _ib(100)


def inventory():
    return PersistenceAwareWrapper(
        dc.envs.NormalizeContinuousActionSpace(gym.make("or_gym:InvManagement-v1"))
    )


def newsvendor():
    # A quick random agent baseline shows the Newsvendor rewards are far too large.
    # We scale by 1e-4.
    return PersistenceAwareWrapper(
        dc.envs.ScaleReward(
            dc.envs.NormalizeContinuousActionSpace(gym.make("or_gym:Newsvendor-v0")),
            1e-4,
        )
    )


def inventory():
    return PersistenceAwareWrapper(
        dc.envs.NormalizeContinuousActionSpace(gym.make("or_gym:InvManagement-v1"))
    )


def newsvendor():
    return PersistenceAwareWrapper(
        dc.envs.ScaleReward(
            dc.envs.NormalizeContinuousActionSpace(gym.make("or_gym:Newsvendor-v0")),
            1e-4,
        )
    )


def add_args(parser):
    parser.add_argument("--name", type=str, required=True, help="Name of the run.")
    parser.add_argument(
        "--epochs", type=int, default=250, help="Number of evolutionary epochs."
    )
    parser.add_argument(
        "--num_seeds", type=int, default=1, help="Number of trials with random seeds."
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=1000,
        help="Number of training steps per epoch.",
    )
    parser.add_argument(
        "--population_size", type=int, default=20, help="Population size"
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=1000,
        help="Maximum steps of an episode.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size of experiences from replay buffer used for training",
    )
    parser.add_argument(
        "--a_max", type=int, default=10, help="Maximum value for hyperparam `a`."
    )
    parser.add_argument(
        "--a_delta", type=int, default=2, help="Delta value for hyperparam `a`."
    )
    parser.add_argument(
        "--c_max", type=int, default=40, help="Maximum value for hyperparam `c`."
    )
    parser.add_argument(
        "--c_delta", type=int, default=5, help="Delta value for hyperparam `c`."
    )
    parser.add_argument(
        "--k_max", type=int, default=15, help="Maximum value for hyperparam `k`."
    )
    parser.add_argument(
        "--eval_episodes", type=int, default=10, help="Episodes per evaluation"
    )
