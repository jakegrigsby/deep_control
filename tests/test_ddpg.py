import argparse

from deep_control.run import load_env
import deep_control

def make_test_ddpg_args(her=False):
    args = argparse.Namespace()
    args.num_episodes = 1
    args.max_episode_steps = 100
    args.batch_size = 2
    args.tau = .999
    args.actor_lr = 1e-4
    args.critic_lr = 1e-3
    args.gamma = .99
    args.eps_start = 1.
    args.eps_final = 1e-3
    args.eps_anneal = 10
    args.theta = .15
    args.sigma = .2
    args.buffer_size = 1000
    args.eval_interval = 100
    args.eval_episodes = 1
    args.warmup_steps = 10
    args.render = False
    args.clip = 1.
    args.name = 'test_ddpg_run'
    args.her = her
    args.opt_steps = 2
    args.actor_l2 = 0.
    args.critic_l2 = .0001
    return args

def test_pendulum_ddpg():
    agent, env = load_env('Pendulum-v0', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())


def test_mountaincar_ddpg():
    agent, env = load_env('MountainCarContinuous-v0', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())


def test_ant_ddpg():
    agent, env = load_env('Ant-v3', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())


def test_walker_ddpg():
    agent, env = load_env('Walker2d-v3', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())


def test_swimmer_ddpg():
    agent, env = load_env('Swimmer-v3', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())


def test_reacher_ddpg():
    agent, env = load_env('Reacher-v2', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())


def test_humanoid_ddpg():
    agent, env = load_env('Humanoid-v2', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())


def test_humanoid_standup_ddpg():
    agent, env = load_env('HumanoidStandup-v2', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())


def test_half_cheetah_ddpg():
    agent, env = load_env('HalfCheetah-v3', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())


def test_fetch_push_ddpg():
    agent, env = load_env('FetchPush-v1', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())


def test_fetch_reach_ddpg():
    agent, env = load_env('FetchReach-v1', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())


def test_fetch_slide_ddpg():
    agent, env = load_env('FetchSlide-v1', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())


def test_fetch_pick_place_ddpg():
    agent, env = load_env('FetchPickAndPlace-v1', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())


def test_hand_reach_ddpg():
    agent, env = load_env('HandReach-v0', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())

def test_manipulate_block_ddpg():
    agent, env = load_env('HandManipulateBlockFull-v0', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())


def test_manipulate_egg_ddpg():
    agent, env = load_env('HandManipulateEggFull-v0', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())


def test_manipulate_pen_ddpg():
    agent, env = load_env('HandManipulatePenFull-v0', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())




