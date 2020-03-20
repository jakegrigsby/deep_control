import argparse

from deep_control.run import load_env
import deep_control

def make_test_naf_args(her=False):
    args = argparse.Namespace()
    args.num_episodes = 1
    args.max_episode_steps = 100
    args.batch_size = 2
    args.tau = .999
    args.lr = 1e-4
    args.gamma = .99
    args.sigma_start = .2
    args.sigma_final = .1
    args.sigma_anneal = 10
    args.theta = .15
    args.sigma = .2
    args.buffer_size = 1000
    args.eval_interval = 100
    args.eval_episodes = 1
    args.warmup_steps = 10
    args.l2 = .0001
    args.render = False
    args.clip = 1.
    args.name = 'test_naf_run'
    args.her = her
    args.opt_steps = 2
    print(vars(args))
    return args

def test_pendulum_naf():
    agent, env = load_env('Pendulum-v0', 'naf')
    deep_control.naf(agent, env, make_test_naf_args())


def test_mountaincar_naf():
    agent, env = load_env('MountainCarContinuous-v0', 'naf')
    deep_control.naf(agent, env, make_test_naf_args())


def test_ant_naf():
    agent, env = load_env('Ant-v3', 'naf')
    deep_control.naf(agent, env, make_test_naf_args())


def test_walker_naf():
    agent, env = load_env('Walker2d-v3', 'naf')
    deep_control.naf(agent, env, make_test_naf_args())


def test_swimmer_naf():
    agent, env = load_env('Swimmer-v3', 'naf')
    deep_control.naf(agent, env, make_test_naf_args())


def test_reacher_naf():
    agent, env = load_env('Reacher-v2', 'naf')
    deep_control.naf(agent, env, make_test_naf_args())


def test_humanoid_naf():
    agent, env = load_env('Humanoid-v2', 'naf')
    deep_control.naf(agent, env, make_test_naf_args())


def test_humanoid_standup_naf():
    agent, env = load_env('HumanoidStandup-v2', 'naf')
    deep_control.naf(agent, env, make_test_naf_args())


def test_half_cheetah_naf():
    agent, env = load_env('HalfCheetah-v3', 'naf')
    deep_control.naf(agent, env, make_test_naf_args())


def test_fetch_push_naf():
    agent, env = load_env('FetchPush-v1', 'naf')
    deep_control.naf(agent, env, make_test_naf_args(her=True))


def test_fetch_reach_naf():
    agent, env = load_env('FetchReach-v1', 'naf')
    deep_control.naf(agent, env, make_test_naf_args(her=True))


def test_fetch_slide_naf():
    agent, env = load_env('FetchSlide-v1', 'naf')
    deep_control.naf(agent, env, make_test_naf_args(her=True))


def test_fetch_pick_place_naf():
    agent, env = load_env('FetchPickAndPlace-v1', 'naf')
    deep_control.naf(agent, env, make_test_naf_args(her=True))


def test_hand_reach_naf():
    agent, env = load_env('HandReach-v0', 'naf')
    deep_control.naf(agent, env, make_test_naf_args(her=True))

def test_manipulate_block_naf():
    agent, env = load_env('HandManipulateBlockFull-v0', 'naf')
    deep_control.naf(agent, env, make_test_naf_args(her=True))


def test_manipulate_egg_naf():
    agent, env = load_env('HandManipulateEggFull-v0', 'naf')
    deep_control.naf(agent, env, make_test_naf_args(her=True))


def test_manipulate_pen_naf():
    agent, env = load_env('HandManipulatePenFull-v0', 'naf')
    deep_control.naf(agent, env, make_test_naf_args(her=True))




