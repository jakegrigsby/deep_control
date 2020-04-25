import argparse

from deep_control.run import load_env
import deep_control
import pybullet

def make_test_naf_args():
    args = argparse.Namespace()
    args.num_episodes = 1
    args.max_episode_steps = 100
    args.batch_size = 2
    args.tau = .001
    args.lr = 1e-4
    args.gamma = .99
    args.sigma_start = .2
    args.sigma_final = .1
    args.sigma_anneal = 10
    args.theta = .15
    args.sigma = .2
    args.buffer_size = 1000
    args.eval_interval = 1
    args.eval_episodes = 1
    args.warmup_steps = 10
    args.l2 = .0001
    args.render = False
    args.clip = 1.
    args.name = 'test_naf_run'
    args.opt_steps = 2
    print(vars(args))
    return args


####################
## Gym Classic CC ##
####################

def test_pendulum_naf():
    agent, env = load_env('Pendulum-v0', 'naf')
    deep_control.naf(agent, env, make_test_naf_args())


def test_mountaincar_naf():
    agent, env = load_env('MountainCarContinuous-v0', 'naf')
    deep_control.naf(agent, env, make_test_naf_args())

################
## Gym MuJoCo ##
################

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

##################
## Pybullet Gym ##
##################


def test_minitaur_bullet__naf():
    agent, env = load_env('MinitaurBulletEnv-v0', 'naf')
    deep_control.naf(agent, env, make_test_naf_args())


def test_humanoid_deep_mimic_backflip_bullet_naf():
    agent, env = load_env('HumanoidDeepMimicBackflipBulletEnv-v1', 'naf')
    deep_control.naf(agent, env, make_test_naf_args())


def test_humanoid_deep_mimic_walk_bullet_naf():
    agent, env = load_env('HumanoidDeepMimicWalkBulletEnv-v1', 'naf')
    deep_control.naf(agent, env, make_test_naf_args())

def test_racecar_bullet_naf():
    agent, env = load_env('RacecarBulletEnv-v0', 'naf')
    deep_control.naf(agent, env, make_test_naf_args())

def test_ant_bullet_naf():
    agent, env = load_env('AntBulletEnv-v0', 'naf')
    deep_control.naf(agent, env, make_test_naf_args())

def test_half_cheetah_bullet_naf():
    agent, env = load_env('HalfCheetahBulletEnv-v0', 'naf')
    deep_control.naf(agent, env, make_test_naf_args())

def test_humanoid_bullet_naf():
    agent, env = load_env('HumanoidBulletEnv-v0', 'naf')
    deep_control.naf(agent, env, make_test_naf_args())

def test_hopper_bullet_naf():
    agent, env = load_env('HopperBulletEnv-v0', 'naf')
    deep_control.naf(agent, env, make_test_naf_args())

def test_walker2d_bullet_naf():
    agent, env = load_env('Walker2DBulletEnv-v0', 'naf')
    deep_control.naf(agent, env, make_test_naf_args())











