import argparse

from deep_control.run import load_env
import deep_control
import pybullet

def make_test_td3_args():
    args = argparse.Namespace()
    args.num_episodes = 1
    args.max_episode_steps = 100
    args.batch_size = 2
    args.tau = .001
    args.actor_lr = 1e-4
    args.critic_lr = 1e-3
    args.gamma = .99
    args.sigma_start = .2
    args.sigma_final = 1e-3
    args.sigma_anneal = 10
    args.theta = .15
    args.buffer_size = 1000
    args.eval_interval = 1
    args.eval_episodes = 1
    args.warmup_steps = 10
    args.render = False
    args.actor_clip = 1.
    args.critic_clip = None
    args.name = 'test_td3_run'
    args.opt_steps = 2
    args.actor_l2 = 0.
    args.critic_l2 = .0001
    args.delay = 2
    args.target_noise_scale = .2
    args.c = .5
    return args

####################
## Gym Classic CC ##
####################

def test_pendulum_td3():
    agent, env = load_env('Pendulum-v0', 'td3')
    deep_control.td3(agent, env, make_test_td3_args())


def test_mountaincar_td3():
    agent, env = load_env('MountainCarContinuous-v0', 'td3')
    deep_control.td3(agent, env, make_test_td3_args())

################
## Gym MuJoCo ##
################

def test_ant_td3():
    agent, env = load_env('Ant-v3', 'td3')
    deep_control.td3(agent, env, make_test_td3_args())


def test_walker_td3():
    agent, env = load_env('Walker2d-v3', 'td3')
    deep_control.td3(agent, env, make_test_td3_args())


def test_swimmer_td3():
    agent, env = load_env('Swimmer-v3', 'td3')
    deep_control.td3(agent, env, make_test_td3_args())


def test_reacher_td3():
    agent, env = load_env('Reacher-v2', 'td3')
    deep_control.td3(agent, env, make_test_td3_args())


def test_humanoid_td3():
    agent, env = load_env('Humanoid-v2', 'td3')
    deep_control.td3(agent, env, make_test_td3_args())


def test_humanoid_standup_td3():
    agent, env = load_env('HumanoidStandup-v2', 'td3')
    deep_control.td3(agent, env, make_test_td3_args())


def test_half_cheetah_td3():
    agent, env = load_env('HalfCheetah-v3', 'td3')
    deep_control.td3(agent, env, make_test_td3_args())

##################
## Pybullet Gym ##
##################


def test_minitaur_bullet__td3():
    agent, env = load_env('MinitaurBulletEnv-v0', 'td3')
    deep_control.td3(agent, env, make_test_td3_args())


def test_humanoid_deep_mimic_backflip_bullet_td3():
    agent, env = load_env('HumanoidDeepMimicBackflipBulletEnv-v1', 'td3')
    deep_control.td3(agent, env, make_test_td3_args())


def test_humanoid_deep_mimic_walk_bullet_td3():
    agent, env = load_env('HumanoidDeepMimicWalkBulletEnv-v1', 'td3')
    deep_control.td3(agent, env, make_test_td3_args())

def test_racecar_bullet_td3():
    agent, env = load_env('RacecarBulletEnv-v0', 'td3')
    deep_control.td3(agent, env, make_test_td3_args())

def test_ant_bullet_td3():
    agent, env = load_env('AntBulletEnv-v0', 'td3')
    deep_control.td3(agent, env, make_test_td3_args())

def test_half_cheetah_bullet_td3():
    agent, env = load_env('HalfCheetahBulletEnv-v0', 'td3')
    deep_control.td3(agent, env, make_test_td3_args())

def test_humanoid_bullet_td3():
    agent, env = load_env('HumanoidBulletEnv-v0', 'td3')
    deep_control.td3(agent, env, make_test_td3_args())

def test_hopper_bullet_td3():
    agent, env = load_env('HopperBulletEnv-v0', 'td3')
    deep_control.td3(agent, env, make_test_td3_args())

def test_walker2d_bullet_td3():
    agent, env = load_env('Walker2DBulletEnv-v0', 'td3')
    deep_control.td3(agent, env, make_test_td3_args())











