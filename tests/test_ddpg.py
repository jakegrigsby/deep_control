import argparse

from deep_control.run import load_env
import deep_control
import pybullet

def make_test_ddpg_args():
    args = argparse.Namespace()
    args.num_steps = 25
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
    args.sigma = .2
    args.buffer_size = 1000
    args.eval_interval = 100
    args.eval_episodes = 1
    args.warmup_steps = 10
    args.render = False
    args.actor_clip = 1.
    args.critic_clip = None
    args.name = 'test_ddpg_run'
    args.actor_l2 = 0.
    args.critic_l2 = .0001
    args.save_interval = 20
    return args

####################
## Gym Classic CC ##
####################

def test_pendulum_ddpg():
    agent, env = load_env('Pendulum-v0', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())


def test_mountaincar_ddpg():
    agent, env = load_env('MountainCarContinuous-v0', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())

################
## Gym MuJoCo ##
################

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

##################
## Pybullet Gym ##
##################


def test_minitaur_bullet__ddpg():
    agent, env = load_env('MinitaurBulletEnv-v0', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())


def test_humanoid_deep_mimic_backflip_bullet_ddpg():
    agent, env = load_env('HumanoidDeepMimicBackflipBulletEnv-v1', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())


def test_humanoid_deep_mimic_walk_bullet_ddpg():
    agent, env = load_env('HumanoidDeepMimicWalkBulletEnv-v1', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())

def test_racecar_bullet_ddpg():
    agent, env = load_env('RacecarBulletEnv-v0', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())

def test_ant_bullet_ddpg():
    agent, env = load_env('AntBulletEnv-v0', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())

def test_half_cheetah_bullet_ddpg():
    agent, env = load_env('HalfCheetahBulletEnv-v0', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())

def test_humanoid_bullet_ddpg():
    agent, env = load_env('HumanoidBulletEnv-v0', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())

def test_hopper_bullet_ddpg():
    agent, env = load_env('HopperBulletEnv-v0', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())

def test_walker2d_bullet_ddpg():
    agent, env = load_env('Walker2DBulletEnv-v0', 'ddpg')
    deep_control.ddpg(agent, env, make_test_ddpg_args())











