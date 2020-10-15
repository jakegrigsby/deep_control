import argparse
import random
import copy
import sys

import deep_control as dc

from deep_control.augmentations import *

from dmc_remastered.benchmarks import *


def train_dmcr_sac(args):
    benchmark_kwargs = {
        "domain": args.domain,
        "task": args.task,
        "num_levels": args.num_levels,
        "frame_stack": args.frame_stack,
        "height": 84,
        "width": 84,
        "frame_skip": args.frame_skip,
        "channels_last": False,
    }

    if args.benchmark == "visual_generalization":
        benchmark = visual_generalization
        args.train_eval_episdodes = 100
        args.test_eval_episodes = 100
    elif args.benchmark == "visual_sim2real":
        benchmark = visual_sim2real
        args.train_eval_episodes = 100
        args.test_eval_episodes = 10
    elif args.benchmark == "classic":
        benchmark = classic
        del benchmark_kwargs["num_levels"]
        benchmark_kwargs["visual_seed"] = args.visual_seed
        args.train_eval_episodes = 0
        args.test_eval_episodes = 10
    elif args.benchmark == "control":
        seed = random.randint(0, 10000)
        train_env = dc.envs.load_dmc(
            domain_name=args.domain,
            task_name=args.task,
            from_pixels=True,
            rgb=True,
            frame_stack=args.frame_stack,
            frame_skip=args.frame_skip,
            seed=seed,
        )
        test_env = dc.envs.load_dmc(
            domain_name=args.domain,
            task_name=args.task,
            from_pixels=True,
            rgb=True,
            frame_stack=args.frame_stack,
            frame_skip=args.frame_skip,
            seed=seed,
        )
        args.train_eval_episodes = 0
        args.test_eval_episodes = 10

    if args.benchmark != "control":
        train_env, test_env = benchmark(**benchmark_kwargs)

    obs_shape = train_env.observation_space.shape
    action_shape = train_env.action_space.shape
    max_action = train_env.action_space.high[0]

    augmentation_lst = [aug(args.batch_size) for aug in eval(args.augmentations)]
    augmenter = AugmentationSequence(augmentation_lst)

    agent = dc.sac_aug.PixelSACAgent(
        obs_shape, action_shape[0], args.log_std_low, args.log_std_high
    )

    # select a replay buffer
    if args.prioritized_replay:
        buffer_t = dc.replay.PrioritizedReplayBuffer
    else:
        buffer_t = dc.replay.ReplayBuffer
    buffer = buffer_t(
        args.buffer_size,
        state_dtype=int,
        state_shape=train_env.observation_space.shape,
        action_shape=train_env.action_space.shape,
    )

    dc.sac_aug.sac_aug(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        buffer=buffer,
        augmenter=augmenter,
        **vars(args),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="walker")
    parser.add_argument("--task", type=str, default="walk")
    parser.add_argument("--benchmark", type=str, default="classic")
    parser.add_argument("--visual_seed", type=int, default=0)
    parser.add_argument("--num_levels", type=int, default=1_000_000)
    parser.add_argument("--frame_stack", type=int, default=3)
    parser.add_argument("--frame_skip", type=int, default=2)
    dc.sac_aug.add_args(parser)  # sac+aug related args
    args = parser.parse_args()

    # auto-adjust the max episode steps to compensate for the frame skipping.
    # dmc (and dmcr) automatically reset after 1k steps, but this allows for
    # infinite bootstrapping (used by CURL and SAC-AE)
    args.max_episode_steps = (1000 + args.frame_skip - 1) // args.frame_skip
    train_dmcr_sac(args)
