import argparse

import deep_control as dc


def add_dmc_args(parser):
    parser.add_argument("--domain_name", type=str, default="fish")
    parser.add_argument("--task_name", type=str, default="swim")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--from_pixels", action="store_true")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dmc_args(parser)
