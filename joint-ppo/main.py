import numpy as np
import tensorflow as tf
import argparse
import pandas as pd
from baselines import logger
from functools import partial
from multiprocessing import JoinableQueue, Queue, Process

from sonic_util import make_local_env, JointEnv, SonicDiscretizer
import baselines.ppo2 as ppo2

def train(args):
    logger.configure(args.params_folder)
    tasks = JoinableQueue()
    results = Queue()

    train_data = pd.read_csv('../sonic-train.csv')
    env_fns = []
    levels = []

    for index, level in train_data.iterrows():
        env_fn = partial(make_local_env, level.game, level.state, True, True)
        env_fns.append(env_fn)
        levels.append((level.game, level.state))

    joint_env = JointEnv(env_fns)
    print("joint initialized")


def main():
    parser = argparse.ArgumentParser(description = 'Joint PPO')
    parser.add_argument('params_folder', help='params directory')
    parser.add_argument('--workers', help='number of worker processes', default=4)
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()

