import os
import numpy as np
import tensorflow as tf
import argparse
import pandas as pd
import os.path as osp
import psutil
import time
from baselines import logger
from functools import partial
from multiprocessing import JoinableQueue, Queue, Process

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from sonic_util import make_local_env, JointEnv, SonicDiscretizer
import baselines.ppo2.policies as policies
import ppo2

#Model
ent_coef=0.001 # entropy coef
steps_per_ep=4500 # number of steps in an ep (i.e., run until the time limit
horizon=steps_per_ep # history of timesteps (?)

#Common
gamma=0.99
lam=0.95
cliprange=0.2
vf_coef=0.5
max_grad_norm=0.5 # ratio of sum of norms, for clipping gradients
nbatch_act=1 # number of envs
nminibatches=4
lr=2e-4 # learning rate
noptepochs=4
nbatch_train = horizon // nminibatches # number of training batches
logs_path='/tmp/tensorflow_logs/joint_ppo'

def train(args):
    logger.configure(args.params_folder)
    checkdir = osp.join(logger.get_dir(), 'checkpoints')
    os.makedirs(checkdir, exist_ok=True)
    last_savepath = checkdir + 'init'

    tasks = JoinableQueue()
    results = Queue()

    train_data = pd.read_csv('../sonic-train.csv')
    env_fns = []
    levels = []

    for index, level in train_data.iterrows():
        env_fn = partial(make_local_env, level.game, level.state, True, True)
        env_fns.append(env_fn)
        levels.append((level.game, level.state))

    joint_env = JointEnv(env_fns, levels)
    print("joint initialized")

    def joint_env_fn():
        return joint_env

    grads = []
    # obs/action space is the same for all environments

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    pid = os.getpid()
    py = psutil.Process(pid)

    # init params
    with tf.Session(config=config):
        model = ppo2.Model(policy=policies.CnnPolicy,
                           ob_space=joint_env.env.observation_space,
                           ac_space=joint_env.env.action_space,
                           nbatch_act=nbatch_act,
                           nsteps=steps_per_ep,
                           nbatch_train=nbatch_train,
                           ent_coef=ent_coef,
                           vf_coef=vf_coef,
                           max_grad_norm=max_grad_norm)

        print('Saving init to', last_savepath)
        model.save(last_savepath)

    tf.reset_default_graph()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    interval = 0
    while True:
        print("epoch: %d" % interval)
        with tf.Session(config=config):
            # first, load previous model
            print("Loading last saved_model")
            model = ppo2.Model(policy=policies.CnnPolicy,
                               ob_space=joint_env.env.observation_space,
                               ac_space=joint_env.env.action_space,
                               nbatch_act=nbatch_act,
                               nsteps=steps_per_ep,
                               nbatch_train=nbatch_train,
                               ent_coef=ent_coef,
                               vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm)

            model.load(last_savepath)

            print("Now computing gradients for all levels")

            runner = ppo2.Runner(
                    env=DummyVecEnv([joint_env_fn]),
                    num_envs=joint_env.num_envs,
                    model=model,
                    nsteps=steps_per_ep,
                    gamma=gamma,
                    lam=lam,
                    lr=lr,
                    cliprange=cliprange,
                    noptepochs=noptepochs,
                    nbatch_train=nbatch_train)

            workers = runner.get_grads()

            # have to save every iteration, since we close the session

            num_workers = len(workers)
            print(num_workers)
            assert len(workers[0]) == 12
            print("Done running workers on all envs, now merging gradients")

            avg_grads = []
            # for each gradient,
            for i in range(len(workers[0])):
                # pool together the grads i from each worker j
                total_grad = workers[0][i][0]
                for j in range(1, num_workers):
                    total_grad += workers[j][i][0]
                avg_grads.append(total_grad / num_workers)
            print("Merge completed")

            model.joint_train(avg_grads)
            print("Joint update completed")

            last_savepath = osp.join(checkdir, '%.5i' % interval)
            print('Saving to', last_savepath)
            model.save(last_savepath)

            interval += 1

        memUse = py.memory_info()[0]/2.**30
        print('memory use:', memUse)

        tf.reset_default_graph()

def main():
    parser = argparse.ArgumentParser(description = 'Joint PPO')
    parser.add_argument('params_folder', help='params directory')
    parser.add_argument('--workers', help='number of worker processes', default=4)
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()

