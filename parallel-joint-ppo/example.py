import os
import os.path as osp
import numpy as np
import tensorflow as tf
import argparse
import pandas as pd
import os.path as osp
import psutil
import time
import glob
import re
import gc

import ppo2
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from sonic_util import make_local_env, JointEnv
import baselines.ppo2.policies as policies
from baselines import logger
from functools import partial
from multiprocessing import JoinableQueue, Queue, Process

KILL_SIGNAL = 2
TRAIN_SIGNAL = 1
LOAD_SIGNAL = 0
NUM_ENVS = 47
DUMMY_ENV = -1
NUM_WORKERS = 3
NUM_LOOPS = 100

#Model
ent_coef=0.001 # entropy coef
steps_per_ep=4500 # number of steps in an ep (i.e., run until the time limit
horizon=steps_per_ep # history of timesteps (?)M

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

#dummy env, just to initialize observation/action space
env = make_local_env(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', stack=True, scale_rew=True)
env.close()

class Worker(Process):
    def __init__(self, tasks, results, id):
        super().__init__()
        self.tasks = tasks
        self.results = results
        self.id = id
        self.completed_count = 0

        train_data = pd.read_csv('../sonic-train.csv')
        levels = []
        for index, level in train_data.iterrows():
            levels.append((level.game, level.state))
        self.levels = levels

    def run(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
                self.init_graph()
                while True:
                    cmd, data = self.tasks.get()
                    if cmd == KILL_SIGNAL:
                        self.results.put("worker %d has completed %d tasks" % (self.id, self.completed_count))
                        self.tasks.task_done()
                        return
                    elif cmd == LOAD_SIGNAL:
                        print(" worker %d loading params" % (self.id))
                        self.load_params(data)
                        self.tasks.task_done()
                    elif cmd == TRAIN_SIGNAL:
                        envIdx = data
                        print(" worker %d running on %s" % (self.id, self.levels[envIdx]))
                        exp = self.get_exp(envIdx)
                        print(" worker %d completed task" % self.id)
                        self.results.put(exp)
                        self.completed_count += 1
                        self.tasks.task_done()

    def init_graph(self):
        self.model = ppo2.Model(policy=policies.CnnPolicy,
                               ob_space=env.observation_space,
                               ac_space=env.action_space,
                               nbatch_act=nbatch_act,
                               nsteps=steps_per_ep,
                               nbatch_train=nbatch_train,
                               ent_coef=ent_coef,
                               vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm)

    def load_params(self, params):
        self.model.load_ram(params)

    def get_exp(self, envIdx):
        level = self.levels[envIdx]
        env = make_local_env(level[0], level[1], True, True)
        def env_fn():
            return env
        # self.model.load_ram(params)
        runner = ppo2.Runner(
                        env=DummyVecEnv([env_fn]),
                        num_envs=1,
                        model=self.model,
                        nsteps=steps_per_ep,
                        gamma=gamma,
                        lam=lam,
                        lr=lr,
                        cliprange=cliprange,
                        noptepochs=noptepochs,
                        nbatch_train=nbatch_train)
        exp = runner.run()
        env.close()
        # tf.reset_default_graph()
        # del runner
        # gc.collect()
        pid = os.getpid()
        py = psutil.Process(pid)
        memUse = py.memory_info()[0]/2.**30
        print('memory use: %.6f GB from worker %d after model' %(memUse, self.id))
        return exp

class Master():
    def __init__(self, args):
        self.tasks = JoinableQueue()
        self.results = Queue()
        self.workers = []
        last_savepath = None

        if args.params_folder:
            logger.configure(args.params_folder + '/')
        else:
            logger.configure('params/')
        checkdir = osp.join(logger.get_dir(), 'checkpoints')
        os.makedirs(checkdir, exist_ok=True)
        if args.last_save_params:
            list_of_params = glob.glob(checkdir + '/*')
            if len(list_of_params) > 0:
                last_savepath=max(list_of_params, key=osp.getctime)
                print('Loading from %s' % last_savepath)

        NUM_LOOPS = int(args.num_loops)
        NUM_ENVS = int(args.num_envs)

        for i in range(NUM_WORKERS):
            self.workers.append(Worker(self.tasks, self.results, i))

        for w in self.workers:
            w.start()

        pid = os.getpid()
        py = psutil.Process(pid)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            model = ppo2.Model(policy=policies.CnnPolicy,
                               ob_space=env.observation_space,
                               ac_space=env.action_space,
                               nbatch_act=nbatch_act,
                               nsteps=steps_per_ep,
                               nbatch_train=nbatch_train,
                               ent_coef=ent_coef,
                               vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm)
            if last_savepath:
                model.load(last_savepath)
                num = re.search('\d+$', last_savepath)[0]
                last_savepath = int(num) + 1
            else:
                last_savepath = 1

            last_savepath = osp.join(checkdir, str(last_savepath))

            with tf.variable_scope('model'):
                params = model.get_params()

            print("Loading params for workers")

            for w in self.workers:
                self.tasks.put((LOAD_SIGNAL, params))
            self.tasks.join() # block

            for step in range(NUM_LOOPS):
                exps = []
                for i in range(NUM_ENVS):
                    self.tasks.put((TRAIN_SIGNAL, i))
                self.tasks.join() # block

                print("step %d completed" % step)

                while not self.results.empty():
                    exps.append(self.results.get())
                assert len(exps) == NUM_ENVS
                nbatch = steps_per_ep
                inds = np.arange(nbatch)
                grads = []
                for _ in range(noptepochs):
                    np.random.shuffle(inds)
                    for start in range(0, nbatch, nbatch_train):
                        mbinds = inds[start:start + nbatch_train]
                        all_slices = []
                        for exp in exps:
                            obs, returns, masks, actions, values, neglogpacs, states, epinfos = exp
                            slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            all_slices.append(slices)
                            grads.append(model.grad(lr, cliprange, *slices))

                print("Done running workers on all envs, now merging gradients")

                avg_grads = []
                # for each gradient variable (NOT gradient colleted from experience),
                for i in range(len(grads[0])):
                    # pool together the grads i from each worker j
                    total_grad = grads[0][i]
                    for j in range(1, len(grads)):
                        total_grad += grads[j][i]
                    avg_grads.append(total_grad / len(grads))

                print("Finished merging gradients, now applying")
                model.joint_train2(lr, avg_grads)

                params = model.get_params()

                for w in self.workers:
                    self.tasks.put((LOAD_SIGNAL, params))
                self.tasks.join()

                memUse = py.memory_info()[0]/2.**30
                print('memory use: %.6f GB from master' %(memUse))

            model.save(last_savepath)

        print("sending kill signal to workers")
        for w in self.workers:
            self.tasks.put((KILL_SIGNAL, DUMMY_ENV))

        print("wrap up")
        self.tasks.join()
        while not self.results.empty():
            print(self.results.get())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Joint PPO')
    parser.add_argument('params_folder', default='params')
    parser.add_argument('--num_loops', help='number of iterations', default = 100)
    parser.add_argument('--num_envs', help='number of levels to train', default=47)
    parser.add_argument('--workers', help='number of worker processes', default=4)
    parser.add_argument('--last_save_params', action='store_true')
    args = parser.parse_args()
    master = Master(args)
