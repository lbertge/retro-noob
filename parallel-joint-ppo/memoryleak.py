import ppo2
import tensorflow as tf
from retro_contest.local import make
from sonic_util import make_local_env
import numpy as np
import baselines.ppo2.policies as policies
import os
import psutil
import pandas as pd
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from functools import partial

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

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

env = make_local_env(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', stack=True, scale_rew=True)
env.close()

train_data = pd.read_csv('../sonic-train.csv')
levels = []
for index, level in train_data.iterrows():
    levels.append((level.game, level.state))

with tf.Session(config=config):
    model = ppo2.Model(policy=policies.CnnPolicy,
                   ob_space=env.observation_space,
                   ac_space=env.action_space,
                   nbatch_act=nbatch_act,
                   nsteps=steps_per_ep,
                   nbatch_train=nbatch_train,
                   ent_coef=ent_coef,
                   vf_coef=vf_coef,
                   max_grad_norm=max_grad_norm)

    # obs = np.zeros((1,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
    for i in range(47):
        level = levels[i]
        print(level[0], level[1])
        env = make_local_env(level[0], level[1], True, True)
        def env_fn():
            return env
        runner = ppo2.Runner(
                        env=DummyVecEnv([env_fn]),
                        num_envs=1,
                        model=model,
                        nsteps=steps_per_ep,
                        gamma=gamma,
                        lam=lam,
                        lr=lr,
                        cliprange=cliprange,
                        noptepochs=noptepochs,
                        nbatch_train=nbatch_train)

        exp = runner.run()
        env.close()
        pid = os.getpid()
        py = psutil.Process(pid)
        memUse = py.memory_info()[0]/2.**30
        print('memory use: %.6f GB' % memUse)
