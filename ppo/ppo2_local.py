#!/usr/bin/env python

"""
Train an agent on Sonic using PPO2 from OpenAI Baselines.
"""

import tensorflow as tf
from retro_contest.local import make
import numpy as np

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.atari_wrappers import WarpFrame, FrameStack
from baselines import logger
import baselines.ppo2.ppo2 as ppo2
import baselines.ppo2.policies as policies
from baselines.deepq import utils
import gym_remote.exceptions as gre

from sonic_util import make_env, SonicDiscretizer

def main():
    """Run PPO until the environment throws an exception."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    env = SonicDiscretizer(env)
    env = WarpFrame(env)
    env = FrameStack(env, 4)
    log = logger.Logger('ppo_v4.1/', ['stdout'])
    logger.Logger.CURRENT = log

    print(logger.get_dir())

    def env_fn():
        return env

    tmpEnv = DummyVecEnv([env_fn])

    print(tmpEnv.num_envs)
    print(tmpEnv.observation_space)
    print(tmpEnv.action_space)

    with tf.Session(config=config):
        Take more timesteps than we need to be sure that
        we stop due to an exception.
        ppo2.learn(policy=policies.CnnPolicy,
                   env=DummyVecEnv([env_fn]),
                   nsteps=4096,
                   nminibatches=8,
                   lam=0.95,
                   gamma=0.99,
                   noptepochs=3,
                   log_interval=1,
                   ent_coef=0.01,
                   lr=lambda _: 2e-4,
                   cliprange=lambda _: 0.1,
                   total_timesteps=int(1e7),
                   save_interval=100)
        # utils.save_state('/home/noob/retro-noob/ppo/params')

        # model = ppo2.Model(policy=policies.CnnPolicy,
                   # ob_space=tmpEnv.observation_space,
                   # ac_space=tmpEnv.action_space,
                   # nbatch_act=1,
                   # nsteps=4096,
                   # nbatch_train=4096 // 4,
                   # ent_coef=0.01,
                   # vf_coef=0.5,
                   # max_grad_norm=0.5)

        # print(tf.trainable_variables())
        # model2 = utils.load_state('/home/noob/retro-noob/ppo/params')


if __name__ == '__main__':
    main()
