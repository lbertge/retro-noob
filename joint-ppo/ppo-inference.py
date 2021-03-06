import tensorflow as tf
import baselines.ppo2.ppo2 as ppo2
import baselines.ppo2.policies as policies
import argparse
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from retro_contest.local import make

from sonic_util import make_local_env

def run(game, state, params_dir):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    env = make_local_env(game=game, state=state, stack=True, scale_rew=True)

    load_path = 'params_3/checkpoints/00151'

    def env_fn():
        return env

    with tf.Session(config=config):
        model = ppo2.Model(policy = policies.CnnPolicy,
                           ob_space = env.observation_space,
                           ac_space = env.action_space,
                           nbatch_act = 1,
                           nsteps = 4500,
                           nbatch_train = 4500 // 4,
                           ent_coef=0.01,
                           vf_coef=0.5,
                           max_grad_norm=0.5)

        print(env.observation_space)
        print(env.action_space)
        model.load(load_path)
        runner = ppo2.Runner(env=DummyVecEnv([env_fn]), model=model, nsteps=4500, gamma=0.99, lam=0.95)
        runner.run()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("game")
    parser.add_argument("state")
    parser.add_argument("params_dir")
    args = parser.parse_args()
    run(game=args.game, state=args.state, params_dir=args.params_dir)


if __name__ == '__main__':
    main()
