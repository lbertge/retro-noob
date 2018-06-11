import os
import time
import joblib
import numpy as np
import os.path as osp
import os
import psutil
import tensorflow as tf
from baselines.common.runners import AbstractEnvRunner

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 9999999, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, 9999999, reuse=True)

        A = train_model.pdtype.sample_placeholder([None], name='A')
        ADV = tf.placeholder(tf.float32, [None], name='ADV')
        R = tf.placeholder(tf.float32, [None], name='R')
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], name='OLDNEGLOG')
        OLDVPRED = tf.placeholder(tf.float32, [None], name='OLDVPRED')
        LR = tf.placeholder(tf.float32, [], name='LR')
        CLIPRANGE = tf.placeholder(tf.float32, [], name='CLIPRANGE')

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        with tf.variable_scope('grads', reuse=False):
            grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        new_grads = []
        restores = []
        for i in range(len(params)):
            # set up placeholder variables when assigning new grads
            tmp = tf.Variable(params[i])
            tmp.assign(grads[i])
            new_grads.append(tmp)

            # also set up placeholder variables when assigning new parameters
            loaded_param = tf.placeholder(params[i].dtype, params[i].get_shape(), 'PARAM' + str(i))
            restores.append(params[i].assign(loaded_param))

        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(list(zip(new_grads, params)))

        def grad(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                new_grads,
                td_map
            )

        def joint_train(lr, avg_grads):
            new_grads = []
            for avg_g in avg_grads:
                new_grads.append(tf.Variable(avg_g))

            new_grads = list(zip(new_grads, params))
            trainer = tf.train.AdamOptimizer(learning_rate=2e-4, epsilon=1e-5)
            _train = trainer.apply_gradients(new_grads)
            tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101
            sess.run(_train, {LR:lr})

        def joint_train2(lr, avg_grads):
            for i in range(len(avg_grads)):
                new_grads[i].load(avg_grads[i])
            print("Done assigning gradients")
            # tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101
            sess.run(_train, {LR:lr})

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        def get_params():
            return sess.run(params)

        def load_ram(loaded_params):
            param_map = {}
            for i in range(len(loaded_params)):
                param_map['model/PARAM' + str(i) + ':0'] = loaded_params[i]
            sess.run(restores, param_map)

        self.train = train
        self.joint_train = joint_train
        self.joint_train2 = joint_train2
        self.grad = grad
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        self.load_ram = load_ram
        self.get_params = get_params
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

class Runner(AbstractEnvRunner):

    def __init__(self, *, env, num_envs, model, nsteps, gamma, lam, lr, cliprange, noptepochs, nbatch_train):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma
        self.lr = lr
        self.cliprange = cliprange
        self.noptepochs = noptepochs
        self.nbatch_train = nbatch_train
        self.num_envs = num_envs

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        rewards = None
        done = None
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)

    def get_grads(self):
        mb_grads = []

        for i in range(self.num_envs):
            obs, returns, masks, actions, values, neglogpacs, states, epinfos = self.run()
            nbatch = self.nsteps # since only 1 env
            inds = np.arange(nbatch)
            for _ in range(self.noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, self.nbatch_train):
                    mbinds = inds[start:start + self.nbatch_train]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mb_grads.append(self.model.grad(self.lr,
                                                    self.cliprange,
                                                    *slices))

        return mb_grads

# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
