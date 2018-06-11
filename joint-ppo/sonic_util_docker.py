"""
Environments and wrappers for Sonic training.
"""

import gym
import numpy as np

from baselines.common.atari_wrappers import WarpFrame, FrameStack
import gym_remote.client as grc

def make_env(stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    env = grc.RemoteEnv('tmp/sock')
    env = SonicDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    if stack:
        env = FrameStack(env, 4)
    return env

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info

class JointEnv(gym.Env):
    def __init__(self, env_fns, levels):
        self.env_fns = env_fns
        self.levels = levels
        self.num_envs = len(env_fns)
        self.env_idx = 0
        self.env = env_fns[0]()
        try:
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space
        finally:
            self.env.close()

    def reset(self, **kwargs):
        if self.env is not None:
            print("Closing env")
            self.env.close()
        self.env_idx = (self.env_idx + 1) % len(self.env_fns)
        self.env = self.env_fns[self.env_idx]()
        print("Now playing %s %s" % (self.levels[self.env_idx]))
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info = info.copy()
        info['env_idx'] = self.env_idx
        return obs, rew, done, info

    def render(self, mode='human'):
        if self.env is None:
            return
        return self.env.render(mode=mode)

    def seed(self, seed=None):
        if self.env is None:
            return
        return self.env.seed(seed=seed)
