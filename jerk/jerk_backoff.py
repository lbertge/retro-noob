#!/usr/bin/env python

"""
A scripted agent called "Just Enough Retained Knowledge".
"""

import random

import gym
import numpy as np

import gym_remote.client as grc
import gym_remote.exceptions as gre

class JerkAgent():
    def __init__(self):
        self.EMA_RATE = 0.2
        self.EXPLOIT_BIAS = 0.25
        self.TOTAL_TIMESTEPS = int(1e6)
        self.env = grc.RemoteEnv('tmp/sock')
        self.env = TrackedEnv(self.env)
        self.backoff_steps = 1
        self.speed_threshold = 1

    def run(self):
        """Run JERK on the attached environment."""
        new_ep = True
        solutions = []
        while self.env.total_steps_ever < self.TOTAL_TIMESTEPS:
            if new_ep:
                if (solutions and
                        random.random() < self.EXPLOIT_BIAS + self.env.total_steps_ever / self.TOTAL_TIMESTEPS):
                    solutions = sorted(solutions, key=lambda x: np.mean(x[0]))
                    best_pair = solutions[-1]
                    new_rew = self.exploit(best_pair[1])
                    best_pair[0].append(new_rew)
                    print('replayed best with reward %f' % new_rew)
                else:
                    self.reset()
                    new_ep = False
            rew, new_ep = self.move(100)
            return
            if not new_ep and rew <= 0:
                print('backtracking due to negative reward: %f' % rew)
                self.backoff_steps *= 2
                _, new_ep = self.move(self.backoff_steps, left=True)
            if new_ep:
                solutions.append(([max(self.env.reward_history)], self.env.best_sequence()))


    def reset(self):
        self.backoff_steps = 1
        self.env.reset()

    def move(self, num_steps, left=False, jump_prob=1.0 / 10.0, jump_repeat=4):
        """
        Move right or left for a certain number of steps,
        jumping periodically.
        JUMP: 0
        DOWN: 5
        LEFT: 6
        RIGHT: 7
        """
        total_rew = 0.0
        prev_rew = 0.0
        done = False
        steps_taken = 0
        jumping_steps_left = 0
        is_moving_fast = False
        while not done and steps_taken < num_steps:
            action = np.zeros((12,), dtype=np.bool)
            action[6] = left
            action[7] = not left
            if is_moving_fast:
                action[5] = True
            if jumping_steps_left > 0:
                action[0] = True
                jumping_steps_left -= 1
            else:
                if random.random() < jump_prob:
                    jumping_steps_left = jump_repeat - 1
                    action[0] = True
            obs, rew, done, info = self.env.step(action)
            np.savetxt('obs%s' % str(steps_taken), obs, delimiter=',')
            # horizontal speed?
            is_moving_fast = abs(prev_rew - rew) >= self.speed_threshold
            prev_rew = rew
            total_rew += rew
            steps_taken += 1
            if done:
                break
        return total_rew, done

    def exploit(self, sequence):
        """
        Replay an action sequence; pad with NOPs if needed.

        Returns the final cumulative reward.
        """
        self.reset()
        for i in range(len(sequence)):
            _, _, done, _ = self.env.step(sequence[i])
        return self.env.total_reward

class TrackedEnv(gym.Wrapper):
    """
    An environment that tracks the current trajectory and
    the total number of timesteps ever taken.
    """
    def __init__(self, env):
        super(TrackedEnv, self).__init__(env)
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0
        self.total_steps_ever = 0

    def best_sequence(self):
        """
        Get the prefix of the trajectory with the best
        cumulative reward.
        """
        max_cumulative = max(self.reward_history)
        for i, rew in enumerate(self.reward_history):
            if rew == max_cumulative:
                return self.action_history[:i+1]
        raise RuntimeError('unreachable')

    # pylint: disable=E0202
    def reset(self, **kwargs):
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0
        env = self.env.reset(**kwargs)
        return env

    def step(self, action):
        self.total_steps_ever += 1
        self.action_history.append(action.copy())
        obs, rew, done, info = self.env.step(action)
        self.total_reward += rew
        self.reward_history.append(self.total_reward)
        return obs, rew, done, info

if __name__ == '__main__':
    try:
        agent = JerkAgent()
        agent.run()
    except gre.GymRemoteError as exc:
        print('exception', exc)
