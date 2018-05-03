from retro_contest.local import make

import numpy as np

def main():
    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    obs = env.reset()
    counter = 0
    while counter < 100:
        obs, rew, done, info = env.step(env.action_space.sample())
        print("saving %d" % counter)
        np.save('obs%s' % str(counter), obs)
        counter += 1
        if done:
            obs = env.reset()
            print("episode complete")


if __name__ == '__main__':
    main()

