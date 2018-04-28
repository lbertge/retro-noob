from retro_contest.local import make

def main():
    env = make(game='CrackDown-Genesis', state='LabyrinthZone.Act1')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        if done:
            obs = env.reset()
            print("episode complete")


if __name__ == '__main__':
    main()

