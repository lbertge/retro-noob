import gym
from retro_contest.local import make

def eval():
    env = make(game="SonicTheHedgehog-Genesis", state='GreenHillZone.Act1')
    env.reset()
    # Evaluate on the env
    env.close()

pop = [eval() for _ in range(10000)]

