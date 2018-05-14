import pandas as pd
import ppo2_local as ppo2
import tensorflow as tf
import argparse

TIMESTEPS_PER_GAME = 1e6
SAVE_INTERVAL = 50

def run(base_dir):
    df = pd.read_csv('../sonic-train.csv')
    df2 = pd.read_csv('../sonic-validation.csv')

    last_dir = None

    for index, row in df.iterrows():
        print("Playing ", row.game, row.state)
        tf.reset_default_graph()
        if index == df.shape[0] - 1:
            params_folder = base_dir + '/final'
        else:
            params_folder = base_dir + row.game + row.state
        ppo2.main(row.game, row.state, TIMESTEPS_PER_GAME, save_interval = 10, last_dir=last_dir, params_folder=params_folder)
        last_dir = base_dir + '/' + row.game + row.state

    for index, row in df2.iterrows():
        print("Playing ", row.game, row.state)
        tf.reset_default_graph()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    args = parser.parse_args()
    run(base_dir = args.results_dir)

if __name__ == '__main__':
    main()
