import argparse
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

from thrifty.algos.thriftydagger import ReplayBuffer, QReplayBuffer
MUS = [0.0, 1.0, 5.0, 50.0, 100.0]
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='robosuite-30.pkl')
    parser.add_argument('--out_dir', type=str, default='./data/test_uncertainty')
    parser.add_argument('--std', type=float, default=0.1)
    return parser.parse_args()

def main(args):

    for dir in os.listdir(args.out_dir):
        if f'std{args.std}' in dir or 'train' in dir:
            for file in os.listdir(os.path.join(args.out_dir, dir)):
                with open(os.path.join(args.out_dir, dir, file), 'rb') as f:
                    metrics = pickle.load(f)
                    uncertaintys = metrics['uncertaintys']
                    label = dir[12:] if 'mu' in dir else 'no noise'
                    plt.hist(uncertaintys, bins=20, alpha=0.7, label=f'{label}')
    plt.legend()
    plt.savefig(os.path.join(f'uncertainty_plot_std{args.std}.png'))

    # plt.title('Uncertainty')
    # save_dir = os.path.join(args.out_dir, args.exp_name)
    # os.makedirs(save_dir)
    # save_file = os.path.join(os.path.join(save_dir, 'uncertainty_metrics.pkl'))
    # with open(save_file, 'wb') as f:
    #     pickle.dump(res, f)

if __name__ == '__main__':
    args = parse_args()
    main(args)