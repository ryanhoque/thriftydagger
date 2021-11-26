import argparse
import pickle
import numpy as np
import os
import torch

from thrifty.algos.thriftydagger import ReplayBuffer, QReplayBuffer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, default='./data/oct12_test_policy/oct12_test_policy_s0/pyt_save/model.pt')
    parser.add_argument('--data_file', type=str, default='robosuite-30.pkl')
    parser.add_argument('--out_dir', type=str, default='./data/test_uncertainty')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--noise_mu', type=float, default=None)
    parser.add_argument('--noise_std', type=float, default=None)
    parser.add_argument('--print_iter', type=int, default=100)
    return parser.parse_args()

def main(args):
    device = torch.device("cpu")
    obs_dim = (51,)
    act_dim = (7,)
    input_data = pickle.load(open(args.data_file, 'rb'))
    # Experience buffer
    # shuffle and create small held out set to check valid loss
    num_bc = len(input_data['obs'])
    replay_size = num_bc
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device)
    idxs = np.arange(num_bc)
    np.random.shuffle(idxs)
    replay_buffer.fill_buffer(input_data['obs'][idxs], input_data['act'][idxs])
    qbuffer = QReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device)
    qbuffer.fill_buffer_from_BC(input_data)
    
    model = torch.load(args.model_checkpoint)

    uncertaintys = []
    safetys = []
    obs_magnitudes = []
    for i, (act, obs) in enumerate(zip(replay_buffer.act_buf, replay_buffer.obs_buf)):
        obs_magnitudes.append(np.linalg.norm(obs))
        if args.noise_mu != None:
            obs += np.random.normal(loc=args.noise_mu, scale=args.noise_std, size=obs.shape)
        uncertainty = model.variance(obs)
        uncertaintys.append(uncertainty)
        safety = model.safety(obs, act)
        safetys.append(safety)
        if i % args.print_iter == 0:
            print(f'=======Iteration {i}/{len(replay_buffer.act_buf)}=======')
            print(f'Uncertainty: {uncertainty}')
            print(f'Safety: {safety}\n')
    res = {
        'uncertaintys': uncertaintys,
        'safetys': safetys,
        'obs_magnitudes': obs_magnitudes,
        'avg_obs_magnitude': np.mean(obs_magnitudes)
    }   
    save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(save_dir)
    save_file = os.path.join(os.path.join(save_dir, 'uncertainty_metrics.pkl'))
    with open(save_file, 'wb') as f:
        pickle.dump(res, f)

if __name__ == '__main__':
    args = parse_args()
    main(args)