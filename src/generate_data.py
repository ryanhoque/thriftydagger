from constants import REACH2D_ACT_MAGNITUDE, REACH2D_MAX_TRAJ_LEN, REACH2D_SUCCESS_THRESH
from environments import Reach2D
from util import get_model_type_and_kwargs, init_model

import argparse
import os
import pickle
import random
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Sampling parameters
    parser.add_argument('--environment', type=str, default='Reach2D', help='Name of environment for data sampling.')
    parser.add_argument('--robosuite', action='store_true', help='Whether or not the environment is a Robosuite environment')
    parser.add_argument('--N_trajectories', type=int, default=1000, help='Number of trajectories (demonstrations) to sample.')
    parser.add_argument('--add_noise', action='store_true', help='If true, noise is added to sampled states.')
    parser.add_argument('--noise_mean', type=float, default=1.0, help='Mean to use for Gaussian noise.')
    parser.add_argument('--noise_std', type=float, default=0.01, help='Std to use for Gaussian noise.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    
    # Saving
    parser.add_argument('--save_dir', default='./data', type=str, help='Directory to save the data in.')
    parser.add_argument('--save_fname', type=str, help='File name for the saved data.')
    parser.add_argument('--overwrite', action='store_true', help='If true and the save file exists already, it will be overwritten with newly-generated data.')
    
    # Arguments specific to the Reach2D environment
    parser.add_argument('--sample_mode', type=str, default='oracle', help='How to sample states/actions. Must be one of [\'oracle\', \'pi_r\',\'oracle_pi_r_mix\'].')
    parser.add_argument('--model_path', type=str, default=None, help='Model path to use as pi_r when args.sample_mode samples from pi_r.')
    parser.add_argument('--arch', type=str, default='LinearModel', help='Model architecture to use.')
    parser.add_argument('--num_models', type=int, default=1, help='Number of models in the ensemble; if 1, a non-ensemble model is used')
    parser.add_argument('--perc_oracle', type=float, default=0.8,
                        help='For use with args.sample_mode == \'oracle_pi_r_mix\' only. \
                            Percentage of oracle trajectories to use (vs. policy-sampled trajectories).')

    args = parser.parse_args()
    return args


def sample_reach(N_trajectories, range_x=3.0, range_y=3.0):
    demos = []

    for _ in range(N_trajectories):
        curr_state = torch.zeros(2)
        # Sample goal from 1st quadrant
        goal_ee_state = torch.tensor([random.uniform(0, range_x), random.uniform(0, range_y)])
        action = goal_ee_state - curr_state
        action = REACH2D_ACT_MAGNITUDE * action / torch.norm(action)
        
        obs = []
        act = []
        while (curr_state[0] < goal_ee_state[0] and curr_state[1] < goal_ee_state[1]):
            o = torch.cat([curr_state, goal_ee_state])
            obs.append(o)
            act.append(action)
            curr_state = curr_state + action
            
        success = torch.norm(o[:2] - o[2:]) <= REACH2D_SUCCESS_THRESH
        demos.append({'obs': obs, 'act': act, 'success': success.item()})
        
    return demos

def sample_pi_r(N_trajectories, model, max_traj_len, range_x=3.0, range_y=3.0, add_noise=False):
    demos = []
    
    for _ in range(N_trajectories):
        obs, act = [], []
        goal_ee_state = torch.tensor([random.uniform(0, range_x), random.uniform(0, range_y)])
        curr_state = torch.zeros(2)
        traj_len = 0
        done = False
        
        while not done and traj_len < max_traj_len:
            o = torch.cat([curr_state, goal_ee_state])
            obs.append(o.clone().detach())
            
            # Record oracle action given observation o
            a_target = goal_ee_state - curr_state
            a_target = REACH2D_ACT_MAGNITUDE * a_target / torch.norm(a_target)
            act.append(a_target.detach())
            
            # Take policy's action given observation o
            a = model(o)
            curr_state = curr_state + a
            
            if add_noise:
                max_variance = 0
                # TODO: allow different parameters for noise
                for _ in range(20):
                    noise = torch.normal(torch.zeros_like(o[:2]), std=0.5)
                    candidate = o[:2] + noise
                    if model.variance(torch.cat([candidate, o[2:]])) >= max_variance:
                        o[:2] = candidate
            
            traj_len += 1
            done = ((o[:2][0] >= o[2:][0]) and (o[:2][1] >= o[2:][1]) or torch.norm(o[:2] - o[2:]) <= REACH2D_SUCCESS_THRESH)
            
        success = torch.norm(o[:2] - o[2:]) <= REACH2D_SUCCESS_THRESH
        demos.append({'obs': obs, 'act': act, 'success': success.item()})
        
    return demos

def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print('Generating data...')
    if args.environment == 'Reach2D':
        max_traj_len = REACH2D_MAX_TRAJ_LEN
        env = Reach2D(device)
        if args.sample_mode == 'oracle':
            demos = sample_reach(args.N_trajectories)
        elif args.sample_mode  == 'pi_r':
            model_type, model_kwargs = get_model_type_and_kwargs(args, obs_dim=env.obs_dim, act_dim=env.act_dim)
            model = init_model(model_type, model_kwargs, device=device, num_models=args.num_models)
            model.to(device)
            demos = sample_pi_r(N_trajectories=args.N_trajectories, max_traj_len=max_traj_len, model=model, 
                                add_noise=args.add_noise)
        elif args.sample_mode == 'oracle_pi_r_mix':
            model_type, model_kwargs = get_model_type_and_kwargs(args, obs_dim=env.obs_dim, act_dim=env.act_dim)
            model = init_model(model_type, model_kwargs, device=device, num_models=args.num_models)
            model.to(device)
            num_oracle = int(args.perc_oracle * args.N_trajectories)
            num_pi_r = args.N_trajectories - num_oracle
            
            oracle_demos = sample_reach(num_oracle)
            pi_r_demos = sample_pi_r(N_trajectories=num_pi_r, max_traj_len=max_traj_len, model=model, 
                                add_noise=args.add_noise)
            demos = oracle_demos + pi_r_demos
        else:
            raise ValueError(f'args.sample_mode must be one of \
                             [\'oracle\', \'pi_r\',\'oracle_pi_r_mix\'] but got {args.sample_mode}!')
    else:
        raise NotImplementedError(f'Data generation for the environment \'{args.environment}\' has not been implemented yet!')
    
    print('Data generated! Saving data...')
    save_path = os.path.join(args.save_dir, args.save_fname)
    if not args.overwrite and os.path.isfile(save_path):
        raise FileExistsError(f'The file {save_path} already exists. If you want to overwrite it, rerun with the argument --overwrite.')
    os.makedirs(args.save_dir, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(demos, f)
    print(f'Data saved to {save_path}!')
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
