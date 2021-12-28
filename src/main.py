from algos import BC, Dagger, HGDagger
from constants import PICKPLACE_MAX_TRAJ_LEN, REACH2D_MAX_TRAJ_LEN
from datasets.util import get_dataset
from datetime import datetime
from environments import Reach2D
from util import get_model_type_and_kwargs, init_model, setup_robosuite


import argparse
import numpy as np
import os
import random
import torch

def parse_args():
    parser = argparse.ArgumentParser()

    # Logging + output
    parser.add_argument('--exp_name', type=str, default=None, help='Unique experiment ID for saving/logging purposes. If not provided, date/time is used as default.')
    parser.add_argument('--out_dir', type=str, default='./out', help='Parent output directory. Files will be saved at /\{args.out_dir\}/\{args.exp_name\}.')
    parser.add_argument('--overwrite', action='store_true', help='If provided, the save directory will be overwritten even if it exists already.')
    parser.add_argument('--save_iter', type=int, default=5, help='Checkpoint will be saved every args.save_iter epochs.')
    
    # Data generation / loading
    parser.add_argument('--data_path', type=str, default='./data/dec14_gen_oracle_reach2d_data_1k/dec14_gen_oracle_reach2d_data_1k_s4/pick-place-data-1000.pkl')
    parser.add_argument('--N', type=int, default=1000, help='Size of dataset.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    # Autonomous evaluation only
    parser.add_argument('--eval_only', action='store_true', help='If true, rolls out the autonomous policy of the provided trained model')
    parser.add_argument('--model_path', type=str, default=None, help='Path to saved model checkpoint for evaulation purposes.')
    parser.add_argument('--N_eval_trajectories', type=int, default=100, help='Number of trajectories to roll out for autonomous-only evaluation.')
    
    # Environment details + rendering
    parser.add_argument('--environment', type=str, default='Reach2D', help='Environment name')
    parser.add_argument('--robosuite', action='store_true', help='Whether or not the environment is a Robosuite environment')
    parser.add_argument('--no_render', action='store_true', help='If true, Robosuite rendering is skipped.')

    # Method / Model details
    parser.add_argument('--method', type=str, required=True, help='One of \{BC, Dagger, ThriftyDagger, HGDagger, LazyDagger\}}')
    parser.add_argument('--arch', type=str, default='LinearModel', help='Model architecture to use.')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size of MLP if args.arch == \'MLP\'')
    parser.add_argument('--num_models', type=int, default=1, help='Number of models in the ensemble; if 1, a non-ensemble model is used')
    
    # Dagger-specific parameter beta
    parser.add_argument('--dagger_beta', type=float, default=0.9, help='DAgger parameter; policy will be (beta * expert_action) + (1-beta) * learned_policy_action')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5, help='Number of iterations to run overall method for')
    parser.add_argument('--policy_train_epochs', type=int, default=5, help='Number of epochs to run when training the policy (for interactive methods only).')
    parser.add_argument('--trajectories_per_rollout', type=int, default=10, help='Number of trajectories to roll out per epoch, required for interactive methods and ignored for offline data methods.')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set up output directories
    if args.exp_name is None:
        args.exp_name = datetime.now().strftime('%m-%d-%Y-%H:%M:%S')
    save_dir = os.path.join(args.out_dir, args.exp_name)
    if not args.overwrite and os.path.isdir(save_dir):
        raise FileExistsError(f'The directory {save_dir} already exists. If you want to overwrite it, rerun with the argument --overwrite.')
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up environment
    if args.robosuite:
        if args.environment == 'PickPlace':
            max_traj_len = PICKPLACE_MAX_TRAJ_LEN
        else:
            raise NotImplementedError(f'Max trajectory length not yet defined for the environment {args.environment}!')
        env, robosuite_cfg = setup_robosuite(args, max_traj_len)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0] 
        act_limit = env.action_space.high[0] 
    elif args.environment == 'Reach2D':
        max_traj_len = REACH2D_MAX_TRAJ_LEN
        env = Reach2D(device)
        robosuite_cfg = None
        obs_dim = env.obs_dim
        act_dim = env.act_dim
        act_limit = float('inf')
    else:
        raise NotImplementedError(f'Environment {args.environment} has not been implemented yet!')
        
    
    # Initialize model
    model_type, model_kwargs = get_model_type_and_kwargs(args, obs_dim, act_dim)
    model = init_model(model_type, model_kwargs, device=device, num_models=args.num_models)
    model.to(device)
        
    # Load model checkpoint if in eval_only mode
    if args.eval_only:
        model.eval()
        ckpt = torch.load(args.model_path)
        if args.num_models > 1:
            for ensemble_model, state_dict in zip(model.models, ckpt['models']):
                ensemble_model.load_state_dict(state_dict)
        else:
            model.load_state_dict(ckpt['model'])
        
    # Set up method
    if args.method == 'Dagger':
        algorithm = Dagger(model, model_kwargs, device=device, save_dir=save_dir, 
                           max_traj_len=max_traj_len, beta=args.dagger_beta)
    elif args.method == 'HGDagger':
        algorithm = HGDagger(model, model_kwargs, device=device, save_dir=save_dir, 
                             max_traj_len=max_traj_len)
    elif args.method == 'BC':
        algorithm = BC(model, model_kwargs, device=device, save_dir=save_dir, 
                       max_traj_len=max_traj_len)
    else:
        raise NotImplementedError(f'Method {args.method} has not been implemented yet!')
    
    # Run algorithm    
    if args.eval_only:
        algorithm.eval_auto(args, env=env, robosuite_cfg=robosuite_cfg)
    else:
        train, val = get_dataset(args.data_path, args.N)
        algorithm.run(train, val, args, env=env, robosuite_cfg=robosuite_cfg)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
