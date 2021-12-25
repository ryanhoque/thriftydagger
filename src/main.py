from algos import BC, Dagger, HGDagger
from datasets.util import get_dataset
from datetime import datetime
from environments import CustomWrapper, Reach2D
from models import Ensemble
from robosuite import load_controller_config
from robosuite.devices import Keyboard
from robosuite.wrappers import GymWrapper, VisualizationWrapper
from util import get_model_type_and_kwargs, MAX_PICKPLACE_TRAJ_LEN, MAX_REACH2D_TRAJ_LEN

import argparse
import numpy as np
import os
import random
import robosuite as suite
import torch

ENVS = ['NutAssembly', 'Reach2D', 'PickPlace']
ARCHS = ['LinearModel', 'MLP']

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

def setup_robosuite(args, max_traj_len):
    render = not args.no_render
    controller_config = load_controller_config(default_controller='OSC_POSE')
    config = {
        "env_name": args.environment,
        "robots": "UR5e",
        "controller_configs": controller_config,
    }

    if args.environment == 'NutAssembly':
        env = suite.make(
            **config,
            has_renderer=render,
            has_offscreen_renderer=False,
            render_camera="agentview",
            single_object_mode=2, # env has 1 nut instead of 2
            nut_type="round",
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            hard_reset=True,
            use_object_obs=True
        )
    elif args.environment == 'PickPlace':
            env = suite.make(
                **config,
                has_renderer=render,
                has_offscreen_renderer=False,
                render_camera="agentview",
                single_object_mode=2,
                object_type='cereal',
                ignore_done=True,
                use_camera_obs=False,
                reward_shaping=True,
                control_freq=20,
                hard_reset=True,
                use_object_obs=True
            )
    else:
        env = suite.make(
            **config,
            has_renderer=render,
            has_offscreen_renderer=False,
            render_camera="agentview",
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            hard_reset=True,
            use_object_obs=True
        )

    env = GymWrapper(env)
    env = VisualizationWrapper(env, indicator_configs=None)
    env = CustomWrapper(env, render=render)

    input_device = Keyboard(pos_sensitivity=0.5, rot_sensitivity=3.0)

    if render:
        env.viewer.add_keypress_callback("any", input_device.on_press)
        env.viewer.add_keyup_callback("any", input_device.on_release)
        env.viewer.add_keyrepeat_callback("any", input_device.on_press)

    arm_ = 'right'
    config_ = 'single-arm-opposed'
    active_robot = env.robots[arm_ == 'left']
    robosuite_cfg = {
        'max_ep_length': max_traj_len, 
        'input_device': input_device,
        'arm': arm_,
        'env_config': config_,
        'active_robot': active_robot
        }

    return env, robosuite_cfg

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
    if args.environment == 'Reach2D': 
        max_traj_len = MAX_REACH2D_TRAJ_LEN
    elif args.environment == 'PickPlace':
        max_traj_len = MAX_PICKPLACE_TRAJ_LEN
    else:
        raise NotImplementedError(f'Max trajectory length not yet defined for the environment {args.environment}!')
        
    if args.robosuite:
        env, robosuite_cfg = setup_robosuite(args, max_traj_len)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0] 
        act_limit = env.action_space.high[0] 
    else:
        # TODO: have config files for non-robosuite environments
        env = Reach2D()
        robosuite_cfg = None
        obs_dim = 4
        act_dim = 2
        act_limit = float('inf')
        
    
    # Initialize model
    model_type, model_kwargs = get_model_type_and_kwargs(args, obs_dim, act_dim)
    
    if args.num_models > 1:
        model_kwargs = dict(model_kwargs=model_kwargs, device=device, 
                            num_models=args.num_models, model_type=model_type)
        model = Ensemble(**model_kwargs)
    elif args.num_models == 1:
        model = model_type(**model_kwargs)
    else:
        raise ValueError(f'Got {args.num_models} for args.num_models, but value must be an integer >= 1!')
        
    # Load model if in eval_only mode
    if args.eval_only:
        model.eval()
        ckpt = torch.load(args.model_path)
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